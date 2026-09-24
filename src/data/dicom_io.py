"""RSNA Knee — DICOM okuma ve on isleme katmani (Faz 0.5-0.9).

Bu modul bir study'nin tek bir serisini alip modele verilebilir bir tensore
cevirene kadarki her adimi icerir:

    seri klasoru
      -> dosyalari oku (pydicom)
      -> slice'lari sirala            (0.5, sort_datasets)
      -> piksel cikar + rescale       (0.6, extract_pixels)
      -> intensite normalize et       (0.6, normalize_volume)
      -> lateraliteyi normalize et    (0.7, detect_laterality + apply_lr_mirror)
      -> yeniden boyutlandir          (resize_volume)
      -> 2.5D pencere cikar           (0.9, make_25d)

Tasarim notu: fonksiyonlar tek tek test edilebilsin diye ayri tutuldu;
`load_series()` hepsini birlestiren pratik giristir.

Faz 0 envanterine dayanan varsayimlar (bkz. PHASE0_FINDINGS.md):
  - ImagePositionPatient / ImageOrientationPatient %100 dolu -> siralama guvenilir
  - Laterality sadece %48.8 dolu -> katmanli heuristic gerekli
  - Transfer syntax %100 sikistirilmamis -> harici decoder zorunlu degil
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pydicom

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# =====================================================================
# 0. Kaggle ortam yardimcilari
# =====================================================================

def safe_listdir(path) -> list:
    """os.listdir, ama FUSE mount'unun firlattigi her hatayi bos listeye cevirir.

    /kaggle/input sirali bir disk degil, ag uzerinden bagli bir dosya sistemi;
    pathlib'in normalde yutup False dondugu bazi hatalar burada istisna olarak
    firlayabiliyor.
    """
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def find_data_root(slug: str = "rsna-knee-abnormality-detection",
                   marker: str = "train_series.csv") -> str:
    """Kaggle'in iki farkli mount yerlesimini de destekleyen veri koku bulucu."""
    base = "/kaggle/input"
    for cand in (base + "/" + slug, base + "/competitions/" + slug):
        if marker in safe_listdir(cand):
            return cand
    skip = {"train_series", "test_series"}
    level1 = [base + "/" + n for n in safe_listdir(base)]
    for p in level1:
        if marker in safe_listdir(p):
            return p
    for p in level1:
        for n in safe_listdir(p):
            if n in skip:
                continue
            q = p + "/" + n
            if marker in safe_listdir(q):
                return q
    raise FileNotFoundError(marker + " iceren bir veri koku bulunamadi (" + base + " altinda)")


def series_dir(data_root: str, study_uid: str, series_uid: str, split: str = "train") -> str:
    return data_root + "/" + split + "_series/" + study_uid + "/" + series_uid


def list_dicom_files(sdir: str) -> list:
    """Bir seri klasorundeki dosya yollari. Uzanti filtreleme toleransli."""
    names = safe_listdir(sdir)
    dcm = [n for n in names if n.lower().endswith(".dcm")]
    names = dcm if dcm else names
    return [sdir + "/" + n for n in names]


# =====================================================================
# 1. Okuma
# =====================================================================

def read_datasets(paths: Sequence[str], workers: int = 8,
                  headers_only: bool = False) -> tuple:
    """Dosyalari paralel oku. (datasets, errors) dondurur.

    Kaggle'da darbogaz CPU degil AG GECIKMESI -> thread'ler gercekten ise yariyor
    (I/O bekleyen thread GIL'i birakir).
    """
    def _read(p):
        try:
            return pydicom.dcmread(p, stop_before_pixels=headers_only), None
        except Exception as e:
            return None, os.path.basename(p) + ": " + type(e).__name__ + ": " + str(e)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_read, paths))
    datasets = [d for d, _ in results if d is not None]
    errors = [e for _, e in results if e is not None]
    return datasets, errors


# =====================================================================
# 2. FAZ 0.5 — Slice siralama
# =====================================================================

def slice_normal(iop):
    """IOP'den slice normal vektoru.

    IOP = [satir_yonu(3), sutun_yonu(3)]. Bu iki vektorun capraz carpimi
    duzleme dik olan ucuncu ekseni verir — yani slice'larin dizildigi yonu.
    """
    if iop is None or len(iop) != 6:
        return None
    row = np.asarray(iop[:3], dtype=np.float64)
    col = np.asarray(iop[3:], dtype=np.float64)
    n = np.cross(row, col)
    norm = np.linalg.norm(n)
    return None if norm < 1e-6 else n / norm


def canonical_normal(n: np.ndarray) -> np.ndarray:
    """Normal vektorun isaretini deterministik hale getir.

    Capraz carpimin yonu IOP'nin nasil yazildigina bagli; ayni anatomik duzlem
    icin +n veya -n gelebilir. Bu, ayni seriyi bazen bas-ayak ters siralamak
    demektir. En buyuk mutlak bileseni pozitif yapmak bu belirsizligi kaldirir:
    ayni duzlemdeki her seri hep ayni yonde siralanir.
    """
    k = int(np.argmax(np.abs(n)))
    return -n if n[k] < 0 else n


def slice_sort_keys(datasets: Sequence) -> tuple:
    """Her dataset icin siralama anahtari. (keys, method) dondurur.

    Birincil yontem: slice pozisyonunun normal vektor uzerine izdusumu.
    Neden InstanceNumber degil: InstanceNumber kayit sirasidir, uzamsal sira
    degil. Multi-echo, yeniden yapilandirilmis veya kesintiye ugrayip devam
    ettirilmis serilerde ikisi ayrisir. Fiziksel konum yalan soylemez.
    """
    iops = [ds.get("ImageOrientationPatient", None) for ds in datasets]
    ipps = [ds.get("ImagePositionPatient", None) for ds in datasets]

    normals = [slice_normal(iop) for iop in iops]
    usable = [n is not None for n in normals]
    have_pos = [ipp is not None and len(ipp) == 3 for ipp in ipps]

    if all(usable) and all(have_pos):
        # Seri icinde oryantasyon sabit kabul edilir; ortalama normali kullanmak
        # kucuk yuvarlama farklarina karsi dayanikli hale getirir.
        n = canonical_normal(np.mean(np.stack(normals), axis=0))
        keys = np.array([float(np.dot(np.asarray(ipp, dtype=np.float64), n))
                         for ipp in ipps])
        return keys, "ImagePositionPatient"

    inst = [ds.get("InstanceNumber", None) for ds in datasets]
    if all(i is not None for i in inst):
        return np.array([float(i) for i in inst]), "InstanceNumber"

    return np.arange(len(datasets), dtype=float), "dosya_adi"


def sort_datasets(datasets: Sequence) -> tuple:
    """Datasets'i uzamsal siraya diz. (sirali_datasets, bilgi) dondurur."""
    keys, method = slice_sort_keys(datasets)
    order = np.argsort(keys, kind="stable")
    ordered = [datasets[i] for i in order]
    sk = keys[order]

    gaps = np.diff(sk)
    info = {
        "method": method,
        "n": len(datasets),
        "n_unique_pos": int(len(np.unique(np.round(sk, 4)))),
        "spacing_median": float(np.median(gaps)) if len(gaps) else float("nan"),
        "spacing_cv": (float(np.std(gaps) / (abs(np.mean(gaps)) + 1e-9))
                       if len(gaps) else float("nan")),
    }
    # Ayni pozisyonda birden fazla slice = multi-echo veya tekrarli kayit.
    info["duplicate_positions"] = info["n"] - info["n_unique_pos"]
    return ordered, info


def plane_from_iop(iop) -> str:
    """Anatomik duzlem adi. Normalin baskin ekseni duzlemi belirler."""
    n = slice_normal(iop)
    if n is None:
        return "Unknown"
    return ["Sagittal", "Coronal", "Axial"][int(np.argmax(np.abs(n)))]


# =====================================================================
# 3. FAZ 0.6 — Piksel cikarimi ve intensite normalizasyonu
# =====================================================================

def extract_pixels(datasets: Sequence) -> tuple:
    """Datasets'ten (N, H, W) float32 hacim cikar. (vol, flags) dondurur.

    RescaleSlope/Intercept UYGULANIR. Sebep mutlak deger degil, SERI ICI
    TUTARLILIK: ayni serinin slice'lari farkli slope ile kaydedilmisse, slope
    uygulanmadan yapilan seri-duzeyi normalizasyonu slice'lar arasinda sahte
    parlaklik farki uretir. (Tum slice'lar ayni slope'a sahipse pozitif-afin
    donusum percentile+min-max altinda zaten etkisizdir — yani bu adimin
    maliyeti sifir, faydasi bazen buyuk.)

    MONOCHROME1 tespit edilirse isaretlenir; ters cevirme normalizasyondan
    sonra yapilir (monoton donusum oldugu icin sonuc ayni, ama tek bir yerde).
    """
    from collections import Counter

    arrs, shapes, mono1 = [], set(), False
    for ds in datasets:
        a = ds.pixel_array.astype(np.float32)
        slope = ds.get("RescaleSlope", None)
        inter = ds.get("RescaleIntercept", None)
        if slope is not None or inter is not None:
            a = a * float(slope if slope is not None else 1.0) \
                + float(inter if inter is not None else 0.0)
        if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
            mono1 = True
        arrs.append(a)
        shapes.add(a.shape)

    if len(shapes) > 1:
        # Seri icinde boyut degisiyor (nadir ama olur). En sik boyutta kalanlari tut.
        target = Counter(a.shape for a in arrs).most_common(1)[0][0]
        keep = [i for i, a in enumerate(arrs) if a.shape == target]
        arrs = [arrs[i] for i in keep]
    else:
        keep = list(range(len(arrs)))

    vol = np.stack(arrs) if arrs else np.zeros((0, 1, 1), dtype=np.float32)
    return vol, {"monochrome1": mono1,
                 "mixed_shapes": len(shapes) > 1,
                 "kept_idx": keep}


def normalize_volume(vol: np.ndarray, p_low: float = 1.0, p_high: float = 99.0,
                     per_slice: bool = False, invert: bool = False) -> np.ndarray:
    """Percentile clip + min-max -> [0, 1] float32.

    Neden percentile, neden sabit bir aralik degil:
    MR'da CT'deki Hounsfield gibi mutlak bir olcek YOKTUR. Ayni dokunun sinyal
    degeri scanner'a, sekansa, bobin mesafesine gore degisir. Tek anlamli
    referans serinin kendi dagilimidir.

    Neden %1-%99, neden min-max degil:
    Tek bir parlak artefakt (metal implant, yag sinyali, bozuk piksel) min-max'i
    tek basina ele gecirir ve dokunun tamami gri bir serit haline sikisir.
    Percentile bu uclari keser.

    per_slice=False (varsayilan) ONEMLI bir tercih: percentile'lar TUM hacim
    uzerinden hesaplanir. Boylece slice'lar arasi gercek parlaklik farklari
    korunur — efuzyon iceren slice parlak kalir. per_slice=True her slice'i
    kendi icinde gerdiginden bu sinyali silebilir.
    """
    if vol.size == 0:
        return vol.astype(np.float32)

    def _scale(x):
        lo, hi = np.percentile(x, [p_low, p_high])
        if hi <= lo:                      # duz goruntu (bos slice, sabit deger)
            lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
        return ((np.clip(x, lo, hi) - lo) / (hi - lo)).astype(np.float32)

    out = (np.stack([_scale(s) for s in vol]) if per_slice else _scale(vol))
    if invert:
        out = 1.0 - out
    return out


# =====================================================================
# 4. FAZ 0.7 — Lateralite
# =====================================================================

# Cok dilli sol/sag anahtar kelimeleri. SeriesDescription genelde Ingilizce'dir
# ama veri 12 dilli, o yuzden yaygin varyantlari da koyduk.
#
# Iki ayri eslesme turu var, cunku tek tur ikisini birden bozuyordu:
#
#   EXACT  — kisa kisaltmalar. Tam kelime olarak eslesmeli, yoksa "lateral"
#            icindeki "l" veya "rodilla" icindeki harfler tetikler.
#   PREFIX — uzun, ayirt edici kokler. Dil cekimlerini yakalamak icin sonuna
#            ne gelirse gelsin eslesir: "link" -> links / linke / linkes / linken.
#
# CIKARILANLAR (ozellikle dikkat):
#   "sag" (TR "sag")  -> diz MR'inda SAG neredeyse her zaman SAGITTAL demek.
#                        Birakirsak "SAG PD FS LEFT KNEE" hem sol hem sag
#                        eslesir, fonksiyon kararsiz kalir ve Ingilizce
#                        serilerin cogunda lateralite sessizce kaybolur.
#   "der" (DE/ES)     -> Almanca'da yaygin bir artikel. "derecha" zaten
#                        "derech" onekiyle yakalaniyor.
#   "sin" (IT)        -> "sinus" gibi kelimelerle carpisir; "sinistr" oneki yeter.
#
# Asimetri kasitli: TR "sag" cikarildigi icin "SAG DIZ" hicbir sey dondurmez
# (None). Yanlis taraf uretmektense karar vermemek yeglenir — yanlis aynalama
# modeli sistematik olarak zehirler, karar vermemek sadece bir sinyali kaybeder.
_LEFT_EXACT = ["l", "lt", "lft", "sol"]
_LEFT_PREFIX = ["left", "link", "gauche", "izq", "sinistr", "esquerd", "venstre"]
_RIGHT_EXACT = ["r", "rt", "rght", "dx"]
_RIGHT_PREFIX = ["right", "recht", "droit", "derech", "destr", "direit", "hoyre"]


def _side_regex(exact, prefix):
    """Tam-kelime ve onek eslesmelerini tek bir desende birlestir."""
    return re.compile(
        r"(?:^|[^a-z])(?:"
        + r"(?:" + "|".join(exact) + r")(?=[^a-z]|$)"
        + r"|(?:" + "|".join(prefix) + r")"
        + r")"
    )


_LEFT_RE = _side_regex(_LEFT_EXACT, _LEFT_PREFIX)
_RIGHT_RE = _side_regex(_RIGHT_EXACT, _RIGHT_PREFIX)


def _text_laterality(text: str):
    if not text:
        return None
    t = text.lower()
    has_l = _LEFT_RE.search(t) is not None
    has_r = _RIGHT_RE.search(t) is not None
    if has_l and not has_r:
        return "L"
    if has_r and not has_l:
        return "R"
    return None                       # ikisi birden veya hicbiri -> karar verme


def image_center(ds):
    """Bir slice'in GORUNTU MERKEZININ hasta koordinatlarindaki konumu (3,).

    Neden koseyi degil merkezi kullaniyoruz (Faz 0.7 bulgusu):
    `ImagePositionPatient` goruntunun (0,0) pikselinin, yani KOSESININ konumudur.
    Hastanin x eksenine gore taraf cikarimi yapmak icin koseye bakmak, bir sehrin
    yerini sormak icin haritanin sol ust kosesine bakmak gibidir.

    Bu, duzlemler arasinda SISTEMATIK bir sapma uretir:
      - Sagittal : hastanin x ekseni slice eksenidir, goruntu icinde degil
                   -> slice'lar uzerinde ortalama alindiginda kose ~ merkez. Sapma yok.
      - Coronal/Axial : x ekseni goruntu ICINDEDIR
                   -> kose, merkezden ~yarim FOV (50-80 mm) uzakta. Sapma var.

    Iki konvansiyonu tek esikle karsilastirmak Faz 0.7'de %86.1 isabete yol acti
    (L ve R dagilimlarinin araliklari ortusuyordu). Merkez kullanmak bu sapmayi kaldirir.

    DICOM geometrisi: piksel (r, c)'nin konumu
        P = IPP + c * PixelSpacing[1] * satir_yonu + r * PixelSpacing[0] * sutun_yonu
    burada satir_yonu = IOP[0:3] (sutun indisi artarken), sutun_yonu = IOP[3:6].
    Merkez icin r = (Rows-1)/2, c = (Columns-1)/2.

    Geometri eksikse IPP'yi oldugu gibi dondurur (kose) — bu durumda cikarim
    eskisi kadar zayif olur ama cokmez.
    """
    ipp = ds.get("ImagePositionPatient", None)
    if ipp is None:
        return None
    p = np.asarray([float(v) for v in ipp], dtype=np.float64)

    iop = ds.get("ImageOrientationPatient", None)
    ps = ds.get("PixelSpacing", None)
    rows = ds.get("Rows", None)
    cols = ds.get("Columns", None)
    if iop is None or ps is None or rows is None or cols is None:
        return p

    o = np.asarray([float(v) for v in iop], dtype=np.float64)
    row_dir, col_dir = o[:3], o[3:6]
    row_sp, col_sp = float(ps[0]), float(ps[1])
    return (p
            + ((float(cols) - 1.0) / 2.0) * col_sp * row_dir
            + ((float(rows) - 1.0) / 2.0) * row_sp * col_dir)


def series_center_x(datasets: Sequence):
    """Serinin slice'lari uzerinde ortalama goruntu-merkezi x'i (mm) veya None.

    Notebook bu degeri ham olarak alip esik taramasi yapabilsin diye ayri fonksiyon.
    """
    xs = [c[0] for c in (image_center(ds) for ds in datasets) if c is not None]
    return float(np.mean(xs)) if xs else None


def detect_laterality(datasets: Sequence, use_ipp_layer: bool = True,
                      ipp_deadzone: float = 30.0) -> tuple:
    """Serinin hangi diz oldugunu bul. (taraf, kaynak) dondurur.

    Katmanli, en guvenilirden en zayifa:
      1. Laterality tag'i                  (Faz 0.4: %48.3 dolu)
      2. SeriesDescription/ProtocolName    (Faz 0.7: %97.9 isabet, kapsam %33.1)
      3. Goruntu merkezinin x-isareti      (VARSAYILAN KAPALI, bkz. asagi)
      4. None -> aynalama yapma

    3. katman ARTIK VARSAYILAN OLARAK ACIK (Faz 1A olcumu, 12.000 tag'li seri):
    `image_center` duzeltmesinden sonra isabet %86.1 -> **%98.7**'ye cikti.

    `ipp_deadzone=30` neden 30 mm: isabet, dizin izomerkeze gore konumuna
    (|merkez_x|) MONOTON bagli. Olculen degerler:

        |merkez_x|   isabet   seri payi
          0-10 mm    0.507      1.7%     <- yazi-tura, sinyal YOK
         10-20 mm    0.647      0.8%
         20-30 mm    0.830      0.7%
         30-50 mm    0.966      4.1%
         50-75 mm    0.993     26.2%
         75-100 mm   0.988     41.1%
          100+ mm    0.983     25.3%

    Sebep fiziksel: bazi merkezlerde diz taranirken izomerkeze ortalaniyor, o
    zaman x isareti dizin tarafini degil gurultuyu olcer. 30 mm olu bolge tam
    bu serileri ayiklar ve MARKADAN BAGIMSIZ calisir — scanner bazli kural
    yazmaya gerek yok (bkz. PHASE1A_FINDINGS.md Bolum 2).

    UYARI — bu fonksiyon SERI duzeyinde calisir. Bir study'de tek diz vardir;
    study tarafi icin `resolve_study_laterality()` kullanin. 25 study'de
    `Laterality` tag'i seriler arasinda CELISKILI ve bu fonksiyon ilk buldugu
    tag'i dondurur — yanlis olabilir. Cogunluk oyu bunu duzeltir.
    """
    for ds in datasets:
        v = ds.get("Laterality", None)
        if v:
            s = str(v).strip().upper()[:1]
            if s in ("L", "R"):
                return s, "Laterality"

    for field_name in ("SeriesDescription", "ProtocolName", "BodyPartExamined"):
        for ds in datasets[:3]:
            side = _text_laterality(str(ds.get(field_name, "") or ""))
            if side:
                return side, field_name

    if use_ipp_layer:
        mx = series_center_x(datasets)
        if mx is not None and abs(mx) > ipp_deadzone:
            # +x hastanin SOLUNA bakar (DICOM hasta koordinat sistemi)
            return ("L" if mx > 0 else "R"), "image_center_x"

    return None, "bulunamadi"


def resolve_study_laterality(series_datasets, use_ipp_layer: bool = True,
                             ipp_deadzone: float = 30.0) -> tuple:
    """Bir study'nin tarafini serilerinin COGUNLUK OYU ile belirle.

    `series_datasets`: her elemani bir serinin dataset listesi olan dizi.
    Dondurur: (taraf, bilgi_sozlugu)

    Neden study duzeyi zorunlu:
    1. Bir study'de tek diz var — seri bazinda farkli cevap vermek anlamsiz.
    2. 25 study'de `Laterality` tag'i seriler arasinda CELISKILI olcusuldu
       (Faz 1A). 25'inin 25'inde merkez-x TEK taraf gosterdi, yani iki diz
       yok, sadece tag bazi serilerde yanlis. Cogunluk oyu bunu duzeltir;
       `detect_laterality`nin tek basina yaptigi "ilk tag'i al" duzeltmez.

    Cogunluk oyunun isabeti ARTIRMADIGINI da bilerek not edelim (seri %98.7 ->
    study %98.2): hatalar study icinde KORELE, cunku kaynak ayni hatali
    konumlandirma. Yani oy vermek gurultuyu ortalamiyor; sadece celiskili
    tag'leri ve seri basina tutarsizligi cozuyor.
    """
    votes, sources = [], []
    for dss in series_datasets:
        if not dss:
            continue
        side, src = detect_laterality(dss, use_ipp_layer=use_ipp_layer,
                                      ipp_deadzone=ipp_deadzone)
        if side in ("L", "R"):
            votes.append(side)
            sources.append(src)

    info = {"n_series_voted": len(votes), "n_L": votes.count("L"),
            "n_R": votes.count("R"), "sources": sorted(set(sources)),
            "conflict": len(set(votes)) > 1}
    if not votes:
        return None, info
    # Esitlikte tag/metin kaynakli oylar merkez-x'e gore ustun tutulur.
    strong = [v for v, s in zip(votes, sources) if s != "image_center_x"]
    pool = strong if (info["n_L"] == info["n_R"] and strong) else votes
    side = "L" if pool.count("L") >= pool.count("R") else "R"
    info["margin"] = abs(pool.count("L") - pool.count("R"))
    return side, info


def lr_flip_axis(iop):
    """Sol-sag aynalamasinin HANGI eksende yapilacagini bul.

    BU, ARCHITECTURE.md'DEKI "yatay flip" TARIFININ DUZELTMESIDIR.

    Sol dizi saga cevirmek, hastanin x ekseni (sol-sag) boyunca bir yansimadir.
    Ama bu yansimanin goruntudeki karsiligi DUZLEME GORE DEGISIR:

      - Coronal / Axial: goruntunun genislik ekseni zaten hastanin x ekseni
        -> yatay cevirme (axis 2). DOGRU.
      - Sagittal: goruntunun eksenleri on-arka ve ust-alt. Hastanin x ekseni
        goruntu icinde DEGIL, SLICE EKSENIDIR.
        -> aynalama = slice sirasini ters cevir (medial<->lateral, axis 0).
        Burada yatay flip yapmak on-arkayi ters cevirir; yani patellayi arkaya
        tasir. Sessizce yanlis veri uretir.

    Genel cozum: hacmin uc ekseninden (slice, yukseklik, genislik) hangisinin
    hasta-x bileseni en buyukse o eksende cevir.
      axis 0 = slice      -> normal vektor
      axis 1 = yukseklik  -> IOP'nin sutun yonu (artan satir indisi)
      axis 2 = genislik   -> IOP'nin satir yonu (artan sutun indisi)
    """
    n = slice_normal(iop)
    if n is None:
        return None
    row = np.asarray(iop[:3], dtype=np.float64)   # -> genislik ekseni (axis 2)
    col = np.asarray(iop[3:], dtype=np.float64)   # -> yukseklik ekseni (axis 1)
    x_components = [abs(n[0]), abs(col[0]), abs(row[0])]
    return int(np.argmax(x_components))


def apply_lr_mirror(vol: np.ndarray, iop) -> tuple:
    """Hacmi sol-sag aynala. (aynalanmis_hacim, eksen) dondurur."""
    ax = lr_flip_axis(iop)
    if ax is None:
        return vol, None
    return np.flip(vol, axis=ax).copy(), ax


# =====================================================================
# 5. Boyutlandirma
# =====================================================================

def resize_volume(vol: np.ndarray, size: int) -> np.ndarray:
    """Her slice'i (size, size) yap.

    Kucultmede INTER_AREA kullaniliyor: hedef pikselin kapsadigi alanin
    ortalamasini alir. INTER_LINEAR kucultmede orneklem atlar ve ince yapilari
    (menisküs yirtigi gibi) tamamen kaybedebilir — asil kaygimiz tam da bu.
    """
    if vol.size == 0:
        return vol
    n, h, w = vol.shape
    if (h, w) == (size, size):
        return vol
    if _HAS_CV2:
        interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
        return np.stack([cv2.resize(s, (size, size), interpolation=interp)
                         for s in vol]).astype(np.float32)
    from PIL import Image
    resample = Image.BILINEAR if (h < size and w < size) else Image.BOX
    return np.stack([
        np.asarray(Image.fromarray(s).resize((size, size), resample), dtype=np.float32)
        for s in vol
    ])


# =====================================================================
# 6. FAZ 0.9 — 2.5D temsil
# =====================================================================

def sample_window_centers(n_slices: int, n_windows: int, k: int = 2,
                          stride: int = 1) -> list:
    """2.5D pencerelerin merkez indisleri — kenarlardan kacinarak esit aralikli.

    Kenarlardan kacinma sebebi: ilk ve son slice'larin komsulari yok, tekrarla
    doldurmak zorunda kaliriz; ustelik seri kenarlarinda genelde anatomi degil
    hava/gurultu olur.
    """
    if n_slices <= 0:
        return []
    half = k * stride
    lo, hi = half, n_slices - 1 - half
    if hi < lo:                              # seri pencereden kisa
        return [n_slices // 2] * n_windows
    if n_windows == 1:
        return [(lo + hi) // 2]
    return [int(round(lo + i * (hi - lo) / (n_windows - 1))) for i in range(n_windows)]


def make_25d(vol: np.ndarray, center: int, k: int = 2, stride: int = 1) -> np.ndarray:
    """Merkez slice +/- k komsuyu KANAL boyutunda stack'le -> (2k+1, H, W).

    Neden 2.5D: tek slice baglamsizdir (bir menisküs yirtigi komsu kesitlerle
    birlikte anlam kazanir), tam 3D CNN ise pahali ve pretrained agirligi yok.
    2.5D, ImageNet pretrained 2D backbone'u koruyup derinlik baglamini kanal
    olarak verir.

    Kenarlarda kirpma (clip) ile tekrar ediyoruz: ilk slice'in "onceki"si yine
    ilk slice olur. Sifirla doldurmak yapay bir kenar olusturur, model bunu
    ogrenebilir.
    """
    if vol.size == 0:
        return vol
    idx = np.clip(np.arange(center - k * stride, center + k * stride + 1, stride),
                  0, len(vol) - 1)
    return vol[idx]


# =====================================================================
# 7. Hepsini birlestiren giris
# =====================================================================

@dataclass
class SeriesVolume:
    vol: np.ndarray                       # (N, H, W) float32, [0, 1]
    plane: str = "Unknown"
    laterality: object = None
    laterality_source: str = ""
    mirrored: bool = False
    mirror_axis: object = None
    sort_method: str = ""
    pixel_spacing: object = None
    n_files: int = 0
    n_slices: int = 0
    info: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


def load_series(sdir: str, size: int = 256, target_side: str = "R",
                p_low: float = 1.0, p_high: float = 99.0,
                max_slices: int = None, workers: int = 8,
                use_ipp_layer: bool = False) -> SeriesVolume:
    """Bir seri klasorunu uctan uca isle.

    target_side=None verilirse lateralite normalizasyonu yapilmaz (ablation icin).
    use_ipp_layer=True, goruntu-merkezi x heuristic'ini (3. katman) devreye alir;
    varsayilan KAPALI, gerekcesi `detect_laterality` docstring'inde.
    """
    paths = list_dicom_files(sdir)
    datasets, errors = read_datasets(paths, workers=workers)
    if not datasets:
        return SeriesVolume(vol=np.zeros((0, size, size), np.float32),
                            n_files=len(paths), errors=errors)

    datasets, sort_info = sort_datasets(datasets)

    if max_slices is not None and len(datasets) > max_slices:
        # Esit aralikli alt orneklem: bastan kesmek anatominin yarisini atar.
        sel = np.linspace(0, len(datasets) - 1, max_slices).round().astype(int)
        datasets = [datasets[i] for i in sel]

    iop = datasets[0].get("ImageOrientationPatient", None)
    vol, flags = extract_pixels(datasets)
    vol = normalize_volume(vol, p_low, p_high, invert=flags["monochrome1"])

    side, side_src = detect_laterality(datasets, use_ipp_layer=use_ipp_layer)
    mirrored, axis = False, None
    if target_side is not None and side is not None and side != target_side:
        vol, axis = apply_lr_mirror(vol, iop)
        mirrored = axis is not None

    vol = resize_volume(vol, size)

    ps = datasets[0].get("PixelSpacing", None)
    return SeriesVolume(
        vol=vol,
        plane=plane_from_iop(iop),
        laterality=side,
        laterality_source=side_src,
        mirrored=mirrored,
        mirror_axis=axis,
        sort_method=sort_info["method"],
        pixel_spacing=(float(ps[0]), float(ps[1])) if ps is not None else None,
        n_files=len(paths),
        n_slices=len(vol),
        info=dict(list(sort_info.items()) +
                  [(k, v) for k, v in flags.items() if k != "kept_idx"]),
        errors=errors,
    )


# ===========================================================================
# Faz 2.1 / 2.2 — seri secimi ve study duzeyinde on isleme
# ===========================================================================
# Bir study'de ortalama 5.5 seri var (Faz 1A). Hepsini islemek pahali, rastgele
# secmek bilgi kaybi. Metadata gudumlu secim: her duzlemden BIR seri, fluid
# sensitive olani onceliklendirerek.
#
# Neden bu isliyor: Faz 1A'da her study'nin ucunde de (Sagittal/Coronal/Axial)
# en az bir seri oldugu olculdu — 4.407/4.407, yani %100. Maskeleme gerekmiyor.
#
# Oncelik gerekcesi (ARCHITECTURE Bolum 1.3): Effusion / Synovitis / Contusion /
# Baker's sivi-duyarli sekanslarda gorunur; menisküs ve ACL sagittal'de;
# MCL coronal'de; PF OA axial'de. Duzlem basina bir seri hepsini kapsar.
PLANES = ("Sagittal", "Coronal", "Axial")


def select_series(series_rows, planes=PLANES):
    """Study'nin serilerinden duzlem basina bir tane sec.

    `series_rows`: her elemani en az su anahtarlari olan sozluk/Series:
        SeriesInstanceUID, Anatomical_Plane, Fluid_Sensitive, Fat_Suppression,
        (opsiyonel) n_files
    Dondurur: {duzlem: secilen_satir}  — bulunamayan duzlem anahtarda olmaz.

    Siralama olcutu, en onemliden en az onemliye:
      1. Fluid_Sensitive  (sivi-duyarli sekanslar 4 bulguyu dogrudan gosterir)
      2. Fat_Suppression  (yag baskilama odemi belirginlestirir)
      3. slice sayisi     (daha fazla kesit = daha iyi 2.5D baglami)
    """
    out = {}
    for pl in planes:
        cand = [r for r in series_rows
                if str(_get(r, "Anatomical_Plane", "")).strip() == pl]
        if not cand:
            continue
        cand.sort(key=lambda r: (int(_get(r, "Fluid_Sensitive", 0) or 0),
                                 int(_get(r, "Fat_Suppression", 0) or 0),
                                 int(_get(r, "n_files", 0) or 0)), reverse=True)
        out[pl] = cand[0]
    return out


def _get(row, key, default=None):
    """dict ve pandas Series'i ayni sekilde oku."""
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if v is None else v


def subsample_slices(vol: np.ndarray, n: int) -> np.ndarray:
    """Slice eksenini n'e esit arali indir (ya da n'den azsa oldugu gibi birak).

    Neden bastan kesmek DEGIL esit aralikli: bastan/sondan kesmek anatominin
    bir ucunu tamamen atar. Esit aralikli ornekleme tum hacmi temsil eder.
    n'den az slice varsa tekrar ederek doldurmuyoruz — gercek slice sayisi
    metadata'da saklaniyor, egitim tarafi buna gore pencere secer.
    """
    if vol.shape[0] <= n:
        return vol
    idx = np.linspace(0, vol.shape[0] - 1, n).round().astype(int)
    return vol[idx]


def to_uint8(vol: np.ndarray) -> np.ndarray:
    """[0,1] float hacmi uint8'e cevir — DISKTE IKI KAT YER KAZANCI.

    normalize_volume zaten percentile clip + min-max yapip [0,1]'e getiriyor.
    256 seviyeye yuvarlamak MR icin pratikte kayipsiz: orijinal 12-bit ama
    normalizasyon sonrasi ayirt edici bilgi bu araliga zaten sikismis durumda.
    float16 saklamak iki kat yer yer, karsiligi gorunmuyor.
    """
    return np.clip(vol * 255.0 + 0.5, 0, 255).astype(np.uint8)


def load_study(data_root, study_uid, series_rows, size=256, n_slices=16,
               target_side="R", workers=8, use_ipp_layer=True,
               ipp_deadzone=30.0, split="train"):
    """Bir study'yi uctan uca isle: seri sec -> oku -> normalize -> aynala -> resize.

    Dondurur: (veri_sozlugu, bilgi_sozlugu)
      veri: {duzlem: uint8 array (n_slices, size, size)}
      bilgi: taraf, aynalama, secilen seri UID'leri, hatalar, sureler

    LATERALITE STUDY DUZEYINDE cozuluyor, seri duzeyinde degil. Sebep Faz 1A
    bulgusu: 25 study'de `Laterality` tag'i seriler arasinda CELISKILI ve
    25'inin 25'inde merkez-x TEK taraf gosterdi (yani tag yanlis, iki diz yok).
    Cogunluk oyu bunu duzeltir; seri basina karar vermek study'yi kendi icinde
    tutarsiz hale getirir — aynalanmis ve aynalanmamis seriler ayni study'de.
    """
    sel = select_series(series_rows)
    dss_per_plane, raw = {}, {}
    errors = []
    for pl, row in sel.items():
        sdir = series_dir(data_root, study_uid,
                          str(_get(row, "SeriesInstanceUID")), split=split)
        paths = list_dicom_files(sdir)
        dss, errs = read_datasets(paths, workers=workers)
        errors.extend(errs)
        if not dss:
            continue
        dss, sort_info = sort_datasets(dss)
        dss_per_plane[pl] = dss
        raw[pl] = (dss, sort_info)

    if not raw:
        return {}, {"laterality": None, "error": "hic seri okunamadi",
                    "errors": errors}

    # 1) taraf: TUM secilen serilerin oyu
    side, lat_info = resolve_study_laterality(
        list(dss_per_plane.values()), use_ipp_layer=use_ipp_layer,
        ipp_deadzone=ipp_deadzone)

    data, per_plane = {}, {}
    for pl, (dss, sort_info) in raw.items():
        iop = dss[0].get("ImageOrientationPatient", None)
        vol, flags = extract_pixels(dss)
        vol = normalize_volume(vol, invert=flags["monochrome1"])
        mirrored, axis = False, None
        if target_side is not None and side is not None and side != target_side:
            vol, axis = apply_lr_mirror(vol, iop)
            mirrored = axis is not None
        vol = subsample_slices(vol, n_slices)
        vol = resize_volume(vol, size)
        data[pl] = to_uint8(vol)
        ps = dss[0].get("PixelSpacing", None)
        per_plane[pl] = {
            "series_uid": str(dss[0].get("SeriesInstanceUID", "")),
            "n_slices_native": len(dss), "n_slices_kept": int(vol.shape[0]),
            "mirrored": mirrored, "mirror_axis": axis,
            "sort_method": sort_info["method"],
            "pixel_spacing": float(ps[0]) if ps is not None else None,
        }

    return data, {"laterality": side, "laterality_info": lat_info,
                  "planes": per_plane, "n_errors": len(errors),
                  "errors": errors[:3]}
