# %% [markdown]
# # Faz 0.3 + 0.4 — Transfer Syntax Doğrulama & DICOM Tag Envanteri
#
# **Amaç:** Modele bir satır bile yazmadan önce iki soruyu kesin olarak yanıtlamak:
#
# 1. **0.3** — Veri setindeki *her* transfer syntax'ı gerçekten decode edebiliyor muyuz?
#    (Edemediğimizi Faz 2'de öğrenmek istemiyoruz.)
# 2. **0.4** — De-identification sonrası hangi DICOM tag'leri hayatta kaldı, ne kadar dolular,
#    ve hangileri **site proxy** olarak fold gruplaması için kullanılabilir?
#
# **Strateji notu:** Kaggle'ın `/kaggle/input` mount'u ağ üzerinden çalışır, dosya başına
# gecikme yüksektir. Bu yüzden dosyalara *bir kez* dokunup ihtiyacımız olan her şeyi
# (transfer syntax + tüm tag'ler + ilgilendiğimiz değerler) aynı okumada topluyoruz.
# Piksel verisini okumuyoruz (`stop_before_pixels=True`) — sadece başlık.
#
# **Çalışma süresi:** ~2-5 dakika. CPU notebook yeterli, GPU kotanızı harcamayın.

# %% [markdown]
# ## 0. Yapılandırma

# %%
SEED = 42

# Kaç study örnekleyelim? Site heterojenliğini yakalamak için study sayısı (dosya sayısı
# değil) önemli — 19+ merkez var, her merkezden birkaç study görmek istiyoruz.
N_STUDIES = 500

# Study başına kaç seri? >=2 olmalı: "Manufacturer bir study içinde sabit mi?" sorusunu
# ancak aynı study'nin farklı serilerine bakarak yanıtlayabiliriz. Bu, grup anahtarının
# geçerli olup olmadığını belirler.
SERIES_PER_STUDY = 2

# Seri başına kaç dosya? 2 = ilk slice + orta slice. Slice'lar arası değişen tag'leri
# (ImagePositionPatient, InstanceNumber) tespit etmek için.
FILES_PER_SERIES = 2

# Her transfer syntax için kaç dosyayı gerçekten decode edip doğrulayalım?
N_DECODE_PER_SYNTAX = 5

# I/O-bound iş → thread'ler işe yarar.
MAX_WORKERS = 16

OUT_DIR = "/kaggle/working"

# %% [markdown]
# ## 1. Ortam kontrolü — decode edici kütüphaneler kurulu mu?
#
# `pydicom` tek başına **sıkıştırılmamış** DICOM'ları okur. JPEG Lossless ve JPEG 2000 için
# harici decoder gerekir. Analoji: pydicom bir medya oynatıcı, bu kütüphaneler codec'ler —
# oynatıcı dosyayı açar ama codec yoksa görüntüyü çözemez.
#
# | Kütüphane | Ne için gerekli |
# |---|---|
# | `pylibjpeg` + `pylibjpeg-libjpeg` | JPEG Lossless (1.2.840.10008.1.2.4.70 vb.) |
# | `pylibjpeg` + `pylibjpeg-openjpeg` | JPEG 2000 (1.2.840.10008.1.2.4.90/91) |
# | `gdcm` | Yukarıdakilerin alternatifi, çoğu formatı kapsar |
#
# Kaggle imajında bunlar genelde hazır gelir. **Eksikse şimdi (internet açıkken) kurun** —
# ama unutmayın: submission notebook'unda internet kapalı olacak, o zaman bu paketleri
# bir Kaggle Dataset'ten offline kurmanız gerekecek (Faz 2.13).

# %%
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 250)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)


def _probe(import_name, pretty=None):
    """Bir paketin kurulu olup olmadigini ve surumunu dondur."""
    pretty = pretty or import_name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        return f"  [OK]      {pretty:<28} {ver}"
    except Exception as e:
        return f"  [EKSIK]   {pretty:<28} ({type(e).__name__})"


print("Python :", sys.version.split()[0])
print("pydicom:", pydicom.__version__)
print("\nDecode edici kutuphaneler:")
for name, pretty in [
    ("pylibjpeg", "pylibjpeg"),
    ("libjpeg", "pylibjpeg-libjpeg"),
    ("openjpeg", "pylibjpeg-openjpeg"),
    ("rle", "pylibjpeg-rle"),
    ("gdcm", "python-gdcm"),
    ("PIL", "Pillow"),
]:
    print(_probe(name, pretty))

# pydicom 2.x ile 3.x arasinda pixel handler API'si degisti; ikisini de destekle.
print("\nAktif pixel data handler'lari:")
try:
    handlers = pydicom.config.pixel_data_handlers  # pydicom 2.x
    for h in handlers:
        avail = "kullanilabilir" if h.is_available() else "YOK"
        print(f"  {h.__name__.split('.')[-1]:<34} {avail}")
except AttributeError:
    print("  pydicom 3.x -> pydicom.pixels backend sistemi")
    try:
        from pydicom.pixels.decoders import JPEGLosslessSV1Decoder, JPEG2000Decoder

        for dec in (JPEGLosslessSV1Decoder, JPEG2000Decoder):
            print(f"  {dec.name:<34} backends={dec.available_backends}")
    except Exception as e:
        print(f"  (backend listesi alinamadi: {e})")

# %% [markdown]
# ## 2. Veri yolunu bul
#
# Yolu sabit kodlamak yerine otomatik buluyoruz — sabit yol, "neden çalışmıyor"un
# en sık sebebidir.
#
# **Dikkat:** `/kaggle/input` sıradan bir klasör değil, ağ üzerinden bağlanmış bir FUSE
# mount'u. `pathlib`'in normalde yutup `False` döndüğü bazı hatalar (`NotADirectoryError`
# gibi) burada istisna olarak fırlayabiliyor. O yüzden aşağıdaki tarama `pathlib` yerine
# her çağrısı `try/except` ile sarılmış düz `os` fonksiyonlarını kullanıyor.
#
# Ayrıca Kaggle'ın iki farklı yerleşimi var; ikisini de deniyoruz:
# `/kaggle/input/<yarisma>/` ve `/kaggle/input/competitions/<yarisma>/`

# %%
KAGGLE_INPUT = "/kaggle/input"

# Otomatik bulucu yanilirsa yolu buraya elle yazip hucreyi tekrar calistirin.
DATA_OVERRIDE = None

# Veri kokunu tanidigimiz imza dosyasi.
MARKER = "train_series.csv"


def safe_listdir(path):
    """os.listdir, ama FUSE mount'unun firlattigi her hatayi bos listeye cevirir."""
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def safe_isdir(path):
    try:
        return os.path.isdir(path)
    except OSError:
        return False


def find_data_root():
    """Veri kokunu bul. Once bilinen yerlesimler, sonra sinirli bir tarama."""
    if DATA_OVERRIDE:
        return DATA_OVERRIDE, "elle belirtildi (DATA_OVERRIDE)"

    slug = "rsna-knee-abnormality-detection"
    for cand in [f"{KAGGLE_INPUT}/{slug}", f"{KAGGLE_INPUT}/competitions/{slug}"]:
        if MARKER in safe_listdir(cand):
            return cand, "bilinen yerlesim"

    # Fallback: iki seviye tara. Agir klasorlere (train_series/ gibi) hic girmiyoruz —
    # her listdir bir ag turu, binlerce study klasorunu gezmek dakikalar surer.
    SKIP = {"train_series", "test_series"}
    level1 = [f"{KAGGLE_INPUT}/{n}" for n in safe_listdir(KAGGLE_INPUT)]
    for p in level1:
        if MARKER in safe_listdir(p):
            return p, "1. seviye tarama"
    for p in level1:
        if not safe_isdir(p):
            continue
        for n in safe_listdir(p):
            if n in SKIP:
                continue
            q = f"{p}/{n}"
            if MARKER in safe_listdir(q):
                return q, "2. seviye tarama"
    return None, None


# --- Once ne gordugumuzu dokelim: hata olsa bile bu cikti teshis icin degerli ---
print("=== /kaggle/input YAPISI ===")
top = safe_listdir(KAGGLE_INPUT)
if not top:
    print("  (BOS veya okunamiyor — dataset bagli mi?)")
for n in top:
    p = f"{KAGGLE_INPUT}/{n}"
    tag = "DIR " if safe_isdir(p) else "FILE"
    kids = safe_listdir(p) if safe_isdir(p) else []
    preview = ", ".join(kids[:6]) + (" ..." if len(kids) > 6 else "")
    print(f"  {tag} {n}")
    if kids:
        print(f"        -> {len(kids)} girdi: {preview}")

DATA_STR, how = find_data_root()
if DATA_STR is None:
    raise FileNotFoundError(
        "Yarisma verisi bulunamadi.\n"
        "  1) Yarisma kurallarini kabul ettiniz mi? (Rules -> I Understand and Accept)\n"
        "  2) Sag panel -> '+ Add Input' -> Competitions -> rsna-knee-abnormality-detection\n"
        "  3) Yukaridaki yapi dokumunde veriyi goruyorsaniz yolu DATA_OVERRIDE'a yazip\n"
        "     bu hucreyi tekrar calistirin."
    )

DATA = Path(DATA_STR)
print(f"\nVERI YOLU: {DATA}   ({how})")

print("\nUst seviye icerik:")
for n in safe_listdir(DATA):
    p = f"{DATA}/{n}"
    if safe_isdir(p):
        print(f"  DIR  {n}")
    else:
        try:
            size = f"{os.path.getsize(p) / 1e6:8.2f} MB"
        except OSError:
            size = "       ? MB"
        print(f"  FILE {n:<28} {size}")


# %% [markdown]
# ## 3. CSV'leri yükle
#
# Asıl EDA Faz 1'de. Burada sadece yapıyı doğruluyor ve iki şeyi not alıyoruz:
# kaç study gold-labeled, ve study başına kaç seri var.

# %%
train = pd.read_csv(DATA / "train.csv")
train_series = pd.read_csv(DATA / "train_series.csv")

print(f"train.csv        : {train.shape[0]:>7,} satir x {train.shape[1]} sutun")
print(f"train_series.csv : {train_series.shape[0]:>7,} satir x {train_series.shape[1]} sutun")
print("\ntrain.csv sutunlari      :", list(train.columns))
print("train_series.csv sutunlari:", list(train_series.columns))

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
present_labels = [c for c in LABELS if c in train.columns]
missing_labels = [c for c in LABELS if c not in train.columns]
if missing_labels:
    print("\n!! UYARI: beklenen etiket sutunlari yok:", missing_labels)
    print("   Gercek sutun adlarini yukaridaki listeden dogrulayin.")

n_gold = n_any = 0
if present_labels:
    lab = train[present_labels]
    n_gold = int(lab.notna().all(axis=1).sum())
    n_any = int(lab.notna().any(axis=1).sum())
    print(f"\nEtiket durumu ({len(present_labels)} sutun uzerinden):")
    print(f"  Tum etiketleri dolu (gold)          : {n_gold:>7,}  ({n_gold/len(train):6.2%})")
    print(f"  En az bir etiketi dolu              : {n_any:>7,}  ({n_any/len(train):6.2%})")
    print(f"  Hicbir etiketi yok (weak-label adayi): {len(train)-n_any:>7,}")

spc = train_series.groupby("StudyInstanceUID").size()
print(f"\nStudy basina seri sayisi: medyan={spc.median():.0f}  "
      f"ort={spc.mean():.2f}  min={spc.min()}  max={spc.max()}")

# %% [markdown]
# ## 4. Örneklemi kur
#
# Örneklem rastgele *dosya* değil, rastgele *study* üzerinden seçiliyor. Sebep: transfer
# syntax ve scanner metadata'sı study/merkez düzeyinde değişir. 5000 dosyayı 50 study'den
# çekersen yalnızca 50 scanner görürsün ve envanter yanıltıcı olur.

# %%
study_ids = train_series["StudyInstanceUID"].drop_duplicates()
n_pick = min(N_STUDIES, len(study_ids))
picked_studies = set(study_ids.sample(n=n_pick, random_state=SEED).tolist())

# Once satirlari karistirip sonra grup basina ilk N'i almak, `groupby.apply(sample)`
# yerine bilincli bir tercih: (a) pandas 2.2+ apply icinde gruplama sutununu dusuruyor,
# (b) `groupby.sample(n=2)` bir grupta 2'den az seri varsa hata veriyor.
# head() ikisinden de etkilenmez.
sub = (
    train_series[train_series["StudyInstanceUID"].isin(picked_studies)]
    .sample(frac=1.0, random_state=SEED)
    .groupby("StudyInstanceUID", sort=False)
    .head(SERIES_PER_STUDY)
    .reset_index(drop=True)
)
print(f"Orneklenen study : {sub['StudyInstanceUID'].nunique():,}")
print(f"Orneklenen seri  : {len(sub):,}")


def pick_files(study_uid, series_uid, k=FILES_PER_SERIES):
    """Bir seri klasorunden k dosya sec (ilk ve orta slice)."""
    d = DATA / "train_series" / study_uid / series_uid
    try:
        names = sorted(os.listdir(d))
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []
    dcm = [n for n in names if n.lower().endswith(".dcm")]
    names = dcm if dcm else names
    if not names:
        return []
    idxs = [0] if (k == 1 or len(names) == 1) else sorted({0, len(names) // 2})[:k]
    return [(study_uid, series_uid, str(d / names[i]), len(names)) for i in idxs]


t0 = time.time()
pairs = list(sub[["StudyInstanceUID", "SeriesInstanceUID"]].itertuples(index=False, name=None))
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    nested = list(ex.map(lambda r: pick_files(r[0], r[1]), pairs))
TASKS = [t for group in nested for t in group]
print(f"Taranacak dosya  : {len(TASKS):,}   (klasor listeleme {time.time()-t0:.1f}s)")

slice_counts = [g[0][3] for g in nested if g]
if slice_counts:
    s = pd.Series(slice_counts)
    print(f"Seri basina slice: medyan={s.median():.0f} ort={s.mean():.1f} "
          f"min={s.min()} p95={s.quantile(0.95):.0f} max={s.max()}")

# %% [markdown]
# ## 5. Tek geçişte başlık taraması
#
# Her dosya için tek okumada üç şeyi birden topluyoruz:
# 1. **Transfer syntax** (0.3 için)
# 2. **Mevcut tüm tag'ler** — iç içe sequence'lar dahil (0.4 için)
# 3. **İlgilendiğimiz tag'lerin değerleri** — site proxy analizi için
#
# Kritik ayrım: *"tag mevcut"* ile *"tag dolu"* aynı şey değil. De-identification bazı
# tag'leri silmek yerine **boşaltır**. Boş bir `InstitutionName` bize hiçbir işe yaramaz,
# o yüzden ikisini ayrı sayıyoruz.

# %%
# Degerini ozellikle merak ettigimiz tag'ler.
INTERESTING = [
    # --- site / scanner proxy adaylari (Faz 1C fold stratejisi) ---
    "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
    "InstitutionName", "StationName", "DeviceSerialNumber", "SoftwareVersions",
    "InstitutionalDepartmentName",
    # --- sekans / protokol ---
    "SeriesDescription", "ProtocolName", "SequenceName", "ScanningSequence",
    "SequenceVariant", "ScanOptions", "MRAcquisitionType", "Modality",
    "EchoTime", "RepetitionTime", "InversionTime", "FlipAngle", "EchoTrainLength",
    # --- geometri (slice siralama, Faz 0.5) ---
    "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
    "SliceThickness", "SpacingBetweenSlices", "PatientPosition",
    "InstanceNumber", "SeriesNumber", "AcquisitionNumber",
    # --- lateralite (mirror, Faz 0.7) ---
    "Laterality", "ImageLaterality", "BodyPartExamined",
    # --- intensite (normalizasyon, Faz 0.6) ---
    "RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth",
    "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation",
    "PhotometricInterpretation", "SamplesPerPixel", "Rows", "Columns",
    "SmallestImagePixelValue", "LargestImagePixelValue",
    # --- hasta / tarih ---
    "PatientSex", "PatientAge", "StudyDate", "SeriesDate", "ContentDate",
]

# Degerini okumaya calismayacagimiz VR'lar (binary veya devasa olabilir).
OPAQUE_VR = {"OB", "OW", "OF", "OD", "OL", "OV", "UN", "SQ", "UT"}


def _stringify(value, limit=90):
    """DICOM degerini kisa, karsilastirilabilir bir string'e cevir."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        s = "\\".join(str(x) for x in value)
    else:
        s = str(value)
    s = s.strip()
    if s == "":
        return None
    return (s[:limit] + "...") if len(s) > limit else s


def scan_header(task):
    study_uid, series_uid, path, n_slices = task
    rec = {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "path": path,
        "n_slices_in_series": n_slices,
        "read_ok": False,
        "error": None,
        "ts_uid": None,
        "ts_name": None,
        "_tags": {},
    }
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    rec["read_ok"] = True

    # --- 1) transfer syntax ---
    try:
        ts = ds.file_meta.TransferSyntaxUID
        rec["ts_uid"] = str(ts)
        rec["ts_name"] = ts.name
    except Exception:
        # file_meta yoksa standart geregi Implicit VR Little Endian varsayilir
        rec["ts_uid"] = "MISSING_FILE_META"
        rec["ts_name"] = "MISSING_FILE_META"

    # --- 2) mevcut tum tag'ler (sequence'lar dahil) ---
    tags = {}
    try:
        for elem in ds.iterall():
            kw = elem.keyword or elem.name or "Unknown"
            key = (str(elem.tag), kw, str(elem.VR))
            if elem.VR in OPAQUE_VR:
                val = None
            else:
                try:
                    val = _stringify(elem.value)
                except Exception:
                    val = None
            if key not in tags:  # ayni tag birden fazla sequence item'inda olabilir
                tags[key] = val
    except Exception as e:
        rec["error"] = f"iterall: {type(e).__name__}: {e}"
    rec["_tags"] = tags

    # --- 3) ilgilendigimiz tag'lerin degerleri ---
    for kw in INTERESTING:
        try:
            rec[kw] = _stringify(ds.get(kw, None))
        except Exception:
            rec[kw] = None
    return rec


t0 = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    RECORDS = list(ex.map(scan_header, TASKS))
elapsed = time.time() - t0

n_files_ok = sum(r["read_ok"] for r in RECORDS)
print(f"Taranan dosya: {len(RECORDS):,}   basarili: {n_files_ok:,}   "
      f"hatali: {len(RECORDS)-n_files_ok:,}   sure: {elapsed:.1f}s "
      f"({1000*elapsed/max(len(RECORDS),1):.1f} ms/dosya)")

errs = [r for r in RECORDS if not r["read_ok"]]
if errs:
    print("\n!! Okunamayan dosya ornekleri:")
    for r in errs[:5]:
        print("  ", r["error"], "->", r["path"])

# %% [markdown]
# ## 6. FAZ 0.3a — Transfer syntax sayımı
#
# Hangi formatlar var ve ne oranda? Bu tablo bir sonraki adımda hangi decoder'ları
# test etmemiz gerektiğini söyler.

# %%
ts_df = (
    pd.DataFrame([(r["ts_uid"], r["ts_name"]) for r in RECORDS if r["read_ok"]],
                 columns=["ts_uid", "ts_name"])
      .value_counts()
      .rename("n_files")
      .reset_index()
)
ts_df["pct"] = (ts_df["n_files"] / ts_df["n_files"].sum() * 100).round(2)

# Study duzeyinde de bakalim: transfer syntax merkeze mi bagli?
ts_study = pd.DataFrame(
    [(r["StudyInstanceUID"], r["ts_name"]) for r in RECORDS if r["read_ok"]],
    columns=["StudyInstanceUID", "ts_name"],
)
ts_df["n_studies"] = ts_df["ts_name"].map(
    ts_study.groupby("ts_name")["StudyInstanceUID"].nunique()
)

print("=== FAZ 0.3a — TRANSFER SYNTAX DAGILIMI ===\n")
print(ts_df.to_string(index=False))

mixed = ts_study.groupby("StudyInstanceUID")["ts_name"].nunique()
print(f"\nIcinde birden fazla transfer syntax gecen study: {(mixed > 1).sum()} / {len(mixed)}")

# %% [markdown]
# ## 7. FAZ 0.3b — Gerçek decode testi
#
# Sayım yeterli değil: "JPEG 2000 var" bilgisi onu *açabildiğimiz* anlamına gelmez.
# Bulunan her transfer syntax için birkaç dosyayı gerçekten `pixel_array`'e çeviriyoruz.
#
# Kaydettiklerimiz doğrudan ön işleme kararlarını etkiler:
# - **shape / dtype / BitsStored** → 16-bit container'da kaç bit gerçekten dolu?
# - **min / max / p1 / p99** → MR'da sabit bir ölçek yok; Faz 0.6 normalizasyonu buna dayanacak
# - **süre** → 570 GB'ı ön işlemenin maliyeti (Faz 2.2/2.3 planlaması)

# %%
by_ts = defaultdict(list)
for r in RECORDS:
    if r["read_ok"]:
        by_ts[r["ts_name"]].append(r["path"])

decode_rows = []
print("=== FAZ 0.3b — DECODE TESTI ===\n")
for ts_name, paths in sorted(by_ts.items()):
    test_paths = paths[:N_DECODE_PER_SYNTAX]
    print(f"[{ts_name}]  ({len(paths)} dosya bulundu, {len(test_paths)} test ediliyor)")
    for p in test_paths:
        row = {"ts_name": ts_name, "path": p, "ok": False, "error": None}
        try:
            t0 = time.time()
            ds = pydicom.dcmread(p)
            arr = ds.pixel_array
            row.update(
                ok=True,
                shape=str(arr.shape),
                dtype=str(arr.dtype),
                vmin=float(np.min(arr)),
                vmax=float(np.max(arr)),
                p01=float(np.percentile(arr, 1)),
                p99=float(np.percentile(arr, 99)),
                bits_stored=int(getattr(ds, "BitsStored", -1)),
                bits_alloc=int(getattr(ds, "BitsAllocated", -1)),
                photometric=str(getattr(ds, "PhotometricInterpretation", "?")),
                sec=round(time.time() - t0, 4),
            )
            print(f"    OK   {arr.shape} {arr.dtype}  range=[{row['vmin']:.0f}, {row['vmax']:.0f}]"
                  f"  stored={row['bits_stored']}bit  {row['sec']*1000:.0f}ms")
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"    FAIL {row['error']}")
        decode_rows.append(row)
    print()

decode_df = pd.DataFrame(decode_rows)

print("--- OZET ---")
summary = decode_df.groupby("ts_name")["ok"].agg(["sum", "count"])
summary.columns = ["basarili", "denenen"]
print(summary.to_string())

failed_ts = summary[summary["basarili"] == 0].index.tolist()
if failed_ts:
    print("\n" + "!" * 72)
    print("!! HIC DECODE EDILEMEYEN TRANSFER SYNTAX VAR:")
    for t in failed_ts:
        print("   -", t)
    print("!! Cozum: pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm kurun.")
    print("!! Submission notebook'unda internet KAPALI olacagi icin bu paketler")
    print("!! bir Kaggle Dataset'e wheel olarak yuklenmeli (Faz 2.13).")
    print("!" * 72)
else:
    print("\n>> Tum transfer syntax'lar basariyla decode edildi.")

if decode_df["ok"].any():
    ok = decode_df[decode_df["ok"]]
    print(f"\nOrtalama decode suresi: {ok['sec'].mean()*1000:.0f} ms/slice")
    est = ok["sec"].mean() * 819_640 / 3600
    print(f"Kaba tahmin: 819,640 dosyanin tamamini tek thread ile decode ~{est:.1f} saat.")
    print("(Seri secimi + paralellestirme ile bu ciddi olcude duser — Faz 2.1/2.2)")

# %% [markdown]
# ## 8. FAZ 0.4a — Tag envanteri
#
# Asıl soru: **hangi tag'ler hayatta kaldı?** Üç metriği birden ölçüyoruz:
#
# | Metrik | Anlamı | Neden gerekli |
# |---|---|---|
# | `pct_present` | Tag dosyada var mı | Yokluk = kullanılamaz |
# | `pct_filled` | Değeri boş değil mi | De-id tag'i silmek yerine boşaltmış olabilir |
# | `n_unique` | Kaç farklı değer | 1 ise gruplama için işe yaramaz |
#
# `pct_present=100%` ama `pct_filled=0%` olan tag'ler çok yaygın bir tuzaktır.

# %%
present = Counter()
filled = Counter()
values = defaultdict(set)
vr_map = {}

for r in RECORDS:
    if not r["read_ok"]:
        continue
    for (tag, kw, vr), val in r["_tags"].items():
        key = (tag, kw)
        vr_map[key] = vr
        present[key] += 1
        if val is not None:
            filled[key] += 1
            if len(values[key]) < 2000:  # bellek koruma
                values[key].add(val)

inv = []
for key, n_pres in present.items():
    tag, kw = key
    vals = values[key]
    inv.append({
        "tag": tag,
        "keyword": kw,
        "VR": vr_map[key],
        "pct_present": round(100 * n_pres / n_files_ok, 2),
        "pct_filled": round(100 * filled[key] / n_files_ok, 2),
        "n_unique": len(vals) if len(vals) < 2000 else 2000,
        "ornek_degerler": " | ".join(sorted(vals, key=str)[:3]),
    })

inv_df = (
    pd.DataFrame(inv)
      .sort_values(["pct_filled", "pct_present", "keyword"], ascending=[False, False, True])
      .reset_index(drop=True)
)

print("=== FAZ 0.4a — TAG ENVANTERI ===")
print(f"{n_files_ok:,} dosyada toplam {len(inv_df)} FARKLI TAG bulundu.")
print("(Yarisma aciklamasi '86 allowlisted tag' diyor — karsilastirin.)\n")
print(f"  Tum dosyalarda mevcut (present=100%) : {(inv_df['pct_present'] == 100).sum()}")
print(f"  Tum dosyalarda dolu   (filled=100%)  : {(inv_df['pct_filled'] == 100).sum()}")
print(f"  Mevcut ama HIC dolu degil            : {(inv_df['pct_filled'] == 0).sum()}")
print()
inv_df

# %% [markdown]
# ## 9. FAZ 0.4b — Aradığımız kritik tag'ler var mı?
#
# `PHASES.md` 0.4'te adı geçen tag'leri tek tek kontrol ediyoruz. Bu tablo Faz 1C'deki
# fold stratejisinin ve Faz 0.5-0.7'deki ön işleme fonksiyonlarının girdisi.

# %%
CRITICAL = {
    "SITE PROXY (fold gruplamasi, Faz 1C)": [
        "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
        "InstitutionName", "StationName", "DeviceSerialNumber", "SoftwareVersions",
    ],
    "GEOMETRI (slice siralama, Faz 0.5)": [
        "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
        "SliceThickness", "SpacingBetweenSlices", "InstanceNumber", "PatientPosition",
    ],
    "LATERALITE (mirror, Faz 0.7)": [
        "Laterality", "ImageLaterality", "BodyPartExamined",
    ],
    "INTENSITE (normalizasyon, Faz 0.6)": [
        "RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth",
        "BitsStored", "BitsAllocated", "PixelRepresentation", "PhotometricInterpretation",
    ],
    "SEKANS (metadata enjeksiyonu, Faz 3.3)": [
        "SeriesDescription", "ProtocolName", "ScanningSequence", "SequenceVariant",
        "ScanOptions", "MRAcquisitionType", "EchoTime", "RepetitionTime", "InversionTime",
    ],
}

lookup = inv_df.drop_duplicates(subset="keyword").set_index("keyword")


def tag_row(kw):
    """Envanterden bir tag satirini getir; yoksa None."""
    if kw not in lookup.index:
        return None
    row = lookup.loc[kw]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


print("=== FAZ 0.4b — KRITIK TAG KONTROL LISTESI ===\n")
for group, kws in CRITICAL.items():
    print(f"--- {group} ---")
    for kw in kws:
        row = tag_row(kw)
        if row is None:
            print(f"  [YOK  ] {kw:<28} -- bu tag hic bulunamadi")
        else:
            mark = "OK " if row["pct_filled"] > 50 else "ZAYIF"
            print(f"  [{mark:<5}] {kw:<28} present={row['pct_present']:>6.1f}%  "
                  f"filled={row['pct_filled']:>6.1f}%  n_unique={row['n_unique']}")
    print()

# %% [markdown]
# ## 10. FAZ 0.4c — Site proxy analizi (fold stratejisini bu belirleyecek)
#
# Bir alanın site proxy'si olabilmesi için **üç koşulu birden** sağlaması gerekir:
#
# 1. **Dolu olmalı** — %50'nin altındaysa çoğu study'yi gruplayamazsın
# 2. **Study içinde tutarlı olmalı** — aynı study'nin iki serisinde farklı `Manufacturer`
#    çıkıyorsa grup anahtarı olamaz (study bölünür → tam da önlemek istediğimiz sızıntı)
# 3. **Makul sayıda grup üretmeli** — 1 grup gruplama değildir; 500 grup da GroupKFold'u
#    anlamsızlaştırır (her fold'da her grup temsil edilmeye başlar)
#
# 5-fold için pratik hedef: **~10-50 grup**, hiçbiri verinin yarısından fazlası değil.

# %%
meta = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                     for r in RECORDS if r["read_ok"]])

PROXY_CANDIDATES = [
    "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
    "InstitutionName", "StationName", "DeviceSerialNumber", "SoftwareVersions",
    "PatientPosition", "PhotometricInterpretation",
]

rows = []
for col in PROXY_CANDIDATES:
    if col not in meta.columns:
        continue
    s = meta[col]
    fill = float(s.notna().mean())
    if fill == 0:
        rows.append({"alan": col, "doluluk": 0.0, "n_unique": 0,
                     "study_ici_tutarli": np.nan, "en_buyuk_grup_payi": np.nan})
        continue
    per_study = meta.groupby("StudyInstanceUID")[col].nunique(dropna=True)
    consistent = float((per_study <= 1).mean())
    study_val = meta.groupby("StudyInstanceUID")[col].first()
    share = (float(study_val.value_counts(normalize=True).iloc[0])
             if study_val.notna().any() else np.nan)
    rows.append({
        "alan": col,
        "doluluk": round(fill, 3),
        "n_unique": int(s.nunique()),
        "study_ici_tutarli": round(consistent, 3),
        "en_buyuk_grup_payi": round(share, 3),
    })

proxy_df = pd.DataFrame(rows).sort_values("n_unique", ascending=False)
print("=== FAZ 0.4c — TEKIL ALAN PROXY DEGERLENDIRMESI ===\n")
print(proxy_df.to_string(index=False))
print("\n  doluluk            : degeri dolu olan dosya orani")
print("  n_unique           : kac farkli deger (1 ise gruplama icin ise yaramaz)")
print("  study_ici_tutarli  : bir study icinde tek deger goren study orani (1.0 olmali)")
print("  en_buyuk_grup_payi : en kalabalik grubun study payi (>0.5 ise dengesiz)")

# %%
# --- Kombinasyon anahtarlari ---
# Tek alan yetersizse (orn. sadece 3 uretici var) bilesik anahtar daha iyi ayirir.
print("\n=== KOMBINASYON GRUP ANAHTARLARI ===\n")

COMBOS = [
    ["Manufacturer"],
    ["Manufacturer", "MagneticFieldStrength"],
    ["Manufacturer", "ManufacturerModelName"],
    ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength"],
    ["InstitutionName"],
    ["StationName"],
]

study_level = meta.groupby("StudyInstanceUID").first()
combo_rows = []
for combo in COMBOS:
    cols = [c for c in combo if c in study_level.columns]
    if not cols:
        continue
    key = study_level[cols].fillna("NA").astype(str).agg(" | ".join, axis=1)
    vc = key.value_counts()
    combo_rows.append({
        "anahtar": " + ".join(cols),
        "n_grup": int(len(vc)),
        "en_buyuk_grup": round(float(vc.iloc[0] / len(key)), 3),
        "medyan_grup_boyutu": int(vc.median()),
        "tek_studyli_grup": int((vc == 1).sum()),
    })

combo_df = pd.DataFrame(combo_rows)
if len(combo_df):
    print(combo_df.to_string(index=False))
    print("\n  Hedef: n_grup 10-50, en_buyuk_grup < 0.5, tek_studyli_grup az.")
    print("  NOT: Rapor dili de bagimsiz bir site proxy'si (Faz 1.3). Nihai grup anahtari")
    print("       muhtemelen (dil + scanner) kombinasyonu olacak — karar Faz 1.12'de.")

# En umut verici alanlarin gercek deger dagilimi
for col in ["Manufacturer", "MagneticFieldStrength", "InstitutionName"]:
    if col in study_level.columns and study_level[col].notna().any():
        print(f"\n--- {col} dagilimi (study duzeyinde) ---")
        print(study_level[col].value_counts(dropna=False).head(20).to_string())

# %% [markdown]
# ## 11. Çıktıları kaydet
#
# CSV'ler `/kaggle/working` altına yazılır. Notebook'u **Save Version** ile kaydedince
# bunlar kalıcı output olur ve indirilebilir.

# %%
os.makedirs(OUT_DIR, exist_ok=True)

inv_df.to_csv(f"{OUT_DIR}/phase0_tag_inventory.csv", index=False)
ts_df.to_csv(f"{OUT_DIR}/phase0_transfer_syntax.csv", index=False)
decode_df.to_csv(f"{OUT_DIR}/phase0_decode_test.csv", index=False)
proxy_df.to_csv(f"{OUT_DIR}/phase0_site_proxy.csv", index=False)
meta.drop(columns=["path"], errors="ignore").to_csv(
    f"{OUT_DIR}/phase0_metadata_sample.csv", index=False
)
if len(combo_df):
    combo_df.to_csv(f"{OUT_DIR}/phase0_group_key_candidates.csv", index=False)

for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith("phase0"):
        print(f"  kaydedildi: {OUT_DIR}/{f}  ({os.path.getsize(OUT_DIR + '/' + f)/1024:.1f} KB)")

# %% [markdown]
# ## 12. Özet — bu bloğu kopyalayıp Claude Code'a yapıştırın
#
# Faz 1'e geçerken bu çıktı gerekli olacak.

# %%
lines = []
add = lines.append
add("=" * 70)
add("FAZ 0.3 + 0.4 SONUC OZETI")
add("=" * 70)
add(f"Orneklem: {meta['StudyInstanceUID'].nunique():,} study / "
    f"{meta['SeriesInstanceUID'].nunique():,} seri / {n_files_ok:,} dosya")
add("")
add("--- TRANSFER SYNTAX ---")
for _, r in ts_df.iterrows():
    sub_d = decode_df[decode_df["ts_name"] == r["ts_name"]]
    add(f"  {r['ts_name']:<44} {r['pct']:>6.2f}%  decode {int(sub_d['ok'].sum())}/{len(sub_d)}")
add("")
add("--- ETIKET DURUMU ---")
if present_labels:
    add(f"  gold (tum etiketler dolu): {n_gold:,} / {len(train):,} ({n_gold/len(train):.2%})")
    add(f"  etiketsiz                : {len(train)-n_any:,}")
add("")
add(f"--- TAG ENVANTERI: {len(inv_df)} farkli tag ---")
add("  Kritik alanlar:")
for kw in ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
           "InstitutionName", "StationName", "Laterality", "ImageLaterality",
           "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
           "RescaleSlope", "RescaleIntercept", "SeriesDescription"]:
    row = tag_row(kw)
    if row is None:
        add(f"    {kw:<26} YOK")
    else:
        add(f"    {kw:<26} filled={row['pct_filled']:>6.1f}%  n_unique={row['n_unique']}")
add("")
add("--- GRUP ANAHTARI ADAYLARI ---")
if len(combo_df):
    for _, r in combo_df.iterrows():
        add(f"  {r['anahtar']:<54} n_grup={r['n_grup']:<5} en_buyuk={r['en_buyuk_grup']}")
add("=" * 70)

report = "\n".join(lines)
print(report)
with open(f"{OUT_DIR}/phase0_summary.txt", "w", encoding="utf-8") as fh:
    fh.write(report)
