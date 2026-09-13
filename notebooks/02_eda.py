# %% [markdown]
# # Faz 1A — Keşifsel Veri Analizi + Lateralite Kapanışı
#
# Bu notebook iki işi birleştiriyor, çünkü ikisi de **aynı DICOM başlık taramasını**
# gerektiriyor ve Kaggle'da dosyaya dokunmak en pahalı işlem:
#
# | Görev | Ne üretiyor |
# |---|---|
# | 1.1 | gold / etiketsiz study oranı |
# | 1.2 | gold alt kümede per-label pozitif dağılımı |
# | 1.3 | rapor dili dağılımı (site proxy'sinin ikinci bacağı) |
# | 1.4 | rapor uzunluk dağılımı |
# | 1.5 | bulgular arası ko-okürans |
# | 1.6 | seri sayısı / düzlem / Fluid_Sensitive × Fat_Suppression |
# | 1.7 | **fold grup anahtarı — tüm veri üzerinde**, (dil × scanner) çaprazı dahil |
# | 0.7 kalanı | lateralite **study düzeyi** kapsamı + katman 3'ün yeni isabeti |
#
# ## Üç tasarım kararı
#
# **1. Örneklem değil, TAM veri.** Faz 0'da 500 study örneklemiştik; envanter çıkarmak
# için yeterliydi. Burada çıktı `study_metadata.csv` ve bu dosya fold ataması için
# kullanılacak — her study'nin grup anahtarı bilinmek zorunda. Örneklem yetmez.
#
# **2. Seri başına İKİ dosya: ilk ve SON slice.** Tek dosya okumak sagittal serilerde
# merkez-x hesabını bozar. Sebep geometrik: sagittal'de hastanın x ekseni *slice
# eksenidir*, yani ilk slice dizin bir kenarında, son slice öbür kenarındadır. İlk
# slice'ın x'i dizin merkezi değil. İlk ve sonun ortalaması ise tam merkeze denk gelir.
# Coronal/axial'de x slice'lar arasında sabittir, dolayısıyla ortalama almak zarar vermez.
# Tek formül iki düzlemde de doğru çalışır.
#
# **3. Lateralite kapsamı study düzeyinde ölçülüyor.** Bir study'de tek diz var. Faz
# 0.7'de ölçtüğümüz %48.3 *seri* düzeyiydi; bir study'nin herhangi bir serisi tarafı
# biliyorsa bütün study'nin tarafı bilinir. Bu yeni bir heuristic değil, saf agregasyon.
#
# **Çalışma süresi:** ~20-35 dakika (çoğu DICOM başlık taraması). CPU yeterli.

# %% [markdown]
# ## 0. Yapılandırma

# %%
SEED = 42

# Kac study taranacak? None = HEPSI (~4.400). Hizli deneme icin 300 yapip tekrar cevir.
N_STUDIES = None

# Seri basina okunacak dosya: ilk + son. Sagittal merkez-x icin 2 sart (yukari bak).
FILES_PER_SERIES = 2

MAX_WORKERS = 16
OUT_DIR = "/kaggle/working"

# Katman 3 esik taramasi: merkez-x'in kac mm otesinde karar verilsin?
DEADZONE_GRID = [0, 5, 10, 20, 30, 50, 75, 100]

# Katman 3'u acmak icin gereken minimum isabet. Altindaysa kapali kalir.
# Neden 0.90: yanlis aynalama 4 medial/lateral etiketine geometrik gurultu enjekte eder.
MIN_LAYER3_ACCURACY = 0.90

# %% [markdown]
# ## 1. Modülü diske yaz
#
# `src/data/dicom_io.py`'nin tek doğru sürümü repo'dadır. Aşağıdaki hücre
# `py2ipynb.py` dönüştürmesi sırasında modülün içeriğiyle doldurulur — yani
# notebook'taki kopya asla elle düzenlenmez, bayatlamaz.

# %%
#!writefile src/data/dicom_io.py /kaggle/working/dicom_io.py

# %%
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, "/kaggle/working")
import dicom_io as dio

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 60)

DATA = dio.find_data_root()
print("VERI YOLU:", DATA)
print("pydicom  :", pydicom.__version__)

# %% [markdown]
# ## 2. FAZ 1.1 — Gold / etiketsiz oranı

# %%
train = pd.read_csv(f"{DATA}/train.csv")
train_series = pd.read_csv(f"{DATA}/train_series.csv")

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
LABELS = [c for c in LABELS if c in train.columns]

lab = train[LABELS]
is_gold = lab.notna().all(axis=1)
has_any = lab.notna().any(axis=1)

print(f"Toplam train study       : {len(train):>6,}")
print(f"Gold (tum etiketler dolu): {is_gold.sum():>6,}  ({is_gold.mean():.2%})")
print(f"En az bir etiketi dolu   : {has_any.sum():>6,}  ({has_any.mean():.2%})")
print(f"Hic etiketi olmayan      : {(~has_any).sum():>6,}  ({(~has_any).mean():.2%})")

# Kismi etiketli study var mi? Varsa weak-label pipeline'da farkli ele alinmali.
partial = has_any & ~is_gold
print(f"KISMI etiketli           : {partial.sum():>6,}")
if partial.sum() > 0:
    print("  -> Bu study'ler hem gold hem weak sinyal tasiyor; etiket bazinda maskeleme gerekir.")

print(f"\nRapor metni dolu olan study: {train['Report'].notna().sum():,} "
      f"({train['Report'].notna().mean():.2%})")

# %% [markdown]
# ## 3. FAZ 1.2 — Gold alt kümede per-label dağılım
#
# Buradaki asıl mesele sayıların kendisi değil, **ne kadar az** oldukları. Bir etiketin
# ROC-AUC'sini anlamlı biçimde ölçebilmek için her iki sınıftan yeterli örnek gerekir.
# 5'ten az pozitifi olan bir etikette AUC neredeyse tamamen gürültüdür: tek bir vakanın
# sıralaması metriği uçtan uca oynatır.
#
# Aşağıdaki tabloda `n_poz` sütununa bak. Kırmızı bayrak: `n_poz < 5`.

# %%
gold = train[is_gold]
rows = []
for c in LABELS:
    v = gold[c].astype(float)
    n_pos = int((v == 1).sum())
    n_neg = int((v == 0).sum())
    rows.append({
        "etiket": c,
        "n_poz": n_pos,
        "n_neg": n_neg,
        "prevalans": round(n_pos / max(n_pos + n_neg, 1), 3),
        "AUC_olculebilir": "HAYIR" if min(n_pos, n_neg) < 5 else "zayif" if min(n_pos, n_neg) < 15 else "evet",
    })
dist = pd.DataFrame(rows).sort_values("n_poz")
print(f"=== GOLD ALT KUMESI: {len(gold)} study ===\n")
print(dist.to_string(index=False))

n_bad = (dist["AUC_olculebilir"] == "HAYIR").sum()
n_weak = (dist["AUC_olculebilir"] == "zayif").sum()
print(f"\n  AUC olculemez (<5 ornek) : {n_bad} / {len(LABELS)} etiket")
print(f"  Zayif (<15 ornek)        : {n_weak} / {len(LABELS)} etiket")
print("\n  Bu tablo, gold-only validation'in TEK BASINA yeterli olup olmadigini soyler.")
print("  Cok sayida 'HAYIR' varsa PHASE0_FINDINGS Bolum 3'teki (B) secenegi")
print("  (kendi gold setimizi uretmek) zorunlu hale gelir.")

# %% [markdown]
# ## 4. FAZ 1.5 — Ko-okürans
#
# Hangi bulgular birlikte görülüyor? İki nedenle önemli:
#
# 1. **Weak-label doğrulaması için sağlama:** LLM çıkarımı `Effusion` ve `Synovitis`'i
#    hiç birlikte işaretlemiyorsa, bu prompt'ta bir sorun olduğunun işaretidir —
#    çünkü klinik olarak sık beraber görülürler.
# 2. **Model tasarımı:** güçlü korelasyon varsa 12 bağımsız sigmoid bir miktar bilgi
#    kaybeder; etiketler arası yapıdan faydalanmak Faz 3'te bir seçenek.
#
# **Uyarı:** 58 study üzerinde hesaplanan korelasyon çok gürültülü. Yönü göstermek
# için okunur, kesin bir büyüklük olarak değil.

# %%
if len(gold) >= 10:
    g = gold[LABELS].astype(float)
    # Jaccard: birlikte pozitif / en az birinde pozitif. Prevalans farklarina karsi
    # korelasyondan daha okunabilir bir olcu.
    J = pd.DataFrame(index=LABELS, columns=LABELS, dtype=float)
    for a in LABELS:
        for b in LABELS:
            both = ((g[a] == 1) & (g[b] == 1)).sum()
            either = ((g[a] == 1) | (g[b] == 1)).sum()
            J.loc[a, b] = round(both / either, 2) if either else np.nan
    print("=== JACCARD (birlikte poz / en az birinde poz) ===\n")
    print(J.to_string())

    pairs = []
    for i, a in enumerate(LABELS):
        for b in LABELS[i + 1:]:
            both = int(((g[a] == 1) & (g[b] == 1)).sum())
            if both > 0:
                pairs.append((a, b, both, J.loc[a, b]))
    pairs.sort(key=lambda x: -x[2])
    print("\n--- En sik birlikte gorulen ciftler ---")
    for a, b, n, j in pairs[:10]:
        print(f"  {a:<18} + {b:<18} n={n:<3} jaccard={j}")
else:
    print("Gold alt kume ko-okurans icin cok kucuk.")

# %% [markdown]
# ## 5. FAZ 1.4 — Rapor uzunluğu
#
# Çok kısa / telegrafik raporlar LLM çıkarımını zorlaştırır: "Normal knee MRI." gibi bir
# rapordan 12 bulgunun hepsini `absent` çıkarmak doğru mudur, yoksa `uncertain` mi?
# Bu dağılım, prompt tasarımında (Faz 1.8) kısa raporlar için özel talimat gerekip
# gerekmediğini söyler.

# %%
rep = train["Report"].fillna("")
n_char = rep.str.len()
n_word = rep.str.split().str.len().fillna(0).astype(int)

print("Karakter sayisi:")
print(n_char.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
print("\nKelime sayisi:")
print(n_word.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())

for thr in (0, 20, 50, 100):
    n = int((n_char <= thr).sum())
    print(f"  <= {thr:>4} karakter: {n:>5,} study ({n/len(train):6.2%})")

print("\n--- En kisa 5 rapor ---")
for t in rep[n_char > 0].sort_values(key=lambda s: s.str.len()).head(5):
    print("  ", repr(t[:120]))

# %% [markdown]
# ## 6. FAZ 1.3 — Rapor dili
#
# Dil iki işe yarıyor: (a) fold gruplaması için bağımsız bir site sinyali — her merkez
# büyük olasılıkla tek dilde rapor yazıyor; (b) weak-label prompt'unun hangi dilleri
# kapsaması gerektiğini söyler.
#
# **Yöntem seçimi:** `fasttext` lid.176 modeli tercih ediliyor çünkü kısa metinlerde
# `langdetect`'ten belirgin şekilde daha isabetli ve 176 dili kapsıyor. `.ftz`
# (sıkıştırılmış) sürümü yalnızca ~917 KB. İnternet kapalıysa veya kurulum başarısız
# olursa `langdetect`'e, o da yoksa yazı-sistemi (script) tabanlı kaba bir ayrıma düşer.

# %%
# Her katmani DONDURMEDEN ONCE gercekten cagirip sinayan prob metinleri.
# Bunlar olmadan bir dedektor "kurulabildigi icin" gecerli sayilir ve hata
# 4.400 raporun ilkinde, yani pahali tarama hucresine gelmeden patlar.
_LANG_PROBES = [
    ("Normal knee MRI. No meniscal tear.", "en"),
    ("Kniegelenk unauffaellig, kein Erguss.", "de"),
    ("Rodilla sin lesiones meniscales.", "es"),
]


def _validate_detector(fn, name):
    """Dedektoru prob metinlerinde CAGIR. Patlarsa veya hicbir sey bulamazsa reddet."""
    try:
        out = [fn(t) for t, _ in _LANG_PROBES]
    except Exception as e:
        print(f"  [RED]  {name}: cagri hatasi -> {type(e).__name__}: {e}")
        return False
    if all(o == "?" for o in out):
        print(f"  [RED]  {name}: cagri calisiyor ama hicbir dil tespit edemedi")
        return False
    print(f"  [OK]   {name}: {out}")
    return True


def _fasttext_predict_factory(model):
    """numpy surumune gore calisan bir predict yolu sec.

    `fasttext`in kendi `predict()`i numpy>=2 ile KIRIK: FastText.py icinde
    `np.array(probs, copy=False)` var, numpy 2.0 bunu ValueError ile reddediyor
    ("Unable to avoid copy while creating an array as requested").
    Alt seviye `model.f.predict()` bu sarmalayiciyi atlar ve (prob, etiket)
    ciftlerinden olusan sade bir liste dondurur.
    """
    def via_wrapper(t):
        lab, prob = model.predict(t, k=1)
        return (lab[0], float(prob[0])) if len(lab) else (None, 0.0)

    def via_raw(t):
        preds = model.f.predict(t, 1, 0.0, "strict")
        if not preds:
            return (None, 0.0)
        prob, lab = preds[0]          # FastText.py'de sira: (prob, label)
        return (lab, float(prob))

    for impl, tag in ((via_wrapper, "model.predict()"),
                      (via_raw, "model.f.predict()  [numpy>=2 yolu]")):
        try:
            impl("test metni")
            print(f"  fasttext cagri yolu: {tag}")
            return impl
        except Exception as e:
            print(f"  fasttext {tag} kullanilamadi: {type(e).__name__}")
    return None


def build_lang_detector():
    """(fonksiyon, yontem_adi) dondur. En iyiden en kabaya, her katmani SINAYARAK."""
    print("Dil tespiti katmanlari deneniyor:")

    # --- 1) fasttext lid.176 ---
    try:
        try:
            import fasttext as ft
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fasttext-wheel"],
                           check=True, timeout=300)
            import fasttext as ft

        # Model dosyasi OUT_DIR'e DEGIL gecici dizine iniyor: /kaggle/working
        # commit edilen cikti klasorudur, oraya 917 KB'lik bir model koymak
        # her sürüme gereksiz yuk bindirir.
        import tempfile
        mp = os.path.join(tempfile.gettempdir(), "lid.176.ftz")
        if not os.path.exists(mp):
            import urllib.request
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz", mp)
        model = ft.load_model(mp)

        predict = _fasttext_predict_factory(model)
        if predict is not None:
            def detect(text):
                t = " ".join(str(text).split())[:1000]
                if len(t) < 3:
                    return "?"
                lab, prob = predict(t)
                if lab is None or prob < 0.30:
                    return "?"
                return lab.replace("__label__", "")

            if _validate_detector(detect, "fasttext-lid176"):
                return detect, "fasttext-lid176"
    except Exception as e:
        print(f"  [RED]  fasttext hazirlanamadi: {type(e).__name__}: {e}")

    # --- 2) langdetect ---
    try:
        try:
            from langdetect import detect as _ld, DetectorFactory
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "langdetect"],
                           check=True, timeout=300)
            from langdetect import detect as _ld, DetectorFactory
        DetectorFactory.seed = SEED

        def detect(text):
            t = " ".join(str(text).split())
            if len(t) < 10:
                return "?"
            try:
                return _ld(t)
            except Exception:
                return "?"

        if _validate_detector(detect, "langdetect"):
            return detect, "langdetect"
    except Exception as e:
        print(f"  [RED]  langdetect hazirlanamadi: {type(e).__name__}: {e}")

    # --- 3) script tabanli kaba ayrim (saf stdlib, kirilamaz) ---
    import unicodedata

    def detect(text):
        t = str(text)
        if len(t.strip()) < 3:
            return "?"
        names = [unicodedata.name(ch, "") for ch in t if ch.isalpha()][:200]
        for tag, key in [("CYRILLIC", "ru?"), ("ARABIC", "ar?"), ("HEBREW", "he?"),
                         ("CJK", "zh/ja?"), ("HANGUL", "ko?"), ("GREEK", "el?"),
                         ("THAI", "th?"), ("DEVANAGARI", "hi?")]:
            if any(tag in n for n in names):
                return key
        return "latin?"

    _validate_detector(detect, "script-fallback")
    print("\n  !! UYARI: Latin alfabeli diller birbirinden AYIRT EDILEMIYOR.")
    print("  !! Dil, site proxy'sinin ikinci bacagi (Faz 1.12) — bu haliyle kullanilamaz.")
    print("  !! Notebook'un geri kalani calisir; dil sonuclarini yok sayin.")
    return detect, "script-fallback"


detect_lang, lang_method = build_lang_detector()
print(f"Dil tespiti yontemi: {lang_method}")

t0 = time.time()

# Tek bir bozuk rapor 30 dakikalik koşuyu dusurmesin: rapor basina hatayi yut,
# ama kac tanesinin patladigini SAY ve bildir (sessizce yutmak daha kotu olurdu).
_lang_err = Counter()


def _safe_lang(t):
    try:
        return detect_lang(t)
    except Exception as e:
        _lang_err[type(e).__name__] += 1
        return "?"


train["lang"] = [_safe_lang(t) for t in rep]
print(f"{len(train):,} rapor {time.time()-t0:.1f}s icinde siniflandirildi")
if _lang_err:
    print(f"  !! {sum(_lang_err.values())} raporda hata (yutuldu, '?' atandi): {dict(_lang_err)}")
print()

lang_vc = train["lang"].value_counts(dropna=False)
lang_tab = pd.DataFrame({"n_study": lang_vc, "pay": (lang_vc / len(train)).round(4)})
print("=== DIL DAGILIMI ===\n")
print(lang_tab.to_string())
print(f"\n  Tespit edilen dil sayisi: {(lang_vc.index != '?').sum()}")
print(f"  Siniflandirilamayan     : {int(lang_vc.get('?', 0)):,}")
print("  (Yarisma aciklamasi '12 dil' diyor — karsilastirin.)")

# Dil, gold etiketle iliskili mi? Iliskiliyse gold alt kume TEK bir merkeze ait
# olabilir ve validation'in genellenebilirligi daha da suphelidir.
print("\n--- Gold study'lerin dil dagilimi (dagilim kaymasi kontrolu) ---")
gold_lang = train.loc[is_gold, "lang"].value_counts()
cmp_lang = pd.DataFrame({
    "gold_n": gold_lang,
    "gold_pay": (gold_lang / max(is_gold.sum(), 1)).round(3),
    "tum_pay": (lang_vc / len(train)).round(3),
}).fillna(0).sort_values("gold_n", ascending=False)
print(cmp_lang.head(15).to_string())
print("\n  gold_pay ile tum_pay birbirinden cok farkliysa gold alt kume")
print("  belirli merkez(ler)den geliyor -> validation genellenebilirligi dusuk.")

# %% [markdown]
# ## 7. FAZ 1.6 — Seri metadata dağılımları

# %%
spc = train_series.groupby("StudyInstanceUID").size()
print("Study basina seri sayisi:")
print(spc.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
print("\nDagilim:")
print(spc.value_counts().sort_index().to_string())

print("\n=== DUZLEM DAGILIMI (seri duzeyinde) ===")
print(train_series["Anatomical_Plane"].value_counts(dropna=False).to_string())

print("\n=== Fluid_Sensitive x Fat_Suppression ===")
ct = pd.crosstab(train_series["Fluid_Sensitive"], train_series["Fat_Suppression"],
                 margins=True)
print(ct.to_string())
print("\n  PROJECT_OVERVIEW notu: bu ikisi korele ama 'her vaka icin esdegil degil'.")
print("  Yukaridaki tabloda kosegen disi hucreler doluysa AYRI feature tutmak dogru.")

print("\n=== DUZLEM x FLUID_SENSITIVE ===")
print(pd.crosstab(train_series["Anatomical_Plane"], train_series["Fluid_Sensitive"]).to_string())

# Her study'de her duzlemden en az bir seri var mi? (Faz 2.1 seri secimi buna dayaniyor)
plane_cov = train_series.pivot_table(index="StudyInstanceUID", columns="Anatomical_Plane",
                                     aggfunc="size", fill_value=0)
print("\n=== DUZLEM KAPSAMI (study duzeyinde) ===")
for c in plane_cov.columns:
    n = int((plane_cov[c] > 0).sum())
    print(f"  {c:<10} en az 1 seri: {n:>6,} / {len(plane_cov):,}  ({n/len(plane_cov):6.2%})")
all3 = int((plane_cov > 0).all(axis=1).sum())
print(f"  UC DUZLEMIN HEPSI      : {all3:>6,} / {len(plane_cov):,}  ({all3/len(plane_cov):6.2%})")
print("\n  Bu oran dusukse Faz 2.1'deki 'her duzlemden bir seri' stratejisi")
print("  study'lerin bir kismi icin eksik girdi uretir -> maskeleme gerekir.")

# %% [markdown]
# ## 8. DICOM başlık taraması
#
# Buradan sonrası DICOM başlıkları gerektiriyor. Her seriden **ilk ve son** dosyayı
# okuyoruz (gerekçesi başlıktaki 2. tasarım kararı). Piksel okunmuyor.

# %%
studies = train_series["StudyInstanceUID"].drop_duplicates()
if N_STUDIES is not None:
    studies = studies.sample(n=min(N_STUDIES, len(studies)), random_state=SEED)
sel = train_series[train_series["StudyInstanceUID"].isin(set(studies))]
print(f"Taranacak: {sel['StudyInstanceUID'].nunique():,} study / {len(sel):,} seri")

HEADER_FIELDS = [
    "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
    "SoftwareVersions", "Laterality", "SeriesDescription", "ProtocolName",
    "BodyPartExamined", "PixelSpacing", "Rows", "Columns", "SliceThickness",
]


def scan_series(row):
    study_uid, series_uid = row
    sdir = dio.series_dir(DATA, study_uid, series_uid)
    rec = {"StudyInstanceUID": study_uid, "SeriesInstanceUID": series_uid,
           "n_files": 0, "ok": False}
    paths = dio.list_dicom_files(sdir)
    rec["n_files"] = len(paths)
    if not paths:
        return rec
    # ilk + son: sagittal'de merkez-x ancak iki ucun ortalamasiyla dogru cikar
    pick = [paths[0]] if len(paths) == 1 else [paths[0], paths[-1]]
    dss = []
    for p in pick[:FILES_PER_SERIES]:
        try:
            dss.append(pydicom.dcmread(p, stop_before_pixels=True))
        except Exception:
            pass
    if not dss:
        return rec
    rec["ok"] = True
    ds0 = dss[0]
    for f in HEADER_FIELDS:
        v = ds0.get(f, None)
        if v is None:
            rec[f] = None
        elif f == "PixelSpacing":
            rec[f] = float(v[0])
        else:
            s = str(v).strip()
            rec[f] = s if s else None
    iop = ds0.get("ImageOrientationPatient", None)
    rec["plane_calc"] = dio.plane_from_iop(iop)
    rec["center_x"] = dio.series_center_x(dss)
    side, src = dio.detect_laterality(dss, use_ipp_layer=False)
    rec["lat_12"] = side          # katman 1+2 (tag + metin)
    rec["lat_12_src"] = src
    rec["lat_tag"] = (str(ds0.get("Laterality", "") or "").strip().upper()[:1] or None)
    if rec["lat_tag"] not in ("L", "R"):
        rec["lat_tag"] = None
    return rec


t0 = time.time()
pairs = list(sel[["StudyInstanceUID", "SeriesInstanceUID"]].itertuples(index=False, name=None))
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    recs = list(ex.map(scan_series, pairs))
el = time.time() - t0

ser = pd.DataFrame(recs)
n_ok = int(ser["ok"].sum())
print(f"Tarama: {el:.1f}s  ({1000*el/max(len(ser),1):.0f} ms/seri)  "
      f"basarili {n_ok:,}/{len(ser):,}")
if n_ok < len(ser):
    print(f"  !! {len(ser)-n_ok} seri okunamadi — bos klasor veya bozuk dosya")

ser = ser[ser["ok"]].copy()
for c in ("Manufacturer", "ManufacturerModelName"):
    ser[c + "_n"] = ser[c].fillna("NA").str.strip().str.upper()

# %% [markdown]
# ## 9. FAZ 1.7 — Fold grup anahtarı (tüm veri üzerinde)
#
# Faz 0.4'te 500 study örnekleminde `Manufacturer + ManufacturerModelName` seçilmişti.
# Burada aynı hesabı **tüm veri** üzerinde tekrarlıyor ve dili de ekliyoruz.
#
# Hatırlatma (`ARCHITECTURE.md` Bölüm 3): hedef, yeterli grup sayısı ve dengeyi sağlayan
# **en KABA** anahtar. Anahtarı inceltmek bir cihazı ikiye bölme riski getirir ve
# sızıntıyı geri çağırır — yani daha fazla grup otomatik olarak daha iyi değildir.

# %%
# Normalizasyon gercekten gerekli miydi? (Faz 0.4'teki "11 tekil uretici" suphesi)
raw_mfr = ser["Manufacturer"].dropna().nunique()
nrm_mfr = ser["Manufacturer_n"].replace("NA", np.nan).dropna().nunique()
print(f"Manufacturer tekil: ham={raw_mfr}  normalize(strip+upper)={nrm_mfr}")
if raw_mfr != nrm_mfr:
    print(f"  -> Normalizasyon {raw_mfr - nrm_mfr} yazim varyantini birlestirdi. Gerekliydi.")
    print("  Ham degerler:", sorted(ser['Manufacturer'].dropna().unique())[:15])
else:
    print("  -> Yazim varyanti yok; normalizasyon zararsiz.")

# Study duzeyine indir. Faz 0.4'te study_ici_tutarli=1.000 olcusuldu; dogrulayalim.
print("\n--- Study ici tutarlilik (tum veri) ---")
for c in ["Manufacturer_n", "ManufacturerModelName_n", "MagneticFieldStrength"]:
    nun = ser.groupby("StudyInstanceUID")[c].nunique(dropna=True)
    frac = float((nun <= 1).mean())
    print(f"  {c:<28} tek degerli study orani: {frac:.4f}"
          + ("" if frac > 0.999 else "   <-- DIKKAT: bazi study'ler bolunuyor!"))

stu = ser.groupby("StudyInstanceUID").agg(
    Manufacturer=("Manufacturer_n", "first"),
    Model=("ManufacturerModelName_n", "first"),
    FieldStrength=("MagneticFieldStrength", "first"),
    n_series=("SeriesInstanceUID", "nunique"),
).reset_index()
stu = stu.merge(train[["StudyInstanceUID", "lang"]], on="StudyInstanceUID", how="left")
stu["is_gold"] = stu["StudyInstanceUID"].isin(train.loc[is_gold, "StudyInstanceUID"])


def key_stats(name, cols):
    k = stu[cols].fillna("NA").astype(str).agg(" | ".join, axis=1)
    vc = k.value_counts()
    return {
        "anahtar": name,
        "n_grup": len(vc),
        "en_buyuk_pay": round(float(vc.iloc[0] / len(k)), 3),
        "medyan_boyut": int(vc.median()),
        "tek_studyli": int((vc == 1).sum()),
        "gold_bulunan_grup": int(k[stu["is_gold"].values].nunique()),
    }


cands = [
    ("Manufacturer", ["Manufacturer"]),
    ("Manufacturer + Model", ["Manufacturer", "Model"]),
    ("Manufacturer + Model + Field", ["Manufacturer", "Model", "FieldStrength"]),
    ("lang", ["lang"]),
    ("lang + Manufacturer", ["lang", "Manufacturer"]),
    ("lang + Manufacturer + Model", ["lang", "Manufacturer", "Model"]),
]
keys = pd.DataFrame([key_stats(n, c) for n, c in cands])
print("\n=== GRUP ANAHTARI ADAYLARI (tum veri, study duzeyinde) ===\n")
print(keys.to_string(index=False))
print("\n  n_grup            : 5-fold icin ~10-50 ideal")
print("  en_buyuk_pay      : < 0.5 olmali")
print("  tek_studyli       : cok olmasi anahtarin fazla ince oldugunun isareti")
print("  gold_bulunan_grup : 58 gold study kac ayri gruba dagilmis?")
print("                      Az ise gold alt kume birkac cihazdan geliyor demektir.")

print("\n--- Manufacturer dagilimi (study duzeyinde) ---")
print(stu["Manufacturer"].value_counts(dropna=False).head(15).to_string())

# Dil ile scanner iliskili mi? Iliskiliyse caprazi almak anahtari bosa inceltir.
if stu["lang"].notna().any():
    print("\n--- Dil x Manufacturer (ilk 8x8) ---")
    print(pd.crosstab(stu["lang"], stu["Manufacturer"]).iloc[:8, :8].to_string())

# %% [markdown]
# ## 10. FAZ 0.7 kalanı — Lateralite
#
# İki soru:
#
# **A. Study düzeyi kapsam.** Bir study'de tek diz var; herhangi bir serisi tarafı
# biliyorsa study'nin tarafı bilinir. Kapsam ne kadar yükseliyor?
#
# **B. Katman 3'ün yeni isabeti.** Merkez-x düzeltmesi (köşe → merkez) düzlem sapmasını
# kaldırdı. Tag'i dolu olan alt kümede isabet artık ne?
#
# Karar kuralı: isabet **%90'ın altındaysa katman 3 KAPALI kalır.** Yanlış aynalama,
# 4 medial/lateral etiketine geometrik gürültü enjekte eder ve bu, eksik aynalamadan
# daha kötüdür.

# %%
print("=== A) LATERALITE KAPSAMI: SERI vs STUDY ===\n")
ser_cov_tag = float(ser["lat_tag"].notna().mean())
ser_cov_12 = float(ser["lat_12"].notna().mean())

# Study duzeyi: study'nin herhangi bir serisi karar verdiyse study kararlidir.
g = ser.groupby("StudyInstanceUID")
stu_tag = g["lat_tag"].apply(lambda s: s.dropna().unique())
stu_12 = g["lat_12"].apply(lambda s: s.dropna().unique())
stu_cov_tag = float((stu_tag.apply(len) > 0).mean())
stu_cov_12 = float((stu_12.apply(len) > 0).mean())

print(f"  {'':<26}{'seri':>10}{'study':>10}{'kazanc':>10}")
print(f"  {'Laterality tag':<26}{ser_cov_tag:>9.1%}{stu_cov_tag:>10.1%}"
      f"{stu_cov_tag-ser_cov_tag:>+10.1%}")
print(f"  {'Katman 1+2 (tag+metin)':<26}{ser_cov_12:>9.1%}{stu_cov_12:>10.1%}"
      f"{stu_cov_12-ser_cov_12:>+10.1%}")

# Celiski var mi? Ayni study'nin iki serisi farkli taraf diyorsa bir yerde hata var.
confl_tag = int((stu_tag.apply(len) > 1).sum())
confl_12 = int((stu_12.apply(len) > 1).sum())
print(f"\n  CELISKILI study (tag)      : {confl_tag}")
print(f"  CELISKILI study (katman1+2): {confl_12}")
if confl_12:
    print("  !! Bir study'de iki farkli taraf gorunuyor. Ya iki diz ayni study'de,")
    print("     ya metin heuristic'i yaniliyor. Ornekleri asagida inceleyin.")
    bad = stu_12[stu_12.apply(len) > 1].index[:5]
    print(ser[ser["StudyInstanceUID"].isin(bad)]
          [["StudyInstanceUID", "SeriesDescription", "lat_tag", "lat_12", "lat_12_src"]]
          .to_string(index=False))

print("\n  Kaynak dagilimi (karar veren seriler):")
print(ser.loc[ser["lat_12"].notna(), "lat_12_src"].value_counts().to_string())

# %%
print("\n=== B) KATMAN 3 — MERKEZ-X DUZELTMESINDEN SONRA ===\n")

ev = ser[ser["lat_tag"].notna() & ser["center_x"].notna()].copy()
print(f"Degerlendirme alt kumesi: {len(ev):,} seri (tag'i dolu VE merkez-x hesaplanabilir)")

if len(ev) == 0:
    print("!! Degerlendirilecek seri yok.")
else:
    print("\n--- merkez-x dagilimi, tag'e gore ---")
    print(ev.groupby("lat_tag")["center_x"]
            .describe(percentiles=[0.05, 0.5, 0.95])
            .round(1).to_string())

    # Faz 0.7'deki asil sorun: duzlemler arasi sistematik sapma. Kalkti mi?
    print("\n--- merkez-x MEDYANI: duzlem x taraf ---")
    piv = ev.pivot_table(index="plane_calc", columns="lat_tag",
                         values="center_x", aggfunc="median").round(1)
    print(piv.to_string())
    print("\n  DUZELTME CALISTIYSA: her satirda L pozitif, R negatif olmali ve")
    print("  ayni tarafin degeri duzlemler arasinda BENZER olmali (eskiden coronal/axial")
    print("  yarim FOV kadar negatife kaymisti).")

    print("\n--- ESIK TARAMASI ---")
    rows = []
    for dz in DEADZONE_GRID:
        dec = ev[ev["center_x"].abs() > dz]
        if len(dec) == 0:
            continue
        pred = np.where(dec["center_x"] > 0, "L", "R")
        acc = float((pred == dec["lat_tag"].values).mean())
        rows.append({"olu_bolge_mm": dz,
                     "kapsam": round(len(dec) / len(ev), 3),
                     "isabet": round(acc, 4),
                     "n_karar": len(dec),
                     "n_hata": int((pred != dec["lat_tag"].values).sum())})
    sweep = pd.DataFrame(rows)
    print(sweep.to_string(index=False))

    ok = sweep[sweep["isabet"] >= MIN_LAYER3_ACCURACY]
    print(f"\n  Faz 0.7'deki eski (kose-tabanli) isabet: 0.861")
    best_acc = float(sweep["isabet"].max())
    print(f"  Yeni en iyi isabet                     : {best_acc:.4f}")
    print(f"  Esik ({MIN_LAYER3_ACCURACY:.2f}) asildi mi?                 : "
          f"{'EVET' if len(ok) else 'HAYIR'}")

    if len(ok):
        # Esigi gecenler arasinda EN YUKSEK KAPSAMLI olani sec.
        pick = ok.sort_values("kapsam", ascending=False).iloc[0]
        print(f"\n  >> KATMAN 3 ACILABILIR.")
        print(f"     olu_bolge = {pick['olu_bolge_mm']} mm  ->  "
              f"kapsam {pick['kapsam']:.1%}, isabet {pick['isabet']:.1%}")
        print(f"     Kullanim: detect_laterality(dss, use_ipp_layer=True, "
              f"ipp_deadzone={pick['olu_bolge_mm']})")
        LAYER3_OK, LAYER3_DZ = True, float(pick["olu_bolge_mm"])
    else:
        print(f"\n  >> KATMAN 3 KAPALI KALIYOR. Yanlis aynalama eksik aynalamadan kotu.")
        LAYER3_OK, LAYER3_DZ = False, None

    # Hatalar nerede yogunlasiyor? Duzlem bazliysa duzeltme eksik kalmis olabilir.
    dec = ev[ev["center_x"].abs() > (LAYER3_DZ or 0)]
    if len(dec):
        pred = np.where(dec["center_x"] > 0, "L", "R")
        err = dec[pred != dec["lat_tag"].values]
        if len(err):
            print(f"\n  --- {len(err)} hatanin duzlem dagilimi ---")
            print((err["plane_calc"].value_counts() /
                   dec["plane_calc"].value_counts()).dropna().round(3).to_string())
            print("  Tek bir duzlemde yogunlasmissa merkez hesabi o duzlemde hala eksik.")

# %% [markdown]
# ## 11. Nihai lateralite kapsamı ve çıktı dosyası
#
# `study_metadata.csv` — Faz 1.12/1.13'ün (fold ataması) ve Faz 2.1'in (seri seçimi)
# doğrudan girdisi. Her study için bir satır.

# %%
def study_side(arr):
    return arr[0] if len(arr) == 1 else (None if len(arr) == 0 else "CELISKI")


side_12 = stu_12.apply(study_side)

if len(ev) and LAYER3_OK:
    ser["lat_3"] = np.where(
        ser["center_x"].notna() & (ser["center_x"].abs() > LAYER3_DZ),
        np.where(ser["center_x"].fillna(0) > 0, "L", "R"), None)
    ser["lat_final"] = ser["lat_12"].fillna(pd.Series(ser["lat_3"], index=ser.index))
    stu_final = ser.groupby("StudyInstanceUID")["lat_final"].apply(
        lambda s: study_side(s.dropna().unique()))
    layer3_note = f"acik (olu_bolge={LAYER3_DZ} mm)"
else:
    stu_final = side_12
    layer3_note = "kapali"

meta = stu.copy()
meta["laterality"] = meta["StudyInstanceUID"].map(stu_final)
meta["lat_from_tag"] = meta["StudyInstanceUID"].map(stu_tag.apply(study_side))
meta["group_key"] = (meta[["Manufacturer", "Model"]]
                     .fillna("NA").astype(str).agg(" | ".join, axis=1))
meta["n_char_report"] = meta["StudyInstanceUID"].map(
    train.set_index("StudyInstanceUID")["Report"].fillna("").str.len())

final_cov = float(meta["laterality"].isin(["L", "R"]).mean())
print(f"=== NIHAI LATERALITE KAPSAMI (study duzeyinde) ===\n")
print(f"  Katman 3 durumu : {layer3_note}")
print(f"  Bilinen taraf   : {final_cov:.1%}")
print(f"  Bilinmeyen      : {1-final_cov:.1%}")
print(f"  Celiskili       : {int((meta['laterality'] == 'CELISKI').sum())}")
print("\n  Taraf dagilimi:")
print(meta["laterality"].value_counts(dropna=False).to_string())

print("\n  Bilinmeyen taraf ne anlama geliyor: o study aynalanmaz, model 4 medial/lateral")
print("  etiketinde hangi diz oldugunu goruntudan cikarmak zorunda kalir. Bu bir")
print("  ablation degiskeni: `laterality` sutununu modele feature olarak vermek de")
print("  bir secenek (Faz 3.3 metadata enjeksiyonu).")

os.makedirs(OUT_DIR, exist_ok=True)
meta.to_csv(f"{OUT_DIR}/study_metadata.csv", index=False)
dist.to_csv(f"{OUT_DIR}/label_distribution.csv", index=False)
keys.to_csv(f"{OUT_DIR}/group_key_candidates.csv", index=False)
lang_tab.to_csv(f"{OUT_DIR}/language_distribution.csv")
if len(ev):
    sweep.to_csv(f"{OUT_DIR}/laterality_layer3_sweep.csv", index=False)
ser.drop(columns=["ok"], errors="ignore").to_csv(f"{OUT_DIR}/series_metadata.csv", index=False)

for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".csv"):
        print(f"  kaydedildi: {f}  ({os.path.getsize(OUT_DIR+'/'+f)/1024:.1f} KB)")

# %% [markdown]
# ## 12. Özet — bunu kopyalayıp Claude Code'a yapıştırın

# %%
L = []
a = L.append
a("=" * 70)
a("FAZ 1A + LATERALITE SONUC OZETI")
a("=" * 70)
a(f"Taranan: {meta['StudyInstanceUID'].nunique():,} study / {len(ser):,} seri")
a("")
a("--- ETIKET ---")
a(f"  toplam {len(train):,} | gold {int(is_gold.sum())} ({is_gold.mean():.2%}) | "
  f"etiketsiz {int((~has_any).sum()):,} | kismi {int(partial.sum())}")
a(f"  AUC olculemez etiket (<5 ornek): {int(n_bad)}/{len(LABELS)}  zayif(<15): {int(n_weak)}")
a("  en nadir 3:")
for _, r in dist.head(3).iterrows():
    a(f"    {r['etiket']:<18} poz={r['n_poz']:<4} prevalans={r['prevalans']}")
a("")
a(f"--- DIL ({lang_method}) ---")
for k, v in lang_vc.head(15).items():
    a(f"    {str(k):<10} {v:>6,}  ({v/len(train):5.1%})")
a("")
a("--- SERI ---")
a(f"  study basina seri: medyan {spc.median():.0f} (min {spc.min()}, max {spc.max()})")
a(f"  uc duzlemin hepsi olan study: {all3:,}/{len(plane_cov):,} ({all3/len(plane_cov):.1%})")
a("")
a("--- GRUP ANAHTARI ---")
for _, r in keys.iterrows():
    a(f"  {r['anahtar']:<30} n_grup={r['n_grup']:<5} en_buyuk={r['en_buyuk_pay']:<6} "
      f"tek_studyli={r['tek_studyli']:<4} gold_grup={r['gold_bulunan_grup']}")
a("")
a("--- LATERALITE ---")
a(f"  tag      : seri {ser_cov_tag:.1%} -> study {stu_cov_tag:.1%}")
a(f"  katman1+2: seri {ser_cov_12:.1%} -> study {stu_cov_12:.1%}")
if len(ev):
    a(f"  katman3  : eski isabet 0.861 -> yeni en iyi {best_acc:.3f}  ({layer3_note})")
a(f"  NIHAI study kapsami: {final_cov:.1%}")
a("=" * 70)

report = "\n".join(L)
print(report)
with open(f"{OUT_DIR}/phase1a_summary.txt", "w", encoding="utf-8") as fh:
    fh.write(report)
