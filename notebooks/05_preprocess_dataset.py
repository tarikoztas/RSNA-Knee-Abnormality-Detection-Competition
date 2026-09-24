# %% [markdown]
# # Faz 2.1–2.3 — Ön İşlenmiş Dataset
#
# 570 GB ham DICOM'u her oturumda yeniden decode etmek israf. Bir kere işleyip
# küçültülmüş bir dataset üretiyoruz: **DICOM → sırala → normalize → aynala →
# 256px → 16 slice → uint8 → `.npz`**
#
# ## İki modda çalışır
#
# | Mod | Ne yapar | Süre |
# |---|---|---|
# | `MODE = "probe"` | 24 study işler, **gerçek boyut ve süreyi ölçüp tüm veriye ölçekler** | ~3-5 dk |
# | `MODE = "full"` | 4.407 study'nin tamamı | probe'un söylediği kadar |
#
# **Önce `probe`.** Boyut tahminini kâğıt üzerinde yaptım (256px/16 slice/3 seri
# → ~13.9 GB) ama sıkıştırma oranını bilmiyorum. Kaggle notebook çıktısı ~20 GB
# ile sınırlı; tahminle 8 saatlik koşuya girip sınırı aşmak istemiyoruz.
#
# ## Tasarım kararları
#
# **Seri seçimi — her düzlemden bir tane.** Study başına ortalama 5.5 seri var;
# hepsini işlemek pahalı, rastgele seçmek bilgi kaybı. Faz 1A'da üç düzlemin de
# **4.407 study'nin 4.407'sinde** mevcut olduğu ölçüldü, yani maskeleme gerekmiyor.
# Sıralama ölçütü: fluid-sensitive → fat-suppressed → slice sayısı. Gerekçesi
# `ARCHITECTURE.md` Bölüm 1.3: efüzyon/sinovit/kontüzyon/Baker sıvı-duyarlı
# sekanslarda görünür, menisküs ve ACL sagittal'de, MCL coronal'de, PF OA axial'de.
#
# **Lateralite study düzeyinde.** Faz 1A'da 25 study'de `Laterality` tag'i seriler
# arasında çelişkili çıktı ve 25'inin 25'inde merkez-x tek taraf gösterdi — tag
# yanlış, iki diz yok. Seri başına karar vermek study'yi kendi içinde tutarsız
# hale getirir: aynı study'de aynalanmış ve aynalanmamış seriler olur. Çoğunluk
# oyu bunu çözüyor.
#
# **uint8, float16 değil.** Normalizasyon zaten [0,1]'e getiriyor; 256 seviyeye
# yuvarlamak MR için pratikte kayıpsız ve diskte iki kat yer kazandırıyor.
#
# **16 slice — bilinçli bir taviz.** Medyan seri 30 slice. Eşit aralıklı 16'ya
# indirmek komşu-slice sürekliliğini seyreltiyor, ki 2.5D tam onu kullanıyor.
# Karşılığında boyut yarıya iniyor. Faz 3.6 ablation'ında geri açılabilir.
#
# ## Runtime
#
# | Ayar | Değer |
# |---|---|
# | Accelerator | **None (CPU)** — darboğaz ağ I/O, GPU boşta bekler |
# | Internet | Açık (gerekmez ama zararsız) |
# | Persistence | No persistence |

# %% [markdown]
# ## 0. Yapılandırma

# %%
MODE = "full"             # "probe" | "full"

IMG_SIZE = 256            # ARCHITECTURE baslangic degeri
N_SLICES = 16             # seri basina saklanacak slice (esit arali)
TARGET_SIDE = "R"         # sol dizler saga aynalanir; None = aynalama yok
N_PROBE = 24              # probe modunda kac study

MAX_WORKERS = 8           # study duzeyinde paralellik (I/O bound)
DICOM_WORKERS = 4         # seri icinde dosya okuma paralelligi
OUT_DIR = "/kaggle/working"
NPZ_DIR = f"{OUT_DIR}/npz"

# Kaggle notebook cikti siniri ~20 GB. Probe bunu asacagini soylerse
# full moda gecmeden N_SLICES veya IMG_SIZE dusurulecek.
SIZE_LIMIT_GB = 18.0

# %% [markdown]
# ## 1. Modülü diske yaz

# %%
#!writefile src/data/dicom_io.py /kaggle/working/dicom_io.py

# %%
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, "/kaggle/working")
import dicom_io as dio

pd.set_option("display.width", 200)
os.makedirs(NPZ_DIR, exist_ok=True)

DATA = dio.find_data_root()
print("VERI:", DATA)
print(f"MODE={MODE}  {IMG_SIZE}px  {N_SLICES} slice  hedef taraf={TARGET_SIDE}")

# %% [markdown]
# ## 2. Study listesi ve seri metadata'sı

# %%
train = pd.read_csv(f"{DATA}/train.csv")
ts = pd.read_csv(f"{DATA}/train_series.csv")
print(f"study {train.StudyInstanceUID.nunique():,}  seri {len(ts):,}")

# Duzlem bilgisi CSV'de var (Faz 0.5'te hesapladigimizla %100 ortusuyor),
# o yuzden DICOM basligi okumadan seri secebiliyoruz — bu buyuk bir hiz kazanci.
need = ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane",
        "Fluid_Sensitive", "Fat_Suppression"]
missing = [c for c in need if c not in ts.columns]
if missing:
    raise KeyError(f"train_series.csv'de eksik sutun: {missing}")

# n_files bilinmiyorsa 0 kabul edilir; siralamada son olcut oldugu icin zararsiz.
if "n_files" not in ts.columns:
    ts["n_files"] = 0

by_study = {k: v.to_dict("records") for k, v in ts.groupby("StudyInstanceUID")}
all_uids = sorted(by_study)

if MODE == "probe":
    rng = np.random.default_rng(42)
    uids = list(rng.choice(all_uids, size=min(N_PROBE, len(all_uids)), replace=False))
elif MODE == "full":
    uids = all_uids
else:
    raise ValueError(f'bilinmeyen MODE: {MODE!r}. Gecerli: "probe" | "full"')
print(f"islenecek study: {len(uids):,}")

# %% [markdown]
# ## 3. İşleme
#
# Her study bir `.npz` dosyasına yazılıyor: üç düzlem + metadata. Dosya varsa
# atlanıyor, yani oturum kesilirse aynı notebook kaldığı yerden devam ediyor.

# %%
def process(uid):
    dst = f"{NPZ_DIR}/{uid}.npz"
    if os.path.exists(dst):
        return {"StudyInstanceUID": uid, "durum": "atlandi",
                "bytes": os.path.getsize(dst)}
    t0 = time.time()
    try:
        data, info = dio.load_study(
            DATA, uid, by_study[uid], size=IMG_SIZE, n_slices=N_SLICES,
            target_side=TARGET_SIDE, workers=DICOM_WORKERS)
    except Exception as e:
        return {"StudyInstanceUID": uid, "durum": f"HATA: {type(e).__name__}",
                "hata": str(e)[:200], "sec": round(time.time() - t0, 2)}
    if not data:
        return {"StudyInstanceUID": uid, "durum": "veri yok",
                "sec": round(time.time() - t0, 2)}

    # Sikistirilmis kaydet: MR uint8 iyi sikisir, oran probe'da olculuyor.
    np.savez_compressed(dst, **data,
                        meta=json.dumps({"laterality": info["laterality"],
                                         "planes": info["planes"]},
                                        ensure_ascii=False))
    row = {"StudyInstanceUID": uid, "durum": "ok",
           "bytes": os.path.getsize(dst),
           "sec": round(time.time() - t0, 2),
           "laterality": info["laterality"],
           "n_duzlem": len(data),
           "n_hata": info.get("n_errors", 0)}
    for pl, d in info["planes"].items():
        row[f"{pl[:3]}_slice"] = d["n_slices_kept"]
        row[f"{pl[:3]}_native"] = d["n_slices_native"]
        row[f"{pl[:3]}_mirror"] = d["mirrored"]
    return row


t0 = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    rows = []
    for i, r in enumerate(ex.map(process, uids), 1):
        rows.append(r)
        if i % max(1, len(uids) // 20) == 0 or i == len(uids):
            el = time.time() - t0
            print(f"  {i}/{len(uids)}  {el:.0f}s  "
                  f"({el/i:.2f} s/study, tahmini kalan {(len(uids)-i)*el/i/60:.1f} dk)")
elapsed = time.time() - t0
res = pd.DataFrame(rows)

# %% [markdown]
# ## 4. Sonuç ve — probe ise — tüm veriye ölçekleme

# %%
ok = res[res.durum == "ok"]
print("=" * 72)
print("SONUC")
print("=" * 72)
print(res.durum.value_counts().to_string())
print()
if len(ok):
    mb = ok.bytes.mean() / 1e6
    sp = ok.sec.mean()
    print(f"  basarili        : {len(ok)}/{len(res)}")
    print(f"  study basina    : {mb:.2f} MB   {sp:.2f} s")
    print(f"  duzlem sayisi   : {dict(ok.n_duzlem.value_counts())}")
    print(f"  taraf           : {dict(ok.laterality.value_counts(dropna=False))}")
    print(f"  aynalanan study : "
          f"{int(ok.filter(like='_mirror').any(axis=1).sum())}/{len(ok)}")
    print(f"  gecen sure      : {elapsed/60:.1f} dk ({len(uids)} study, "
          f"{MAX_WORKERS} worker)")

    N_ALL = len(all_uids)
    proj_gb = mb * N_ALL / 1000
    proj_h = sp * N_ALL / MAX_WORKERS / 3600
    print()
    print("--- TUM VERIYE OLCEKLEME ---")
    print(f"  {N_ALL:,} study -> tahmini {proj_gb:.1f} GB, {proj_h:.2f} saat")
    print(f"  cikti siniri {SIZE_LIMIT_GB} GB")
    if proj_gb > SIZE_LIMIT_GB:
        oran = SIZE_LIMIT_GB / proj_gb
        print()
        print("  " + "!" * 66)
        print(f"  !! SINIR ASILIYOR ({proj_gb:.1f} > {SIZE_LIMIT_GB} GB).")
        print(f"  !! Kucultme gerekli. Secenekler (hedef {oran:.0%} boyut):")
        print(f"  !!   N_SLICES {N_SLICES} -> {max(int(N_SLICES*oran), 8)}"
              f"   (boyut dogrusal azalir)")
        print(f"  !!   IMG_SIZE {IMG_SIZE} -> {int(IMG_SIZE*oran**0.5)}"
              f"   (boyut karesiyle azalir)")
        print("  !! Cozunurluk mu slice sayisi mi? Menisküs/kartilaj icin")
        print("  !! duzlem-ici cozunurluk daha kritik -> once N_SLICES dusur.")
        print("  " + "!" * 66)
    else:
        print(f"  >> SIGIYOR. full moda gecilebilir (~{proj_h:.1f} saat).")

if (res.durum != "ok").any():
    print()
    print("--- BASARISIZ ORNEKLERI ---")
    print(res[res.durum != "ok"].head(8).to_string(index=False))

res.to_csv(f"{OUT_DIR}/preprocess_{MODE}_log.csv", index=False)

# %% [markdown]
# ## 5. Görsel doğrulama
#
# Sayılar doğru görünse bile görüntüler bozuk olabilir. Bir study'nin üç
# düzlemini gözle kontrol ediyoruz — bu, sessiz bir hatayı (yanlış eksen,
# ters normalizasyon, bozuk aynalama) yakalayan en hızlı kontrol.

# %%
if len(ok):
    import matplotlib.pyplot as plt

    uid = ok.iloc[0].StudyInstanceUID
    z = np.load(f"{NPZ_DIR}/{uid}.npz", allow_pickle=False)
    planes = [k for k in z.files if k != "meta"]
    meta = json.loads(str(z["meta"]))
    print(f"study {uid[:28]}...  taraf={meta['laterality']}")

    n_show = 5
    fig, axes = plt.subplots(len(planes), n_show,
                             figsize=(2.2 * n_show, 2.4 * len(planes)))
    axes = np.atleast_2d(axes)
    for r, pl in enumerate(planes):
        v = z[pl]
        idx = np.linspace(0, v.shape[0] - 1, n_show).round().astype(int)
        for c, j in enumerate(idx):
            axes[r, c].imshow(v[j], cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_axis_off()
            if c == 0:
                axes[r, c].set_title(f"{pl}  {v.shape}", fontsize=9, loc="left")
    plt.tight_layout()
    plt.show()

    print()
    print("Neye bakmalisin:")
    print("  - Diz anatomisi taninabiliyor mu (femur ustte, tibia altta)?")
    print("  - Kontrast makul mu, her sey beyaz/siyah degil mi?")
    print("  - Sagittal'de patella ONDE mi? (arkadaysa aynalama ekseni yanlis)")
    print("  - Uc duzlem gercekten farkli gorunuyor mu?")

# %% [markdown]
# ## 6. TESLİM ÖZETİ

# %%
print("=" * 72)
print("TESLIM OZETI")
print("=" * 72)
print(f"  MODE={MODE}  {IMG_SIZE}px  {N_SLICES} slice  worker={MAX_WORKERS}")
tot = sum(os.path.getsize(f"{NPZ_DIR}/{f}") for f in os.listdir(NPZ_DIR)) \
      if os.path.exists(NPZ_DIR) else 0
print(f"  uretilen npz : {len(os.listdir(NPZ_DIR)) if os.path.exists(NPZ_DIR) else 0}"
      f"  ({tot/1e6:.1f} MB)")
if len(ok):
    print(f"  study basina : {ok.bytes.mean()/1e6:.2f} MB, {ok.sec.mean():.2f} s")
    # /1e9 olmali: ok.bytes BYTE cinsinden. Ilk surumde /1000 yazmisim ve
    # ozet "~9441592.6 GB" basmisti — Bolum 4'teki dogru hesap 9.4 GB diyordu.
    print(f"  TUM VERI     : ~{ok.bytes.mean()*len(all_uids)/1e9:.1f} GB, "
          f"~{ok.sec.mean()*len(all_uids)/MAX_WORKERS/3600:.2f} saat")
print(f"  durum        : {dict(res.durum.value_counts())}")
print("=" * 72)
