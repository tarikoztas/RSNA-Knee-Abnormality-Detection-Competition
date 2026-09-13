# %% [markdown]
# # Faz 0.5 – 0.9 — Ön İşleme Katmanının Doğrulanması
#
# Faz 0.3/0.4 bize *hangi metadata'nın var olduğunu* söyledi. Bu notebook bir adım
# ötesini yapıyor: **yazdığımız fonksiyonlar gerçek veride doğru çalışıyor mu?**
#
# | Görev | Burada nasıl doğrulanıyor |
# |---|---|
# | 0.5 slice sıralama | `ImagePositionPatient` sırası ile `InstanceNumber` sırası kaç seride ayrışıyor? Slice aralığı düzgün mü? |
# | 0.6 normalizasyon | Histogramlar, seri-öncesi/sonrası istatistikler |
# | 0.7 laterality | **Tag'i dolu olan %48.8'lik alt kümede heuristic'lerin doğruluğunu ölç** |
# | 0.8 görselleştirme | Düzlem × sekans ızgarası, gözle sağlama |
# | 0.9 2.5D | Pencere görselleştirmesi + kenar davranışı |
# | bonus | Seri başına işleme süresi → Faz 2.2'nin maliyet tahmini |
#
# **Önemli tasarım notu:** Aşağıdaki hücre `src/data/dicom_io.py`'yi diske yazar.
# Modülün tek doğru sürümü repo'dadır; bu notebook onu *kopyalamaz*, `py2ipynb.py`
# dönüştürme sırasında gömer. Böylece iki farklı sürümü debug etme derdi olmaz.
#
# **Çalışma süresi:** ~10-15 dakika. CPU yeterli.

# %% [markdown]
# ## 0. Yapılandırma

# %%
SEED = 42

# Metadata denetimi (hizli, sadece basliklar okunur) kac seri uzerinde yapilsin?
N_AUDIT_SERIES = 300

# Piksel okunacak (yavas) kac seri? Gorsellestirme ve zamanlama icin.
N_PIXEL_SERIES = 12

# On isleme hedef cozunurlugu (Faz 2.2 ile ayni olmali)
IMG_SIZE = 256

# 2.5D pencere yaricapi: k=2 -> 5 kanal
K_NEIGHBORS = 2

MAX_WORKERS = 16
OUT_DIR = "/kaggle/working"

# %% [markdown]
# ## 1. Modülü diske yaz
#
# `%%writefile` hücrenin içeriğini dosyaya kaydeder ve **çalıştırmaz**. Sonraki
# hücrede normal bir Python modülü olarak import edeceğiz.

# %%
#!writefile src/data/dicom_io.py /kaggle/working/dicom_io.py

# %%
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/kaggle/working")
import dicom_io as dio

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# --- Hata yakalayici: her traceback'i dosyaya da yaz ---
# Sebep: Kaggle ciktisindaki uzun traceback'i kopyalamak zahmetli. Bu kanca
# hatayi /kaggle/working/last_error.txt'ye yazar; Output sekmesinden indirip
# oldugu gibi paylasabilirsin. Notebook'un calismasini hicbir sekilde etkilemez.
try:
    import re as _re
    from IPython import get_ipython as _gi

    _ip = _gi()
    if _ip is not None and not getattr(_ip, "_err_hook_installed", False):
        _orig_tb = _ip._showtraceback

        def _tb_to_file(etype, evalue, stb):
            try:
                os.makedirs(OUT_DIR, exist_ok=True)
                clean = _re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(stb))
                with open(OUT_DIR + "/last_error.txt", "w", encoding="utf-8") as fh:
                    fh.write(clean)
            except Exception:
                pass
            return _orig_tb(etype, evalue, stb)

        _ip._showtraceback = _tb_to_file
        _ip._err_hook_installed = True
        print("Hata yakalayici kurulu -> " + OUT_DIR + "/last_error.txt")
except Exception as _e:
    print("Hata yakalayici kurulamadi (onemsiz):", _e)

DATA = dio.find_data_root()
print("VERI YOLU:", DATA)

train = pd.read_csv(DATA + "/train.csv")
train_series = pd.read_csv(DATA + "/train_series.csv")
print(f"train: {train.shape}   train_series: {train_series.shape}")
print("seri sutunlari:", list(train_series.columns))

# %% [markdown]
# ## 2. Örneklem
#
# Düzleme göre dengeli örnekliyoruz. Sebep: sagittal seriler veride baskın, ama
# lateralite aynalaması **düzleme göre farklı eksende** çalışıyor — coronal ve axial
# serileri az örneklersek o kod yolunu hiç test etmemiş oluruz.

# %%
rng = np.random.default_rng(SEED)

plane_col = "Anatomical_Plane"
groups = []
for plane, grp in train_series.groupby(plane_col, dropna=False):
    take = min(len(grp), N_AUDIT_SERIES // max(train_series[plane_col].nunique(), 1))
    groups.append(grp.sample(n=take, random_state=SEED))
audit = pd.concat(groups).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print("Denetlenecek seri sayisi:", len(audit))
print(audit[plane_col].value_counts(dropna=False).to_string())

# %% [markdown]
# ## 3. FAZ 0.5 — Slice sıralama denetimi
#
# Üç soruyu sayısal olarak yanıtlıyoruz:
#
# 1. **Sıralama yöntemi hangi oranda birincil yola düştü?** (`ImagePositionPatient`)
# 2. **`InstanceNumber` sırası ile fiziksel sıra kaç seride ayrışıyor?**
#    Ayrışma yoksa fallback'e güvenebiliriz; varsa `InstanceNumber` kullanan her kod
#    sessizce yanlış hacim üretiyor demektir.
# 3. **Slice aralığı düzgün mü?** (`spacing_cv` yüksekse seride boşluk/çakışma var)
#
# Ayrıca hesapladığımız düzlemi `train_series.csv`'nin `Anatomical_Plane` sütunuyla
# karşılaştırıyoruz — bu, geometri kodumuzun bağımsız bir çapraz kontrolü.

# %%
def audit_series(row):
    """Sadece BASLIKLARI okuyup siralama + lateralite metadata'sini denetle."""
    sdir = dio.series_dir(DATA, row["StudyInstanceUID"], row["SeriesInstanceUID"])
    paths = dio.list_dicom_files(sdir)
    out = {
        "StudyInstanceUID": row["StudyInstanceUID"],
        "SeriesInstanceUID": row["SeriesInstanceUID"],
        "plane_csv": row.get(plane_col),
        "fluid": row.get("Fluid_Sensitive"),
        "fatsat": row.get("Fat_Suppression"),
        "n_files": len(paths),
        "ok": False,
    }
    if not paths:
        out["error"] = "dosya yok"
        return out

    dss, errs = dio.read_datasets(paths, workers=4, headers_only=True)
    if not dss:
        out["error"] = errs[0] if errs else "okunamadi"
        return out

    ordered, info = dio.sort_datasets(dss)
    iop = ordered[0].get("ImageOrientationPatient", None)

    # InstanceNumber sirasi fiziksel sirayla ayni mi?
    inst = [d.get("InstanceNumber", None) for d in ordered]
    if all(i is not None for i in inst):
        iv = np.array([float(i) for i in inst])
        # Artan VEYA azalan olmasi yeterli: ters yon bir "ayrisma" degil,
        # sadece numaralandirma yonudur.
        out["instnum_agrees"] = bool(np.all(np.diff(iv) > 0) or np.all(np.diff(iv) < 0))
    else:
        out["instnum_agrees"] = None

    side_tag, _ = None, None
    v = None
    for d in ordered:
        v = d.get("Laterality", None)
        if v:
            break
    side_tag = str(v).strip().upper()[:1] if v else None

    desc = str(ordered[0].get("SeriesDescription", "") or "")
    proto = str(ordered[0].get("ProtocolName", "") or "")
    xs = [float(d.get("ImagePositionPatient")[0]) for d in ordered
          if d.get("ImagePositionPatient", None) is not None]

    out.update({
        "ok": True,
        "n_slices": info["n"],
        "sort_method": info["method"],
        "dup_positions": info["duplicate_positions"],
        "spacing_median": info["spacing_median"],
        "spacing_cv": info["spacing_cv"],
        "plane_calc": dio.plane_from_iop(iop),
        "mirror_axis": dio.lr_flip_axis(iop),
        "lat_tag": side_tag,
        "lat_desc": dio._text_laterality(desc) or dio._text_laterality(proto),
        "desc": desc[:60],
        "ipp_x_mean": float(np.mean(xs)) if xs else np.nan,
        "n_errors": len(errs),
    })
    return out


t0 = time.time()
rows = list(audit.to_dict("records"))
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    AUD = list(ex.map(audit_series, rows))
aud = pd.DataFrame(AUD)
print(f"Denetim suresi: {time.time()-t0:.1f}s   basarili: {aud['ok'].sum()}/{len(aud)}")

ok = aud[aud["ok"]].copy()

print("\n=== 0.5a — SIRALAMA YONTEMI ===")
print(ok["sort_method"].value_counts().to_string())

print("\n=== 0.5b — InstanceNumber fiziksel sirayla UYUMLU MU? ===")
ia = ok["instnum_agrees"]
print(f"  uyumlu    : {int((ia == True).sum())}")
print(f"  AYRISIYOR : {int((ia == False).sum())}")
print(f"  bilinmiyor: {int(ia.isna().sum())}")
if (ia == False).any():
    print("\n  -> Ayrisan seriler var. InstanceNumber'a guvenen her kod bu serilerde")
    print("     slice'lari YANLIS sirada dizer. IPP birincil yontem olarak kalmali.")
    print(ok[ia == False][["desc", "n_slices", "sort_method"]].head(8).to_string(index=False))

print("\n=== 0.5c — SLICE ARALIGI DUZENLILIGI ===")
print(ok["spacing_cv"].describe().to_string())
irregular = ok[ok["spacing_cv"] > 0.10]
print(f"\n  Duzensiz aralikli seri (cv > 0.10): {len(irregular)} / {len(ok)}")
print(f"  Ayni pozisyonda tekrar eden slice iceren seri: {int((ok['dup_positions'] > 0).sum())}")
if (ok["dup_positions"] > 0).any():
    print("  (multi-echo olabilir — Faz 2.1 seri seciminde dikkate alinacak)")

print("\n=== 0.5d — HESAPLANAN DUZLEM vs CSV ===")
xt = pd.crosstab(ok["plane_calc"], ok["plane_csv"], dropna=False)
print(xt.to_string())
diag = sum(xt.loc[p, p] for p in xt.index if p in xt.columns)
print(f"\n  Uyum: {diag}/{len(ok)} ({diag/max(len(ok),1):.1%})")
print("  (Dusukse geometri kodunda ya da CSV etiketinde sorun var — ikisi bagimsiz kaynak.)")

print("\n=== 0.5e — DUZLEM BASINA AYNALAMA EKSENI ===")
print(pd.crosstab(ok["plane_calc"], ok["mirror_axis"]).to_string())
print("\n  Beklenen: Sagittal -> 0 (slice sirasi),  Coronal/Axial -> 2 (yatay)")

# %% [markdown]
# ## 4. FAZ 0.7 — Lateralite heuristic'lerinin doğruluğu
#
# Bu bölüm bu notebook'un en değerli kısmı. `Laterality` tag'i serilerin yarısında var;
# **o yarıyı cevap anahtarı olarak kullanıp** diğer katmanların ne kadar isabetli
# olduğunu ölçüyoruz.
#
# Analoji: sözlüğün yarısını kapatıp kendi çevirinle karşılaştırmak. Kalan yarıda
# ne kadar güvenebileceğini ancak böyle bilirsin.
#
# Yanlış aynalama, eksik aynalamadan **daha zararlıdır**: eksik aynalama sadece
# varyansı artırır, yanlış aynalama modele sistematik olarak ters anatomi öğretir.
# O yüzden isabet oranı yüksek değilse katmanı kapatırız.

# %%
print("=== 0.7a — TAG KAPSAMI ===")
cov = ok["lat_tag"].notna().mean()
print(f"  Laterality tag'i dolu: {cov:.1%} ({ok['lat_tag'].notna().sum()}/{len(ok)})")
print(ok["lat_tag"].value_counts(dropna=False).to_string())

labeled = ok[ok["lat_tag"].isin(["L", "R"])].copy()

print("\n=== 0.7b — SeriesDescription/ProtocolName HEURISTIC'I ===")
if len(labeled):
    has_pred = labeled["lat_desc"].notna()
    print(f"  Karar verebildigi seri : {has_pred.sum()}/{len(labeled)} ({has_pred.mean():.1%})")
    if has_pred.any():
        sub = labeled[has_pred]
        acc = (sub["lat_desc"] == sub["lat_tag"]).mean()
        print(f"  Karar verdiginde isabet: {acc:.1%}")
        print("\n  Karsilastirma tablosu (satir=heuristic, sutun=tag):")
        print(pd.crosstab(sub["lat_desc"], sub["lat_tag"]).to_string())
        wrong = sub[sub["lat_desc"] != sub["lat_tag"]]
        if len(wrong):
            print(f"\n  YANLIS ORNEKLER ({len(wrong)}):")
            print(wrong[["desc", "lat_desc", "lat_tag"]].head(10).to_string(index=False))
    else:
        print("  Hic karar veremedi -> SeriesDescription'da taraf bilgisi yok.")

print("\n=== 0.7c — ImagePositionPatient x-ISARETI HEURISTIC'I ===")
if len(labeled):
    valid = labeled[labeled["ipp_x_mean"].abs() > 1.0]
    if len(valid):
        pred_x = np.where(valid["ipp_x_mean"] > 0, "L", "R")
        acc_x = float((pred_x == valid["lat_tag"].values).mean())
        print(f"  Karar verebildigi seri : {len(valid)}/{len(labeled)} ({len(valid)/len(labeled):.1%})")
        print(f"  Karar verdiginde isabet: {acc_x:.1%}")
        print("\n  x ortalamasinin tag'e gore dagilimi:")
        print(valid.groupby("lat_tag")["ipp_x_mean"].describe()[["count", "mean", "50%", "min", "max"]].to_string())
        print("\n  KARAR KURALI:")
        if acc_x >= 0.90:
            print("    >= %90  -> katman guvenli, acik kalsin.")
        elif acc_x >= 0.70:
            print("    %70-90  -> supheli. Yanlis aynalama eksik aynalamadan kotudur;")
            print("               bu katmani KAPATMAK muhtemelen daha dogru.")
        else:
            print("    < %70   -> KAPAT. Yazi-tura atmaktan iyi degil.")
            print("               (Beklenen sonuc: izomerkez dizin uzerine ortalandigi icin")
            print("                x isareti taraf hakkinda bilgi tasimayabilir.)")
    else:
        print("  Hicbir seride |x| > 1mm degil -> katman zaten devre disi kalir.")

# %% [markdown]
# ## 5. Piksel okuma — örnek seriler
#
# Buradan sonrası gerçek piksel gerektiriyor, o yüzden az sayıda seri.

# %%
pool = ok[ok["n_slices"] >= 2 * K_NEIGHBORS + 1]
# groupby.apply(sample) yerine karistir-sonra-head: pandas 2.2+ apply icinde
# gruplama sutununu dusuruyor, groupby.sample ise grup kucukse hata veriyor.
per_plane = max(N_PIXEL_SERIES // max(pool["plane_calc"].nunique(), 1), 1)
pick = (pool.sample(frac=1.0, random_state=SEED)
            .groupby("plane_calc", sort=False)
            .head(per_plane)
            .head(N_PIXEL_SERIES))
assert len(pick), "piksel okunacak uygun seri bulunamadi"
print("Piksel okunacak seriler:")
print(pick[["plane_calc", "fluid", "fatsat", "n_slices", "desc"]].to_string(index=False))

loaded, timings = [], []
for _, r in pick.iterrows():
    sdir = dio.series_dir(DATA, r["StudyInstanceUID"], r["SeriesInstanceUID"])
    t0 = time.time()
    sv = dio.load_series(sdir, size=IMG_SIZE, workers=MAX_WORKERS)
    dt = time.time() - t0
    timings.append({"n_slices": sv.n_slices, "sec": dt, "plane": sv.plane})
    loaded.append((r, sv))
    print(f"  {sv.plane:<9} slices={sv.n_slices:<3} lat={str(sv.laterality):<4} "
          f"({sv.laterality_source:<22}) mirror={str(sv.mirrored):<5} "
          f"axis={sv.mirror_axis} sort={sv.sort_method:<22} {dt:5.2f}s")

tim = pd.DataFrame(timings)

# %% [markdown]
# ## 6. FAZ 0.6 — Normalizasyonun görsel ve sayısal kontrolü
#
# Sol: ham piksel değerleri (her seri kendi ölçeğinde — MR'da mutlak birim yok).
# Sağ: normalizasyon sonrası. Hepsi [0,1]'e oturmalı ve **dağılım şekli korunmalı**.
# Dağılım tek bir dikey çizgiye çökerse normalizasyon kontrastı öldürmüş demektir.

# %%
# Not: bu hucre tek bir bozuk seri yuzunden komple durmasin diye her seriyi ayri
# try/except icinde isliyor. Bir teshis hucresinin kendisi kirilgan olmamali —
# aradigimiz sorunu bulmadan cokerse hicbir ise yaramaz.
def _safe_hist(ax, arr, label):
    """Bos veya tek-degerli diziyi hist'e vermeden ele — matplotlib bunlarda patlar."""
    a = np.asarray(arr).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "bos"
    if float(a.min()) == float(a.max()):
        return f"sabit deger ({a.min():.3g})"
    ax.hist(a, bins=100, histtype="step", label=label)
    return None


fig, axes = plt.subplots(1, 2, figsize=(14, 4))
drawn, skipped = 0, []
for r, sv in loaded[:6]:
    try:
        sdir = dio.series_dir(DATA, r["StudyInstanceUID"], r["SeriesInstanceUID"])
        dss, errs = dio.read_datasets(dio.list_dicom_files(sdir)[:5], workers=8)
        if not dss:
            skipped.append((sv.plane, "dosya okunamadi: " + (errs[0] if errs else "?")))
            continue
        raw, _ = dio.extract_pixels(dss)
        p1 = _safe_hist(axes[0], raw, sv.plane)
        p2 = _safe_hist(axes[1], sv.vol, sv.plane)
        if p1 or p2:
            skipped.append((sv.plane, f"ham={p1} norm={p2}"))
        else:
            drawn += 1
    except Exception as e:
        skipped.append((sv.plane, f"{type(e).__name__}: {e}"))

axes[0].set_title("HAM piksel degerleri (seriler ayri olceklerde)")
axes[1].set_title("Normalize sonrasi [0,1]")
for a in axes:
    a.set_yscale("log")
    if a.get_legend_handles_labels()[0]:
        a.legend(fontsize=7)
plt.tight_layout()
plt.show()

print(f"Histograma cizilen seri: {drawn}/{len(loaded[:6])}")
if skipped:
    print("Atlananlar:")
    for pl, why in skipped:
        print(f"  {pl:<10} {why}")

# %%
rows = []
for _, sv in loaded:
    v = sv.vol
    if v.size == 0:
        rows.append({"plane": sv.plane, "slices": sv.n_slices, "BOS": True})
        continue
    rows.append({
        "plane": sv.plane, "slices": sv.n_slices, "BOS": False,
        "shape": str(v.shape),
        "min": float(v.min()), "max": float(v.max()),
        "mean": float(v.mean()), "std": float(v.std()),
        "n_nan": int(np.isnan(v).sum()),
        "spacing": sv.pixel_spacing,
    })
stats = pd.DataFrame(rows)
print(stats.to_string(index=False))

# --- Saglik kontrolleri: assert yerine RAPOR. ---
# assert bir teshis notebook'unda yanlis arac: ilk sorunda durur ve geri kalan
# bilgiyi gormeni engeller. Burada istedigimiz sey tam tersi — butun resmi gormek.
print("\n=== SAGLIK KONTROLLERI ===")
problems = []
if stats.get("BOS", pd.Series(dtype=bool)).any():
    problems.append(f"{int(stats['BOS'].sum())} seri BOS hacim dondurdu")
if "min" in stats:
    if stats["min"].min() < 0.0:
        problems.append(f"min < 0 (en dusuk {stats['min'].min():.4f})")
    if stats["max"].max() > 1.0:
        problems.append(f"max > 1 (en yuksek {stats['max'].max():.4f})")
    if (stats["n_nan"] > 0).any():
        problems.append(f"{int(stats['n_nan'].sum())} NaN piksel")
    flat = stats[stats["std"] < 0.01]
    if len(flat):
        problems.append(f"{len(flat)} seride std < 0.01 (kontrast yok — normalizasyon coktu?)")

if problems:
    print("!! SORUN VAR:")
    for p in problems:
        print("   -", p)
else:
    print(">> Tum hacimler [0,1] araliginda, NaN yok, kontrast var.")

# %% [markdown]
# ## 7. FAZ 0.8 — Görselleştirme
#
# Buradan itibaren gözle bakıyoruz. Aranacak şeyler:
# - Diz anatomisi tanınabiliyor mu, görüntü ters/aynalı mı görünüyor?
# - Sıralama düzgün mü — kesitler yumuşak geçiyor mu, yoksa zıplıyor mu?
# - `Fluid_Sensitive=1` serilerde sıvı (eklem sıvısı, kist) **parlak** görünmeli.

# %%
def montage(sv, title, n=8):
    idx = np.linspace(0, sv.n_slices - 1, min(n, sv.n_slices)).round().astype(int)
    fig, axes = plt.subplots(1, len(idx), figsize=(2 * len(idx), 2.4))
    axes = np.atleast_1d(axes)
    for a, i in zip(axes, idx):
        a.imshow(sv.vol[i], cmap="gray", vmin=0, vmax=1)
        a.set_title(f"#{i}", fontsize=7)
        a.axis("off")
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.show()


for r, sv in loaded[:6]:
    montage(sv, f"{sv.plane} | fluid={r['fluid']} fatsat={r['fatsat']} | "
                f"lat={sv.laterality}({sv.laterality_source}) mirrored={sv.mirrored} | {r['desc']}")

# %% [markdown]
# ### 7b. Aynalamanın görsel kanıtı
#
# Burada `lr_flip_axis` düzeltmesini gözle doğruluyoruz. Aynı seriyi aynalamadan ve
# aynalayarak gösteriyoruz:
#
# - **Coronal/Axial**: görüntü yatay dönmeli (sol-sağ yer değiştirir).
# - **Sagittal**: tek tek görüntüler **neredeyse aynı kalmalı**, değişen şey
#   *slice sırası* olmalı. Sagittal'de yatay çevirme yapılsaydı patella (diz kapağı)
#   önden arkaya geçerdi — yani anatomiyi bozardı.

# %%
# Her duzlemden EN FAZLA BIR ornek — yoksa 12 seri x 1 figur ekrani doldurur.
shown_planes = set()
for r, sv in loaded:
    if sv.plane in shown_planes or sv.n_slices < 3:
        continue
    sdir = dio.series_dir(DATA, r["StudyInstanceUID"], r["SeriesInstanceUID"])
    raw = dio.load_series(sdir, size=IMG_SIZE, target_side=None, workers=MAX_WORKERS)
    iop_ds, _ = dio.read_datasets(dio.list_dicom_files(sdir)[:1], workers=1, headers_only=True)
    iop = iop_ds[0].get("ImageOrientationPatient", None) if iop_ds else None
    mir, ax = dio.apply_lr_mirror(raw.vol, iop)
    if ax is None:
        continue
    shown_planes.add(sv.plane)
    mid = raw.n_slices // 2
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    axes[0].imshow(raw.vol[mid], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"orijinal (slice {mid})", fontsize=8)
    axes[1].imshow(mir[mid], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"aynalanmis, axis={ax} (slice {mid})", fontsize=8)
    for a in axes:
        a.axis("off")
    fig.suptitle(f"{raw.plane}: axis={ax} "
                 f"({'slice sirasi ters cevrildi -> tek kesit AYNI kalmali' if ax == 0 else 'yatay cevirme -> gorunur degisim'})",
                 fontsize=9)
    plt.tight_layout()
    plt.show()
    if len(shown_planes) >= 3:
        break

# %% [markdown]
# ## 8. FAZ 0.9 — 2.5D pencere
#
# Her kanal ayrı bir komşu slice. Kanallar arasında yumuşak bir geçiş görmeliyiz —
# aynı anatomi hafifçe kayarak ilerlemeli. Sert sıçrama varsa sıralama bozuktur.
#
# Alt sıra kenar davranışını gösteriyor: `center=0`'da ilk kanallar tekrar eder
# (sıfır dolgu yerine). Sıfır dolgu yapay bir siyah kenar yaratır ve model bunu
# "seri başı" işareti olarak öğrenebilir.

# %%
assert loaded, "hic seri yuklenemedi"
r, sv = max(loaded, key=lambda t: t[1].n_slices)   # en cok slice'li seri en ogretici
centers = dio.sample_window_centers(sv.n_slices, n_windows=3, k=K_NEIGHBORS)
print(f"{sv.plane} serisi, {sv.n_slices} slice -> pencere merkezleri: {centers}")

for c in centers:
    st = dio.make_25d(sv.vol, c, k=K_NEIGHBORS)
    fig, axes = plt.subplots(1, len(st), figsize=(2 * len(st), 2.4))
    for i, a in enumerate(np.atleast_1d(axes)):
        a.imshow(st[i], cmap="gray", vmin=0, vmax=1)
        a.set_title(f"kanal {i}", fontsize=7)
        a.axis("off")
    fig.suptitle(f"2.5D pencere, merkez={c}, sekil={st.shape}", fontsize=9)
    plt.tight_layout()
    plt.show()

edge = dio.make_25d(sv.vol, 0, k=K_NEIGHBORS)
fig, axes = plt.subplots(1, len(edge), figsize=(2 * len(edge), 2.4))
for i, a in enumerate(np.atleast_1d(axes)):
    a.imshow(edge[i], cmap="gray", vmin=0, vmax=1)
    a.set_title(f"kanal {i}", fontsize=7)
    a.axis("off")
fig.suptitle("KENAR DAVRANISI: center=0 -> ilk kanallar tekrar eder", fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Maliyet tahmini — Faz 2.2 planlaması için
#
# 570 GB'ı bir kere ön işleyip küçültülmüş bir Kaggle Dataset üreteceğiz (Faz 2.3).
# O işin ne kadar süreceğini **şimdi** bilmek, o fazın tasarımını belirler:
# tek oturumda biter mi, yoksa parçalara bölüp checkpoint'lemek gerekir mi?

# %%
print(tim.to_string(index=False))
per_series = float(tim["sec"].mean())
per_slice = float((tim["sec"] / tim["n_slices"].clip(lower=1)).mean())
n_series_total = len(train_series)

print(f"\nSeri basina ortalama : {per_series:.2f} s")
print(f"Slice basina ortalama: {per_slice*1000:.1f} ms")
print(f"\ntrain_series.csv'de toplam {n_series_total:,} seri var.")
for n_sel in (1, 2, 3):
    tot = per_series * len(train) * n_sel / 3600
    print(f"  Study basina {n_sel} seri islenirse: ~{tot:.1f} saat (tek surec)")
print("\nNOT: Kaggle notebook siniri 9 saat. Yukaridaki sure tek bir surec icin;")
print("     study duzeyinde paralellestirme ve/veya isi parcalara bolup ayri")
print("     oturumlarda calistirmak Faz 2.3'un tasarim kararidir.")

# %% [markdown]
# ## 10. Çıktıları kaydet ve özetle

# %%
os.makedirs(OUT_DIR, exist_ok=True)
aud.to_csv(OUT_DIR + "/phase0_series_audit.csv", index=False)

lines = []
add = lines.append
add("=" * 70)
add("FAZ 0.5-0.9 DOGRULAMA OZETI")
add("=" * 70)
add(f"Denetlenen seri: {len(ok)} (basliklar)  /  piksel okunan: {len(loaded)}")
add("")
add("--- 0.5 SIRALAMA ---")
for m, c in ok["sort_method"].value_counts().items():
    add(f"  {m:<24} {c}")
add(f"  InstanceNumber ayrisan seri : {int((ok['instnum_agrees'] == False).sum())}")
add(f"  Duzensiz aralik (cv>0.10)   : {int((ok['spacing_cv'] > 0.10).sum())}")
add(f"  Tekrarli pozisyon iceren    : {int((ok['dup_positions'] > 0).sum())}")
add(f"  Duzlem uyumu (hesap vs CSV) : {diag}/{len(ok)}")
add("")
add("--- 0.7 LATERALITE ---")
add(f"  Tag kapsami: {cov:.1%}")
if len(labeled):
    hp = labeled["lat_desc"].notna()
    if hp.any():
        s = labeled[hp]
        add(f"  SeriesDescription: kapsam {hp.mean():.1%}, isabet "
            f"{(s['lat_desc'] == s['lat_tag']).mean():.1%}")
    else:
        add("  SeriesDescription: hic karar veremedi")
    v = labeled[labeled["ipp_x_mean"].abs() > 1.0]
    if len(v):
        px = np.where(v["ipp_x_mean"] > 0, "L", "R")
        add(f"  IPP x-isareti    : kapsam {len(v)/len(labeled):.1%}, isabet "
            f"{float((px == v['lat_tag'].values).mean()):.1%}")
add("")
add("--- MALIYET ---")
add(f"  Seri basina {per_series:.2f} s  |  slice basina {per_slice*1000:.1f} ms")
add("=" * 70)

report = "\n".join(lines)
print(report)
with open(OUT_DIR + "/phase0_pipeline_summary.txt", "w", encoding="utf-8") as fh:
    fh.write(report)
