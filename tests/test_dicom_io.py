"""dicom_io.py'nin saf-numpy mantigini test et.

Kullanim:  python tests/test_dicom_io.py

pydicom yerelde kurulu olmak zorunda degil — sahte modul ile stub'lanir.
Gercek DICOM okuma yolu burada test EDILMEZ; o Kaggle'da
notebooks/01_phase0_pipeline ile gorsel olarak dogrulanir.
"""
import sys, types, pathlib

# pydicom ve cv2 yerelde yok; modul import edilebilsin diye sahte modul koy.
sys.modules.setdefault("pydicom", types.ModuleType("pydicom"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src" / "data"))

import numpy as np
import dicom_io as dio

FAILS = []


def check(name, cond, extra=None):
    suffix = "" if extra is None else "  " + str(extra)
    print(("  OK   " if cond else "  FAIL ") + name + suffix)
    if not cond:
        FAILS.append(name)


class FakeDS:
    """pydicom Dataset'in .get() arayuzunu taklit eden minimal nesne."""
    def __init__(self, **kw):
        self._d = kw

    def get(self, key, default=None):
        return self._d.get(key, default)


AXIAL = [1, 0, 0, 0, 1, 0]
CORONAL = [1, 0, 0, 0, 0, -1]
SAGITTAL = [0, 1, 0, 0, 0, -1]

print("\n--- duzlem tespiti (plane_from_iop) ---")
check("axial", dio.plane_from_iop(AXIAL) == "Axial", dio.plane_from_iop(AXIAL))
check("coronal", dio.plane_from_iop(CORONAL) == "Coronal", dio.plane_from_iop(CORONAL))
check("sagittal", dio.plane_from_iop(SAGITTAL) == "Sagittal", dio.plane_from_iop(SAGITTAL))
check("iop yok -> Unknown", dio.plane_from_iop(None) == "Unknown")

print("\n--- aynalama ekseni (lr_flip_axis) — ASIL DUZELTME ---")
check("sagittal -> axis 0 (slice sirasi)", dio.lr_flip_axis(SAGITTAL) == 0, dio.lr_flip_axis(SAGITTAL))
check("coronal  -> axis 2 (yatay)", dio.lr_flip_axis(CORONAL) == 2, dio.lr_flip_axis(CORONAL))
check("axial    -> axis 2 (yatay)", dio.lr_flip_axis(AXIAL) == 2, dio.lr_flip_axis(AXIAL))

# Oblik sagittal (diz MR'inda ACL icin yaygin): 20 derece dondurulmus.
th = np.deg2rad(20)
OBLIQUE_SAG = [np.sin(th), np.cos(th), 0, 0, 0, -1]
check("oblik sagittal -> hala axis 0", dio.lr_flip_axis(OBLIQUE_SAG) == 0, dio.lr_flip_axis(OBLIQUE_SAG))

print("\n--- normal vektorun isareti deterministik mi (canonical_normal) ---")
n1 = dio.canonical_normal(np.array([-1.0, 0.0, 0.0]))
n2 = dio.canonical_normal(np.array([1.0, 0.0, 0.0]))
check("+n ve -n ayni sonuca gidiyor", np.allclose(n1, n2), n1)

print("\n--- slice siralama (sort_datasets) ---")
# Kasten karisik sirada, InstanceNumber ile pozisyon UYUSMUYOR.
pos = [5.0, 1.0, 3.0, 2.0, 4.0]
dss = [FakeDS(ImageOrientationPatient=AXIAL,
              ImagePositionPatient=[0.0, 0.0, p],
              InstanceNumber=i + 1) for i, p in enumerate(pos)]
ordered, info = dio.sort_datasets(dss)
got = [d.get("ImagePositionPatient")[2] for d in ordered]
check("pozisyona gore siralandi", got == sorted(pos), got)
check("yontem = ImagePositionPatient", info["method"] == "ImagePositionPatient")
check("duplicate yok", info["duplicate_positions"] == 0)
check("spacing medyani 1.0", abs(info["spacing_median"] - 1.0) < 1e-9, info["spacing_median"])

# IPP yoksa InstanceNumber'a dusmeli
dss2 = [FakeDS(InstanceNumber=n) for n in [3, 1, 2]]
ordered2, info2 = dio.sort_datasets(dss2)
check("fallback -> InstanceNumber", info2["method"] == "InstanceNumber")
check("fallback siralamasi dogru",
      [d.get("InstanceNumber") for d in ordered2] == [1, 2, 3])

# Multi-echo: ayni pozisyonda iki slice
dss3 = [FakeDS(ImageOrientationPatient=AXIAL, ImagePositionPatient=[0, 0, z], InstanceNumber=i)
        for i, z in enumerate([1.0, 1.0, 2.0, 2.0])]
_, info3 = dio.sort_datasets(dss3)
check("duplicate pozisyon yakalandi", info3["duplicate_positions"] == 2, info3["duplicate_positions"])

print("\n--- intensite normalizasyonu (normalize_volume) ---")
rng = np.random.default_rng(0)
vol = rng.normal(500, 80, size=(6, 32, 32)).astype(np.float32)
out = dio.normalize_volume(vol)
check("cikti [0,1] araliginda", out.min() >= 0.0 and out.max() <= 1.0, (out.min(), out.max()))
check("dtype float32", out.dtype == np.float32)

# Pozitif-afin donusume karsi degismezlik (RescaleSlope tartismasinin dayanagi)
out2 = dio.normalize_volume(vol * 3.7 + 120.0)
check("afin donusume karsi degismez", np.allclose(out, out2, atol=1e-5),
      float(np.abs(out - out2).max()))

# Tek bir parlak artefakt min-max'i ele gecirir mi?
vol_art = vol.copy()
vol_art[0, 0, 0] = 1e6
out_art = dio.normalize_volume(vol_art)
check("artefakt kontrastı ezmiyor (std korunuyor)",
      abs(float(out_art.std()) - float(out.std())) < 0.02,
      (round(float(out.std()), 4), round(float(out_art.std()), 4)))

flat = np.full((3, 8, 8), 7.0, dtype=np.float32)
check("duz goruntu cokmuyor (NaN yok)", np.isfinite(dio.normalize_volume(flat)).all())
check("bos hacim sorun cikarmiyor", dio.normalize_volume(np.zeros((0, 4, 4), np.float32)).size == 0)

inv = dio.normalize_volume(vol, invert=True)
check("MONOCHROME1 tersleme dogru", np.allclose(inv, 1.0 - out, atol=1e-6))

print("\n--- 2.5D (make_25d / sample_window_centers) ---")
v = np.arange(10 * 4 * 4, dtype=np.float32).reshape(10, 4, 4)
st = dio.make_25d(v, center=5, k=2)
check("stack sekli (5,H,W)", st.shape == (5, 4, 4), st.shape)
check("dogru slice'lar", [int(s[0, 0]) for s in st] == [int(v[i][0, 0]) for i in range(3, 8)])

edge = dio.make_25d(v, center=0, k=2)
check("kenarda tekrar ediyor (sifir degil)",
      [int(s[0, 0]) for s in edge] == [int(v[i][0, 0]) for i in [0, 0, 0, 1, 2]],
      [int(s[0, 0]) for s in edge])

st_stride = dio.make_25d(v, center=5, k=2, stride=2)
check("stride=2 calisiyor",
      [int(s[0, 0]) for s in st_stride] == [int(v[i][0, 0]) for i in [1, 3, 5, 7, 9]])

c = dio.sample_window_centers(30, 4, k=2)
check("merkezler kenardan uzak", all(2 <= x <= 27 for x in c), c)
check("merkezler artan sirada", c == sorted(c), c)
check("tek pencere ortada", dio.sample_window_centers(30, 1, k=2) == [14])
short = dio.sample_window_centers(3, 4, k=2)
check("kisa seride cokmuyor", len(short) == 4 and all(0 <= x < 3 for x in short), short)
check("bos seri -> bos liste", dio.sample_window_centers(0, 3) == [])

print("\n--- lateralite metin cikarimi (_text_laterality) ---")
cases = [
    ("SAG PD FS LEFT KNEE", "L"),
    ("Cor T1 RT KNEE", "R"),
    ("KNEE R", "R"),
    ("linkes Knie", "L"),                 # Almanca cekim eki
    ("rechtes Knie", "R"),
    ("Genou gauche", "L"),
    ("rodilla derecha", "R"),
    ("joelho direito", "R"),
    ("KNEE", None),
    ("left vs right comparison", None),   # ikisi birden -> karar verme
    # --- carpismaya karsi regresyon testleri ---
    ("SAGITTAL PD FS", None),             # SAG != TR "sag"
    ("SAG T2 FS", None),
    ("COR STIR", None),
    ("AX PD lateral meniscus", None),     # "lateral" icindeki l tetiklememeli
    ("3D DESS iso", None),
    ("PD FS SAG L", "L"),                 # sagittal + gercek taraf birlikte
]
for text, want in cases:
    got = dio._text_laterality(text)
    check("'" + text + "' -> " + str(want), got == want, "got=" + str(got))

print("\n--- aynalama uygulamasi (apply_lr_mirror) ---")
vol_s = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
m_sag, ax_sag = dio.apply_lr_mirror(vol_s, SAGITTAL)
check("sagittal: slice sirasi tersine dondu", np.allclose(m_sag, vol_s[::-1]) and ax_sag == 0)
m_cor, ax_cor = dio.apply_lr_mirror(vol_s, CORONAL)
check("coronal: yatay dondu", np.allclose(m_cor, vol_s[:, :, ::-1]) and ax_cor == 2)
check("iki kez aynalamak birim islem",
      np.allclose(dio.apply_lr_mirror(m_sag, SAGITTAL)[0], vol_s))

# ---------------------------------------------------------------------------
print("\n--- goruntu merkezi ve lateralite katman 3 (image_center) ---")

# Sentetik bir coronal seri: diz merkezi hasta-x = +100 mm (SOL diz).
# IOP coronal -> satir yonu = +x, yani x ekseni GORUNTU ICINDE.
# 200 sutun x 0.5 mm = 100 mm FOV -> kose, merkezden 50 mm SOLDA (x = +50).
COR_CENTER_X = 100.0
FOV = 100.0
cor_slices = [
    FakeDS(ImagePositionPatient=[COR_CENTER_X - FOV / 2, -60.0, -10.0 + 3.0 * j],
           ImageOrientationPatient=CORONAL,
           PixelSpacing=[0.5, 0.5], Rows=200, Columns=200)
    for j in range(20)
]
c = dio.image_center(cor_slices[0])
HALF = ((200 - 1) / 2) * 0.5          # 49.75 mm — piksel MERKEZLERI arasi yaricap
check("coronal: merkez = kose + (N-1)/2*spacing",
      abs(c[0] - (COR_CENTER_X - FOV / 2 + HALF)) < 1e-9, c[0])
check("coronal: kose yaniltiyordu (+50), merkez dogru (+100)",
      abs(float(cor_slices[0].get("ImagePositionPatient")[0]) - 50.0) < 1e-6)
check("coronal: seri merkez x = +99.75", abs(dio.series_center_x(cor_slices) - 99.75) < 1e-9,
      dio.series_center_x(cor_slices))

# Sentetik sagittal seri: x ekseni SLICE eksenidir, kose zaten ~merkez.
# Slice'lar x = +90..+110 arasinda -> ortalama +100.
sag_slices = [
    FakeDS(ImagePositionPatient=[89.75 + j, 0.0, 0.0],
           ImageOrientationPatient=SAGITTAL,
           PixelSpacing=[0.5, 0.5], Rows=200, Columns=200)
    for j in range(21)
]
check("sagittal: merkez x = kose x (sapma yok)",
      abs(dio.image_center(sag_slices[0])[0] - 89.75) < 1e-9,
      dio.image_center(sag_slices[0])[0])
check("sagittal: seri merkez x = +99.75", abs(dio.series_center_x(sag_slices) - 99.75) < 1e-9,
      dio.series_center_x(sag_slices))

# ASIL TEST: iki duzlem ayni fiziksel tarafi ayni isaretle raporlamali.
check("DUZLEM SAPMASI KALKTI: coronal ve sagittal ayni x veriyor",
      abs(dio.series_center_x(cor_slices) - dio.series_center_x(sag_slices)) < 1e-6)

# Katman 3 artik varsayilan ACIK (Faz 1A: isabet 0.861 -> 0.987).
# Kapatmak hala mumkun olmali — ablation icin gerekiyor.
side_off, src_off = dio.detect_laterality(cor_slices, use_ipp_layer=False)
check("katman 3 KAPATILABILIR", side_off is None and src_off == "bulunamadi",
      (side_off, src_off))
side_def, src_def = dio.detect_laterality(cor_slices)
check("varsayilan cagri katman 3'u kullaniyor",
      side_def == "L" and src_def == "image_center_x", (side_def, src_def))
side_on, src_on = dio.detect_laterality(cor_slices, use_ipp_layer=True)
check("katman 3 acikken +x -> L", side_on == "L" and src_on == "image_center_x",
      (side_on, src_on))

# Sag diz: merkez x = -100
cor_right = [
    FakeDS(ImagePositionPatient=[-COR_CENTER_X - FOV / 2, -60.0, -10.0 + 3.0 * j],
           ImageOrientationPatient=CORONAL,
           PixelSpacing=[0.5, 0.5], Rows=200, Columns=200)
    for j in range(20)
]
check("katman 3: -x -> R", dio.detect_laterality(cor_right, use_ipp_layer=True)[0] == "R")

# Olu bolge: merkez sifira cok yakinsa karar vermemeli.
cor_mid = [
    FakeDS(ImagePositionPatient=[-FOV / 2, -60.0, 0.0], ImageOrientationPatient=CORONAL,
           PixelSpacing=[0.5, 0.5], Rows=200, Columns=200)
]
check("olu bolge: merkez ~0 -> karar yok",
      dio.detect_laterality(cor_mid, use_ipp_layer=True, ipp_deadzone=5.0)[0] is None)

# Geometri eksikse cokmemeli (IPP'ye geri dus).
bare = [FakeDS(ImagePositionPatient=[42.0, 0.0, 0.0])]
check("geometri eksik -> IPP kosesine geri dusuyor",
      abs(dio.image_center(bare[0])[0] - 42.0) < 1e-9)
check("IPP tamamen yoksa None", dio.image_center(FakeDS()) is None)

# Tag ve metin katmanlari hala 3. katmanin ONUNDE olmali.
tagged = FakeDS(Laterality="L", ImagePositionPatient=[-200.0, 0.0, 0.0],
                ImageOrientationPatient=CORONAL, PixelSpacing=[0.5, 0.5],
                Rows=200, Columns=200)
check("tag, x-isaretini eziyor",
      dio.detect_laterality([tagged], use_ipp_layer=True) == ("L", "Laterality"))

# ---------------------------------------------------------------------------
print()
print("--- study duzeyi taraf cozumu (resolve_study_laterality) ---")

def cor_series(center_x, tag=None, desc="", n=6):
    """Merkez-x'i verilen sentetik coronal seri."""
    half = ((200 - 1) / 2) * 0.5
    kw = dict(ImageOrientationPatient=CORONAL, PixelSpacing=[0.5, 0.5],
              Rows=200, Columns=200, SeriesDescription=desc)
    if tag:
        kw["Laterality"] = tag
    return [FakeDS(ImagePositionPatient=[center_x - half, -60.0, 3.0 * j], **kw)
            for j in range(n)]

# Faz 1A'da olculen gercek durum: 25 study'de tag seriler arasi CELISKILI,
# ama merkez-x hepsinde TEK taraf gosterdi -> cogunluk dogruyu bulmali.
study = [cor_series(+119, tag="R"),      # tag YANLIS (merkez-x sol diyor)
         cor_series(+111, tag="L"),
         cor_series(+119, tag="L"),
         cor_series(+136, tag="L"),
         cor_series(+118, tag="L")]
side, info = dio.resolve_study_laterality(study)
check("celiskili tag: cogunluk dogruyu buluyor (L)", side == "L", (side, info["n_L"], info["n_R"]))
check("celiski isaretlendi", info["conflict"] is True)

# Tag hic yoksa katman 3 devreye girer
study2 = [cor_series(-95), cor_series(-102), cor_series(-88)]
side2, info2 = dio.resolve_study_laterality(study2)
check("tag yok -> merkez-x ile R", side2 == "R", (side2, info2["sources"]))
check("kaynak image_center_x", info2["sources"] == ["image_center_x"])

# Olu bolge: diz izomerkezde -> karar verilmemeli (0-10mm kovasinda isabet 0.507)
study3 = [cor_series(+6), cor_series(-4), cor_series(+2)]
side3, info3 = dio.resolve_study_laterality(study3, ipp_deadzone=30.0)
check("izomerkezdeki diz -> karar YOK", side3 is None, (side3, info3["n_series_voted"]))

# Esitlikte tag/metin, merkez-x'i ezmeli
study4 = [cor_series(+80, tag="L"), cor_series(-80)]
side4, info4 = dio.resolve_study_laterality(study4)
check("esitlikte tag kazaniyor", side4 == "L", (side4, info4))

# Metin katmani hala tag'den sonra, merkez-x'ten once
study5 = [cor_series(-90, desc="SAG PD FS LEFT KNEE")]
side5, info5 = dio.resolve_study_laterality(study5)
check("metin, merkez-x'i eziyor", side5 == "L" and "SeriesDescription" in info5["sources"],
      (side5, info5["sources"]))

check("bos girdi -> None", dio.resolve_study_laterality([])[0] is None)
check("bos seri listesi atlaniyor", dio.resolve_study_laterality([[], cor_series(+90)])[0] == "L")

# Varsayilanlar Faz 1A kararini yansitmali
import inspect
d = inspect.signature(dio.detect_laterality).parameters
check("katman 3 varsayilan ACIK", d["use_ipp_layer"].default is True)
check("varsayilan olu bolge 30 mm", d["ipp_deadzone"].default == 30.0)

print("\n" + "=" * 60)
if FAILS:
    print("BASARISIZ (" + str(len(FAILS)) + "):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("TUM TESTLER GECTI")
