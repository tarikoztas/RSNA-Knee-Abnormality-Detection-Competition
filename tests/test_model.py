"""src/models/model.py — sekil ve mantik testleri, CPU'da saniyeler icinde.

Neden yerelde ve sentetik: GPU'da bulunacak bir sekil hatasini CPU'da bulmak,
6 saatlik egitim kosusunun 3. dakikasinda `shape mismatch` almaktan iyi.
Ayni yaklasim Faz 0'da sentetik DICOM ile pandas hatasini, Faz 1'de butce
sikismasini yakalamisti.

pretrained=False: agirlik indirmeye gerek yok, sekiller ayni.

Kullanim: python tests/test_model.py
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src" / "models"))
import model as M

FAILS = []


def check(name, cond, extra=None):
    print(("  OK   " if cond else "  FAIL ") + name +
          ("" if extra is None else "  " + str(extra)))
    if not cond:
        FAILS.append(name)


B, S, C, H, W = 2, 3, 5, 64, 64          # 2 study, 3 seri, 5 kanal (2.5D k=2)
L = len(M.LABELS)
x = torch.randn(B, S, C, H, W)
mask = torch.ones(B, S)

print("--- mean pooling modeli ---")
torch.manual_seed(0)
m = M.KneeModel(pretrained=False, in_chans=C, pooling="mean")
m.eval()
with torch.no_grad():
    out = m(x, mask)
check("cikti sekli (B, 12)", out.shape == (B, L), tuple(out.shape))
check("sigmoid UYGULANMAMIS (logit)", bool((out.abs() > 1e-6).any()))
check("NaN yok", bool(torch.isfinite(out).all()))

print()
print("--- attention pooling modeli ---")
torch.manual_seed(0)
ma = M.KneeModel(pretrained=False, in_chans=C, pooling="attention")
ma.eval()
with torch.no_grad():
    outa = ma(x, mask)
check("cikti sekli (B, 12)", outa.shape == (B, L), tuple(outa.shape))
check("NaN yok", bool(torch.isfinite(outa).all()))

print()
print("--- 2.5D kanal sayisi degistirilebiliyor ---")
for c in (3, 5, 7):
    mm = M.KneeModel(pretrained=False, in_chans=c, pooling="mean")
    mm.eval()
    with torch.no_grad():
        o = mm(torch.randn(1, 2, c, 64, 64), torch.ones(1, 2))
    check(f"in_chans={c}", o.shape == (1, L), tuple(o.shape))

print()
print("--- MASKE: eksik seri embedding'i sulandirmamali ---")
# Study 0: 3 gercek seri. Study 1: 1 gercek + 2 sahte (maskeli).
x2 = x.clone()
x2[1, 1:] = 999.0                        # sahte serilere sacma deger koy
mask2 = torch.tensor([[1., 1., 1.], [1., 0., 0.]])
torch.manual_seed(0)
m2 = M.KneeModel(pretrained=False, in_chans=C, pooling="mean")
m2.eval()
with torch.no_grad():
    o_masked = m2(x2, mask2)
    # Ayni study'yi SADECE gercek seriyle, tek seri olarak calistir
    o_solo = m2(x2[1:2, :1], torch.ones(1, 1))
check("maskeli seri ciktiyi etkilemiyor",
      torch.allclose(o_masked[1], o_solo[0], atol=1e-4),
      float((o_masked[1] - o_solo[0]).abs().max()))

print()
print("--- attention'da da maske calisiyor ---")
torch.manual_seed(0)
ma2 = M.KneeModel(pretrained=False, in_chans=C, pooling="attention")
ma2.eval()
with torch.no_grad():
    a_masked = ma2(x2, mask2)
    a_solo = ma2(x2[1:2, :1], torch.ones(1, 1))
check("maskeli seri ciktiyi etkilemiyor",
      torch.allclose(a_masked[1], a_solo[0], atol=1e-4),
      float((a_masked[1] - a_solo[0]).abs().max()))
with torch.no_grad():
    allzero = ma2(x2[:1], torch.zeros(1, S))
check("tum seriler maskeliyse NaN uretmiyor", bool(torch.isfinite(allzero).all()))

print()
print("--- seri metadata enjeksiyonu (Faz 3.3) ---")
mmeta = M.KneeModel(pretrained=False, in_chans=C, pooling="mean", meta_dim=5)
mmeta.eval()
with torch.no_grad():
    om = mmeta(x, mask, meta=torch.randn(B, S, 5))
check("metadata ile cikti sekli", om.shape == (B, L), tuple(om.shape))

print()
print("--- masked_bce ---")
logits = torch.zeros(2, L)               # sigmoid(0) = 0.5
target = torch.full((2, L), 0.5)
w = torch.ones(2, L)
loss = M.masked_bce(logits, target, w)
check("hedef=tahmin=0.5 -> loss = ln2", abs(float(loss) - 0.6931) < 1e-3, float(loss))

w0 = torch.zeros(2, L)
check("tum agirliklar 0 -> loss 0", float(M.masked_bce(logits, target, w0)) == 0.0)

# Maskeli cift loss'a GIRMEMELI: sadece maskeli hucreyi bozup loss ayni mi?
t_bad = target.clone(); t_bad[0, 0] = 1.0
w_mask = torch.ones(2, L); w_mask[0, 0] = 0.0
l_ref = M.masked_bce(logits, target, w_mask)
l_bad = M.masked_bce(logits, t_bad, w_mask)
check("maskeli hucre loss'u degistirmiyor", abs(float(l_ref) - float(l_bad)) < 1e-9,
      (float(l_ref), float(l_bad)))

# Payda: agirlikli ORTALAMA olmali, agirlikli TOPLAM degil
w_half = torch.full((2, L), 0.5)
check("payda agirlik toplami (olcek degismez)",
      abs(float(M.masked_bce(logits, target, w_half)) - float(loss)) < 1e-6,
      (float(M.masked_bce(logits, target, w_half)), float(loss)))

# Gold agirligi weak'ten agir basmali.
# DIKKAT: logits=0'da sigmoid=0.5 ve BCE hedeften BAGIMSIZ olarak ln2 cikar —
# ilk yazdigim test bu yuzden kayan nokta gurultusuyle "gecmis"ti, gercek
# davranisi hic sinamiyordu. Sifir olmayan logit sart.
lg = torch.full((2, L), 2.0)             # sigmoid(2) ~ 0.88
t_g = torch.zeros(2, L)                  # hedef 0 -> model cok yaniliyor
w_gold = torch.ones(2, L); w_gold[0] = 8.0
l_flat = M.masked_bce(lg, t_g, torch.ones(2, L))
check("sifir olmayan logit'te hedef gercekten onemli",
      abs(float(M.masked_bce(lg, torch.ones(2, L), torch.ones(2, L))) -
          float(l_flat)) > 1.0)
# Sadece 0. satiri yanlis, 1. satir dogru yap: gold agirligi o hatayi buyutmeli
t_mix = torch.zeros(2, L); t_mix[1] = 1.0
l_gold = M.masked_bce(lg, t_mix, w_gold)
l_even = M.masked_bce(lg, t_mix, torch.ones(2, L))
check("gold satirinin hatasi agirlikla buyuyor", float(l_gold) > float(l_even) + 0.1,
      (round(float(l_gold), 4), round(float(l_even), 4)))

print()
print("--- geri yayilim calisiyor mu ---")
m.train()
out = m(x, mask)
loss = M.masked_bce(out, torch.rand(B, L), torch.ones(B, L))
loss.backward()
g = [p.grad for p in m.parameters() if p.grad is not None]
check("gradyan olustu", len(g) > 0, f"{len(g)} tensor")
check("gradyanlar sonlu", all(bool(torch.isfinite(t).all()) for t in g))
check("ilk conv gradyani sifir degil (5 kanal ogreniyor)",
      bool(next(m.backbone.parameters()).grad.abs().sum() > 0))

print()
print("=" * 60)
if FAILS:
    print("BASARISIZ (" + str(len(FAILS)) + "):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("TUM MODEL TESTLERI GECTI")
