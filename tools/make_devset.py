"""Faz 1.8 icin gelistirme seti ve gold ayrimi uret.

Neden ayri bir script: prompt'u iterasyonla duzeltirken bir olcut lazim, ama 58
gold study'nin TAMAMINI prompt ayarlamak icin kullanirsak Katman 3 dogrulamasi
kontamine olur (kendi ayar setinde olcum yapmak).

Ayrim:
  gold_dev      20 study  -> prompt'u bunlarda ayarla, serbestce bak
  gold_holdout  38 study  -> DOKUNMA. Katman 3'un nihai olcumu
  devset        50 rapor  -> gold_dev + dile gore tabakali ek ornekler
"""
import pathlib
import sys

import numpy as np
import pandas as pd

SEED = 42
N_GOLD_DEV = 20
N_DEVSET = 50

train = pd.read_csv("outputs/train.csv")
meta = pd.read_csv("outputs/phase1a/study_metadata.csv")[
    ["StudyInstanceUID", "lang", "is_gold"]]
df = train.merge(meta, on="StudyInstanceUID", how="left")
df["n_char"] = df["Report"].fillna("").str.len()

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
gold = df[df[LABELS].notna().all(axis=1)].copy()
print(f"gold study: {len(gold)}")

# gold_dev secimi: her etiketin pozitiflerini dev/holdout arasinda PAYLI bol.
# Neden rastgele arama: tek gecisli acgozlu bir secim (orn. "nadir pozitifi olanlari
# one al") holdout'u tam olcmek istedigimiz vakalardan bosaltiyor. 12 etiketi birden
# dengelemek kucuk bir kombinatoryal problem; 58 study'de rastgele arama anında
# cozuyor ve ne optimize ettigi seffaf.
TARGET = N_GOLD_DEV / len(gold)          # dev'e dusmesi gereken pay (~0.345)
rng = np.random.default_rng(SEED)
idx = gold.index.to_numpy()
best, best_cost = None, np.inf
for _ in range(4000):
    cand = rng.choice(idx, size=N_GOLD_DEV, replace=False)
    dev = gold.loc[cand]
    cost = 0.0
    for k in LABELS:
        tot = float(gold[k].sum())
        if tot == 0:
            continue
        frac = float(dev[k].sum()) / tot
        # nadir etiketlerde sapmayi daha agir cezalandir
        cost += ((frac - TARGET) ** 2) * (1.0 / max(tot, 1.0)) * 100
    # dil cesitliligi de istiyoruz: dev'de kac dil var
    cost += 0.02 * (gold["lang"].nunique() - dev["lang"].nunique()) ** 2
    if cost < best_cost:
        best, best_cost = cand, cost

gold_dev = gold.loc[best]
gold_holdout = gold.drop(index=best)
print(f"gold_dev {len(gold_dev)} | gold_holdout {len(gold_holdout)}  "
      f"(hedef dev payi {TARGET:.3f})")
print("  gold_dev dil sayisi:", gold_dev["lang"].nunique(),
      "/", gold["lang"].nunique())
print()
print("  etiket bazinda pozitif dagilimi:")
print(f"    {'etiket':<18}{'toplam':>7}{'dev':>5}{'holdout':>9}{'dev_payi':>10}")
for k in sorted(LABELS, key=lambda c: gold[c].sum()):
    tot = int(gold[k].sum()); dv = int(gold_dev[k].sum())
    print(f"    {k:<18}{tot:>7}{dv:>5}{tot-dv:>9}{dv/max(tot,1):>10.2f}")

# --- devset: gold_dev + tabakali ek ---
rest = df[~df["StudyInstanceUID"].isin(gold["StudyInstanceUID"])]
n_extra = N_DEVSET - len(gold_dev)
extra = []
# 1) en kisa raporlar (telegrafik -> "partial" karari zor)
extra.append(rest.nsmallest(6, "n_char"))
# 2) dil '?' olanlar (muhtemelen karisik dilli)
q = rest[rest["lang"] == "?"]
if len(q):
    extra.append(q.sample(n=min(4, len(q)), random_state=SEED))
# 3) kalan kotayi dile gore paya yakin bicimde doldur
used = pd.concat(extra)["StudyInstanceUID"] if extra else pd.Series(dtype=object)
pool = rest[~rest["StudyInstanceUID"].isin(used)]
need = n_extra - sum(len(e) for e in extra)
if need > 0:
    share = pool["lang"].value_counts(normalize=True)
    for lg, p in share.items():
        k = int(round(p * need))
        if k > 0:
            s = pool[pool["lang"] == lg]
            extra.append(s.sample(n=min(k, len(s)), random_state=SEED))
devset = pd.concat([gold_dev] + extra).drop_duplicates("StudyInstanceUID").head(N_DEVSET)

cols = ["StudyInstanceUID", "lang", "n_char", "Report", "is_gold"] + LABELS
devset[cols].to_csv("outputs/phase1a/devset_50.csv", index=False)
gold_dev[["StudyInstanceUID"]].assign(split="gold_dev").to_csv(
    "outputs/phase1a/gold_split.csv", index=False)
gold_holdout[["StudyInstanceUID"]].assign(split="gold_holdout").to_csv(
    "outputs/phase1a/gold_split.csv", mode="a", header=False, index=False)

print(f"\ndevset: {len(devset)} rapor")
print("  dil       :", dict(devset["lang"].value_counts()))
print("  gold olan :", int(devset["is_gold"].fillna(False).sum()))
print(f"  uzunluk   : medyan {devset['n_char'].median():.0f}, "
      f"min {devset['n_char'].min()}, max {devset['n_char'].max()}")
print("\nkaydedildi: outputs/phase1a/devset_50.csv")
print("kaydedildi: outputs/phase1a/gold_split.csv  <- gold_holdout'a DOKUNMA")

# --- UID listelerini Python modulu olarak da yaz ---------------------------
# Neden: 03_weak_labels notebook'u Kaggle'da calisacak ve dev set'i tanimasi
# lazim. Ayri bir Kaggle Dataset yuklemek yerine UID listesini koda gomuyoruz
# (~7 KB) — bir adim eksik, bayatlama riski yok.
mod = pathlib.Path("src/labels/devset_ids.py")
mod.parent.mkdir(parents=True, exist_ok=True)
with mod.open("w", encoding="utf-8") as fh:
    fh.write('"""tools/make_devset.py tarafindan URETILDI — elle duzenlemeyin.\n\n')
    fh.write("Faz 1.8 dev set'i ve gold ayrimi. UID listeleri koda gomulu ki\n")
    fh.write("notebooks/03_weak_labels_llm Kaggle'da ek dataset olmadan calissin.\n")
    fh.write('"""\n\n')
    for name, ids in [("DEVSET", devset["StudyInstanceUID"].tolist()),
                      ("GOLD_DEV", gold_dev["StudyInstanceUID"].tolist()),
                      ("GOLD_HOLDOUT", gold_holdout["StudyInstanceUID"].tolist())]:
        fh.write(f"{name} = [\n")
        for u in ids:
            fh.write(f'    "{u}",\n')
        fh.write("]\n\n")
    fh.write("assert not (set(GOLD_DEV) & set(GOLD_HOLDOUT)), "
             '"gold_dev ve gold_holdout kesisiyor"\n')
print(f"kaydedildi: {mod}  (DEVSET {len(devset)}, GOLD_DEV {len(gold_dev)}, "
      f"GOLD_HOLDOUT {len(gold_holdout)})")
