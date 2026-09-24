"""Katman 4 — Katman 1 ham ciktisindan egitim icin soft etiket dosyasi uret.

GPU gerekmiyor: Kaggle'dan inen JSONL yeterli, gerisi CPU'da.

Uretilen sutunlar (etiket basina):
  <etiket>_soft    egitim hedefi, [0,1].  present -> 0.5+0.5*conf
                                         absent  -> 0.5-0.5*conf
                                         uncertain/missing -> 0.5
  <etiket>_w       ornek-etiket agirligi (loss'ta carpan)
  <etiket>_status  present/absent/uncertain/missing (teshis icin)

Agirliklandirma gerekcesi (ARCHITECTURE Bolum 2, Katman 4):
  1. GOLD ornekler weak'ten cok daha agir (GOLD_WEIGHT).
  2. `uncertain`/`missing` olanlar loss'tan MASKELENIR (agirlik 0) — modelin
     bilmedigimiz seyi ogrenmesini istemiyoruz.
  3. Katman 3'te F1'i dusuk cikan etiketlerin weak agirligi dusurulur; olculen
     F1 dogrudan agirliga cevriliyor, elle katsayi uydurmuyoruz.

Kullanim:
  python tools/make_weak_labels.py outputs/phase1b/weak_layer1_minimal_VAR.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "labels"))
import devset_ids as D
import evaluate as EV
import prompt as P

GOLD_WEIGHT = 8.0        # gold ornekler weak'in kac kati
MIN_LABEL_W = 0.25       # en zayif etiket bile tamamen atilmasin
UNCERTAIN_W = 0.0        # uncertain -> loss'tan maskele


def main(jsonl_path):
    raw = pd.DataFrame([json.loads(l) for l in open(jsonl_path, encoding="utf-8")])
    train = pd.read_csv("outputs/train.csv")
    print(f"ham kayit: {len(raw):,}  parse_ok: {raw.parse_ok.mean():.1%}")

    gold_cols = ["StudyInstanceUID"] + P.LABELS
    is_gold = train[P.LABELS].notna().all(axis=1)
    gold = train.loc[is_gold, gold_cols]
    hold = train[train.StudyInstanceUID.isin(D.GOLD_HOLDOUT)][gold_cols]
    print(f"gold: {len(gold)}  (holdout {len(hold)})")

    # --- Katman 3: etiket basina F1, DOKUNULMAMIS holdout uzerinde ---
    ev = EV.evaluate(raw, hold, uncertain="absent")
    f1 = ev.set_index("etiket")["F1"].to_dict()
    print()
    print("Katman 3 (holdout) F1 -> etiket agirligi:")
    label_w = {}
    for k in P.LABELS:
        f = float(f1.get(k, np.nan))
        # F1 0.5 -> agirlik ~0, F1 1.0 -> agirlik 1. Rastgeleden iyi olmayan
        # bir etiketin weak sinyali ogretici degil.
        w = np.clip((f - 0.5) / 0.5, MIN_LABEL_W, 1.0) if np.isfinite(f) else MIN_LABEL_W
        label_w[k] = round(float(w), 3)
        print(f"  {k:<18} F1={f:.3f} -> w={label_w[k]}")

    # --- soft hedefler ---
    P.add_soft_labels(raw, conf_suffix="_conf")

    out = pd.DataFrame({"StudyInstanceUID": raw["StudyInstanceUID"]})
    out["parse_ok"] = raw["parse_ok"]
    out["exam_completeness"] = raw["exam_completeness"]
    gold_set = set(train.loc[is_gold, "StudyInstanceUID"])
    out["is_gold"] = out.StudyInstanceUID.isin(gold_set)
    out["is_holdout"] = out.StudyInstanceUID.isin(set(D.GOLD_HOLDOUT))

    gold_lookup = train.set_index("StudyInstanceUID")
    for k in P.LABELS:
        st = raw[f"{k}_status"].to_numpy()
        soft = raw[f"{k}_soft"].to_numpy(dtype=float)
        # Gold ornekte GERCEK etiket kullanilir, weak degil.
        g = out.StudyInstanceUID.map(gold_lookup[k]).to_numpy(dtype=float)
        use_gold = out["is_gold"].to_numpy() & np.isfinite(g)
        soft = np.where(use_gold, g, soft)

        w = np.where(np.isin(st, ["uncertain", "missing"]), UNCERTAIN_W, label_w[k])
        w = np.where(use_gold, GOLD_WEIGHT, w)        # gold her zaman agir ve maskesiz
        out[f"{k}_soft"] = np.round(soft, 4)
        out[f"{k}_w"] = np.round(w, 4)
        out[f"{k}_status"] = np.where(use_gold, "gold", st)

    Path("outputs/phase1b").mkdir(parents=True, exist_ok=True)
    dst = "outputs/phase1b/weak_labels.csv"
    out.to_csv(dst, index=False)

    print()
    print("=" * 66)
    print(f"kaydedildi: {dst}   ({len(out):,} satir, {out.shape[1]} sutun)")
    print("=" * 66)
    wcols = [f"{k}_w" for k in P.LABELS]
    masked = (out[wcols] == 0).to_numpy().mean()
    print(f"  maskelenen (etiket,ornek) cifti : {masked:.1%}")
    print(f"  gold satir                      : {int(out.is_gold.sum())}"
          f"  (holdout {int(out.is_holdout.sum())})")
    print(f"  ortalama etiket agirligi (weak)  : "
          f"{np.mean([label_w[k] for k in P.LABELS]):.3f}")
    print()
    print("  Egitimde kullanim (Faz 2.8):")
    print("    loss = BCE(pred, <etiket>_soft, reduction='none') * <etiket>_w")
    print("    validation SADECE is_holdout satirlarinda (gold-only kurali)")
    return out


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "outputs/phase1b/weak_layer1_minimal_VAR.jsonl"
    main(src)
