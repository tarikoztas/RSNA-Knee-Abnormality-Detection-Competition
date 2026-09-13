"""Katman 3 — weak-labeler'i gold uzerinde degerlendir.

Merkezi tasarim sorusu: `uncertain` ciktisini nasil sayalim?

Uc yol var ve UCU DE raporlanmali, cunku farkli sorulara cevap veriyorlar:

  maskele  : uncertain olanlari metrikten CIKAR
             -> "karar verdiginde ne kadar dogru?" (precision/recall karar verilen
                alt kumede). Kapsami gizler: her seye uncertain diyen bir model
                burada mukemmel gorunur.
  absent   : uncertain -> 0 kabul et
             -> gercekci senaryo, cunku egitimde uncertain'i maskelemezsek boyle
                davranmis oluruz. Recall'u dusurur.
  0.5      : uncertain -> 0.5 soft hedef
             -> AUC icin dogru yol: siralamaya katilir ama ne pozitif ne negatif
                tarafa tam agirlik verir.

AUC ayrica ozel: yarismanin metrigi macro ROC-AUC, yani weak-labeler'in
SIRALAMA kalitesi F1'inden daha alakali. Bu yuzden hem F1 hem AUC hesaplaniyor.
"""
import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def _auc(y, s):
    """ROC-AUC, sklearn'e bagimlilik olmadan (Mann-Whitney U, baglar 0.5 sayilir)."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # bagli skorlara ortalama rank ver
    df = pd.DataFrame({"s": s, "r": ranks})
    ranks = df.groupby("s")["r"].transform("mean").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else np.nan
    r = tp / (tp + fn) if (tp + fn) else np.nan
    f = 2 * p * r / (p + r) if (p and r and (p + r) > 0) else (0.0 if (tp + fp + fn) else np.nan)
    return p, r, f


def evaluate(pred: pd.DataFrame, gold: pd.DataFrame, uncertain="mask") -> pd.DataFrame:
    """Per-label metrikler.

    pred : StudyInstanceUID + her etiket icin <k> (1/0/None) ve <k>_status
    gold : StudyInstanceUID + her etiket icin 0/1
    uncertain: "mask" | "absent" | "half"
    """
    g = gold.set_index("StudyInstanceUID")
    p = pred.set_index("StudyInstanceUID")
    common = g.index.intersection(p.index)
    g, p = g.loc[common], p.loc[common]

    rows = []
    for k in LABELS:
        if k not in g.columns or k not in p.columns:
            continue
        yt = g[k].astype(float).to_numpy()
        raw = p[k].to_numpy(dtype=object)
        st = p.get(f"{k}_status", pd.Series(index=p.index, dtype=object)).to_numpy()

        is_unc = np.array([s in ("uncertain", "missing") or v is None
                           for s, v in zip(st, raw)])
        yp = np.array([np.nan if v is None else float(v) for v in raw])

        if uncertain == "mask":
            keep = ~is_unc & np.isfinite(yt)
        elif uncertain == "absent":
            yp = np.where(is_unc, 0.0, yp)
            keep = np.isfinite(yt)
        elif uncertain == "half":
            yp = np.where(is_unc, 0.5, yp)
            keep = np.isfinite(yt)
        else:
            raise ValueError(uncertain)

        yt_k, yp_k = yt[keep], yp[keep]
        hard = (yp_k >= 0.5).astype(int)
        tp = int(((hard == 1) & (yt_k == 1)).sum())
        fp = int(((hard == 1) & (yt_k == 0)).sum())
        fn = int(((hard == 0) & (yt_k == 1)).sum())
        tn = int(((hard == 0) & (yt_k == 0)).sum())
        prec, rec, f1 = _prf(tp, fp, fn)

        rows.append({
            "etiket": k,
            "n_deger": int(keep.sum()),
            "kapsam": round(float((~is_unc).mean()), 3),
            "n_gold_poz": int((yt == 1).sum()),
            "precision": round(prec, 3) if np.isfinite(prec) else np.nan,
            "recall": round(rec, 3) if np.isfinite(rec) else np.nan,
            "F1": round(f1, 3) if np.isfinite(f1) else np.nan,
            "AUC": round(_auc(yt_k, yp_k), 3),
            "dogruluk": round((tp + tn) / max(len(yt_k), 1), 3),
        })
    return pd.DataFrame(rows)


def summarise(ev: pd.DataFrame) -> dict:
    return {
        "macro_F1": round(float(ev["F1"].mean(skipna=True)), 4),
        "macro_AUC": round(float(ev["AUC"].mean(skipna=True)), 4),
        "min_F1": round(float(ev["F1"].min(skipna=True)), 3),
        "en_zayif": ev.loc[ev["F1"].idxmin(), "etiket"] if ev["F1"].notna().any() else None,
        "ort_kapsam": round(float(ev["kapsam"].mean()), 3),
    }


def confidence_quality(pred: pd.DataFrame, gold: pd.DataFrame,
                       conf_suffix="_conf") -> pd.DataFrame:
    """Guven skoru gercekten dogrulukla korele mi?

    Katman 4 weak-label'lari guvene gore agirliklandiracak. Guven rastgeleyse
    agirliklandirma zarar verir. Bunu iki secenek icin ayri ayri olculuyoruz:
      _conf   : modelin kendi bildirdigi guven
      _lpconf : status token'inin logprob'undan turetilen guven
    """
    g = gold.set_index("StudyInstanceUID")
    p = pred.set_index("StudyInstanceUID")
    common = g.index.intersection(p.index)
    g, p = g.loc[common], p.loc[common]

    rows = []
    for k in LABELS:
        col = f"{k}{conf_suffix}"
        if k not in g.columns or col not in p.columns:
            continue
        yt = g[k].astype(float).to_numpy()
        raw = p[k].to_numpy(dtype=object)
        conf = pd.to_numeric(p[col], errors="coerce").to_numpy()
        m = np.array([v is not None for v in raw]) & np.isfinite(conf) & np.isfinite(yt)
        if m.sum() < 8:
            continue
        correct = np.array([float(v) == t for v, t in zip(raw[m], yt[m])], float)
        c = conf[m]
        if len(np.unique(c)) < 2 or len(np.unique(correct)) < 2:
            rows.append({"etiket": k, "n": int(m.sum()), "AUC_guven": np.nan,
                         "dogru_ort": np.nan, "yanlis_ort": np.nan})
            continue
        rows.append({
            "etiket": k, "n": int(m.sum()),
            "AUC_guven": round(_auc(correct, c), 3),
            "dogru_ort": round(float(c[correct == 1].mean()), 3),
            "yanlis_ort": round(float(c[correct == 0].mean()), 3),
        })
    return pd.DataFrame(rows)
