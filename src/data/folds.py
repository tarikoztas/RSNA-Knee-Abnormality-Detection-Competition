"""Fold atamasi ve site proxy normalizasyonu (Faz 1.12 / 1.13).

Karar gerekcesi: PHASE1A_FINDINGS.md Bolum 3, ARCHITECTURE.md Bolum 3.
Ozet: grup anahtari = (rapor dili, uretici ailesi). Amac, yeterli grup sayisi ve
denge saglayan EN KABA anahtari secmek — anahtari inceltmek tek bir fiziksel
cihazi ikiye bolebilir ve sizintiyi geri cagirir.
"""
import numpy as np
import pandas as pd

# --- Uretici adi -> aile eslemesi -------------------------------------------
# Faz 1A'da 11 ham uretici adi bulundu; bunlar 5 fiziksel aileye karsilik geliyor.
# strip().upper() bunlari birlestirmiyor, yani esleme olmadan ayni cihaz ailesi
# birden fazla gruba bolunur. Sirket satin almalari da dahil:
#   Canon, Toshiba Medical'i satin aldi  ·  Fujifilm, Hitachi'nin MR bolumunu aldi
MANUFACTURER_FAMILY = {
    "SIEMENS": "SIEMENS",
    "SIEMENS HEALTHINEERS": "SIEMENS",
    "PHILIPS": "PHILIPS",
    "PHILIPS MEDICAL SYSTEMS": "PHILIPS",
    "PHILIPS HEALTHCARE": "PHILIPS",
    "GE MEDICAL SYSTEMS": "GE",
    "GEHC": "GE",
    "GE HEALTHCARE": "GE",
    "TOSHIBA": "CANON",
    "CANON_MEC": "CANON",
    "CANON MEDICAL SYSTEMS": "CANON",
    "FUJIFILM": "FUJIFILM",
    "FUJIFILM HEALTHCARE CORPORATION": "FUJIFILM",
    "HITACHI": "FUJIFILM",
    "HITACHI MEDICAL CORPORATION": "FUJIFILM",
}

# --- Dil kumesi eslemesi ----------------------------------------------------
# fasttext sh/hr/bs'yi ayri etiketler ama bunlar ayni dil kumesi (Sirp-Hirvat-Bosnak).
# Ayri tutmak grup anahtarini gereksiz inceltir.
LANGUAGE_GROUP = {"sh": "hbs", "hr": "hbs", "bs": "hbs", "sr": "hbs"}


def manufacturer_family(s: pd.Series) -> pd.Series:
    """Ham Manufacturer degerlerini uretici ailesine indir."""
    norm = s.fillna("NA").astype(str).str.strip().str.upper()
    return norm.map(lambda x: MANUFACTURER_FAMILY.get(x, x))


def language_group(s: pd.Series) -> pd.Series:
    """Ayni dil kumesindeki kodlari birlestir."""
    return s.fillna("?").astype(str).replace(LANGUAGE_GROUP)


def build_group_key(df: pd.DataFrame, lang_col: str = "lang",
                    mfr_col: str = "Manufacturer") -> pd.Series:
    """Study basina site proxy grup anahtari: '<dil> | <uretici ailesi>'."""
    return (language_group(df[lang_col]) + " | " + manufacturer_family(df[mfr_col]))


def group_kfold_assign(groups, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Gruplari fold'lara ata; her grup TAMAMEN tek bir fold'a gider.

    sklearn.model_selection.GroupKFold ile ayni acgozlu algoritma: gruplari
    buyukten kucuge sirala, her birini o an en az dolu fold'a koy. Bu, fold
    boyutlarini dengelerken grup butunlugunu korur.

    sklearn'e bagimlilik yok — yerel makinede kurulu olmasi gerekmiyor ve
    algoritma yeterince basit ki seffaf olmasi tercih edilir.
    """
    g = pd.Series(groups).astype(str)
    vc = g.value_counts()
    # Esit boyutlu gruplar arasinda deterministik sira icin isme gore de sirala
    order = sorted(vc.index, key=lambda k: (-vc[k], k))
    load = np.zeros(n_splits, dtype=np.int64)
    assign = {}
    for name in order:
        f = int(np.argmin(load))
        assign[name] = f
        load[f] += int(vc[name])
    return g.map(assign).to_numpy()


def make_folds(study_meta: pd.DataFrame, n_splits: int = 5,
               seed: int = 42) -> pd.DataFrame:
    """study_metadata benzeri bir tablodan fold atamasi uret.

    Beklenen sutunlar: StudyInstanceUID, lang, Manufacturer (+ varsa is_gold).
    Dondurur: StudyInstanceUID, group_key, fold (+ is_gold korunur).
    """
    out = pd.DataFrame({"StudyInstanceUID": study_meta["StudyInstanceUID"].values})
    out["group_key"] = build_group_key(study_meta).values
    out["fold"] = group_kfold_assign(out["group_key"], n_splits, seed)
    if "is_gold" in study_meta.columns:
        out["is_gold"] = study_meta["is_gold"].values
    return out


def fold_report(folds: pd.DataFrame) -> pd.DataFrame:
    """Fold dengesini ve gold dagilimini ozetle — atama sonrasi saglama icin."""
    rows = []
    for f, d in folds.groupby("fold"):
        rows.append({
            "fold": int(f),
            "n_study": len(d),
            "n_grup": d["group_key"].nunique(),
            "n_gold": int(d["is_gold"].sum()) if "is_gold" in d else -1,
        })
    return pd.DataFrame(rows)
