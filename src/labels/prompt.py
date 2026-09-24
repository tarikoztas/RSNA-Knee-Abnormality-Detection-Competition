"""Katman 1 — rapor metninden 12 bulgunun yapilandirilmis cikarimi.

Saglayicidan BAGIMSIZ: burada sadece prompt metni, JSON semasi ve ayristirici var.
Hangi modele gonderilecegi cagiran tarafin isi (src/labels/run_extract.py).

Tasarim gerekcesi: ARCHITECTURE.md Bolum 2, Katman 1.
"""
import json
import re

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

# Etiket tanimlari prompt'a AYNEN giriyor. Bunlar yarismanin kendi tanimlari —
# ozellikle kompartman ayrimlari (Medial/Lateral/PF OA) modelin kafasini
# karistirabilecek tek yer, o yuzden acik yazildi.
LABEL_DEFINITIONS = """\
1.  ACL              — anterior cruciate ligament injury: tear (partial or complete),
                       sprain, rupture, or graft failure.
2.  MCL              — medial collateral ligament injury: tear, sprain, grade I-III.
3.  Medial Meniscus  — medial meniscus tear of any type (horizontal, vertical, radial,
                       complex, root tear, bucket-handle). Degeneration WITHOUT tear
                       does not count.
4.  Lateral Meniscus — same, for the lateral meniscus.
5.  Medial OA        — osteoarthritis of the MEDIAL tibiofemoral compartment:
                       cartilage loss, joint space narrowing, subchondral sclerosis,
                       osteophytes in that compartment.
6.  Lateral OA       — same, for the LATERAL tibiofemoral compartment.
7.  PF OA            — patellofemoral osteoarthritis: chondral loss / osteophytes
                       between patella and femoral trochlea. Note this is a SEPARATE
                       compartment from 5 and 6.
8.  Effusion         — joint effusion / intra-articular fluid accumulation.
9.  Synovitis        — synovial inflammation, synovial thickening, synovial enhancement.
                       Effusion ALONE is not synovitis.
10. Baker's          — Baker's cyst / popliteal cyst / parameniscal cyst in the
                       popliteal fossa.
11. Contusion        — bone contusion / bone marrow oedema / bone bruise WITHOUT a
                       fracture line.
12. Fracture         — any fracture: cortical, subchondral, avulsion, insufficiency,
                       stress fracture, osteochondral fracture with a fracture line."""

FEW_SHOT = [
    # Ingilizce — normal, kapsamli rapor. "absent" kararinin nasil verilecegini gosterir.
    {
        "report": "MRI right knee. Technique: sagittal, coronal and axial PD and "
                  "T2-weighted fat-saturated sequences. Findings: Cruciate and "
                  "collateral ligaments are intact. Both menisci are of normal "
                  "signal and morphology, no tear. Articular cartilage is preserved "
                  "in all three compartments. No joint effusion. No bone marrow "
                  "oedema or fracture. Impression: Normal MRI of the right knee.",
        "output": {
            "exam_completeness": "comprehensive",
            "findings": {
                "ACL": {"status": "absent", "confidence": 0.97, "evidence": "Cruciate ... ligaments are intact"},
                "MCL": {"status": "absent", "confidence": 0.96, "evidence": "collateral ligaments are intact"},
                "Medial Meniscus": {"status": "absent", "confidence": 0.97, "evidence": "Both menisci ... no tear"},
                "Lateral Meniscus": {"status": "absent", "confidence": 0.97, "evidence": "Both menisci ... no tear"},
                "Medial OA": {"status": "absent", "confidence": 0.93, "evidence": "cartilage is preserved in all three compartments"},
                "Lateral OA": {"status": "absent", "confidence": 0.93, "evidence": "cartilage is preserved in all three compartments"},
                "PF OA": {"status": "absent", "confidence": 0.93, "evidence": "cartilage is preserved in all three compartments"},
                "Effusion": {"status": "absent", "confidence": 0.97, "evidence": "No joint effusion"},
                "Synovitis": {"status": "absent", "confidence": 0.80, "evidence": ""},
                "Baker's": {"status": "absent", "confidence": 0.75, "evidence": ""},
                "Contusion": {"status": "absent", "confidence": 0.96, "evidence": "No bone marrow oedema"},
                "Fracture": {"status": "absent", "confidence": 0.97, "evidence": "or fracture"},
            },
        },
    },
    # Almanca — pozitif bulgular + kompartman ayrimi + negation ayni raporda.
    {
        "report": "MRT linkes Knie. Befund: Komplette Ruptur des vorderen Kreuzbandes. "
                  "Horizontalriss des Innenmeniskus im Hinterhorn. Aussenmeniskus "
                  "unauffaellig. Deutlicher Gelenkerguss. Retropatellar hoehergradige "
                  "Chondropathie mit Osteophyten. Medialer und lateraler "
                  "Femorotibialgelenkspalt regelrecht. Kein Knochenmarksoedem, keine "
                  "Fraktur. Kleine Bakerzyste.",
        "output": {
            "exam_completeness": "comprehensive",
            "findings": {
                "ACL": {"status": "present", "confidence": 0.98, "evidence": "Komplette Ruptur des vorderen Kreuzbandes"},
                "MCL": {"status": "uncertain", "confidence": 0.40, "evidence": ""},
                "Medial Meniscus": {"status": "present", "confidence": 0.96, "evidence": "Horizontalriss des Innenmeniskus"},
                "Lateral Meniscus": {"status": "absent", "confidence": 0.93, "evidence": "Aussenmeniskus unauffaellig"},
                "Medial OA": {"status": "absent", "confidence": 0.88, "evidence": "Medialer ... Femorotibialgelenkspalt regelrecht"},
                "Lateral OA": {"status": "absent", "confidence": 0.88, "evidence": "lateraler Femorotibialgelenkspalt regelrecht"},
                "PF OA": {"status": "present", "confidence": 0.92, "evidence": "Retropatellar hoehergradige Chondropathie mit Osteophyten"},
                "Effusion": {"status": "present", "confidence": 0.97, "evidence": "Deutlicher Gelenkerguss"},
                "Synovitis": {"status": "uncertain", "confidence": 0.30, "evidence": ""},
                "Baker's": {"status": "present", "confidence": 0.96, "evidence": "Kleine Bakerzyste"},
                "Contusion": {"status": "absent", "confidence": 0.95, "evidence": "Kein Knochenmarksoedem"},
                "Fracture": {"status": "absent", "confidence": 0.96, "evidence": "keine Fraktur"},
            },
        },
    },
    # Turkce + TELEGRAFIK rapor. Kisa raporda "absent" DEGIL "uncertain" denmesi
    # gerektigini ogretir — Faz 1A'da raporlarin %? kadari bu kadar kisa.
    {
        "report": "Diz MR. Bulgular: Medial menisküs posterior boynuzda yırtık. "
                  "Eklem içinde efüzyon mevcut.",
        "output": {
            "exam_completeness": "partial",
            "findings": {
                "ACL": {"status": "uncertain", "confidence": 0.20, "evidence": ""},
                "MCL": {"status": "uncertain", "confidence": 0.20, "evidence": ""},
                "Medial Meniscus": {"status": "present", "confidence": 0.96, "evidence": "Medial menisküs posterior boynuzda yırtık"},
                "Lateral Meniscus": {"status": "uncertain", "confidence": 0.25, "evidence": ""},
                "Medial OA": {"status": "uncertain", "confidence": 0.20, "evidence": ""},
                "Lateral OA": {"status": "uncertain", "confidence": 0.20, "evidence": ""},
                "PF OA": {"status": "uncertain", "confidence": 0.20, "evidence": ""},
                "Effusion": {"status": "present", "confidence": 0.96, "evidence": "Eklem içinde efüzyon mevcut"},
                "Synovitis": {"status": "uncertain", "confidence": 0.20, "evidence": ""},
                "Baker's": {"status": "uncertain", "confidence": 0.20, "evidence": ""},
                "Contusion": {"status": "uncertain", "confidence": 0.20, "evidence": ""},
                "Fracture": {"status": "uncertain", "confidence": 0.20, "evidence": ""},
            },
        },
    },
]

# "minimal -> uncertain" kurali AYRI tutuluyor cunku ABLATION adayi: Tur 3'te
# kapsam 0.788 -> 0.662 dustu ve F1_absent 0.7251 -> 0.6805 geriledi. Bas suphe
# bu kural, ama ayni turda gudumlu decoding de degistigi icin atif yapilamadi.
# Acip kapatarak olcebilmek icin sablondan soyuldu.
RULE_MINIMAL = '''7. MINIMAL FINDINGS. If the only wording is "minimal", "trace", "mild",
   "grade I", "questionable" or "suspected", prefer "uncertain" (0.4-0.6) over
   "present". A definite finding is what "present" is for.

'''

SYSTEM_PROMPT_TEMPLATE = """\
You extract structured findings from knee MRI radiology reports.

Reports come from 19+ imaging centres on 5 continents and are written in at least
9 different languages. Do not ask for or state the language — read it and work in it.

For EACH of the 12 findings below, output exactly one status:

  "present"   — the report states or strongly implies the finding IS there
  "absent"    — the report states the finding is NOT there, or states that the
                relevant structure is normal/intact
  "uncertain" — the report does not let you decide

THE 12 FINDINGS
{LABEL_DEFINITIONS}

RULES — read all of them, they are where mistakes happen.

1. NEGATION IS AN ANSWER, NOT A GAP. "No meniscal tear", "yırtık izlenmedi",
   "keine Fraktur", "sin derrame", "ligaments intact", "unauffällig",
   "regelrecht" → these mean "absent", NOT "uncertain". Getting negation right
   matters as much as getting positives right.

2. NOT MENTIONED depends on how complete the exam report is. First decide
   "exam_completeness":
     "comprehensive" — the report systematically walks through the knee
                       (ligaments, menisci, cartilage, fluid, bone) or ends with
                       a global normal statement ("Normal MRI of the knee")
     "partial"       — short, telegraphic, or clearly only reporting the
                       abnormality found
   Then, for a finding that is NOT mentioned at all:
     - comprehensive → "absent", confidence 0.70-0.90 (a thorough radiologist
       who saw it would have written it)
     - partial       → "uncertain", confidence 0.15-0.35 (silence tells you nothing)

3. COMPARTMENTS ARE SEPARATE. "Medial OA", "Lateral OA" and "PF OA" are three
   different findings. "Retropatellar chondropathy" is PF OA and says nothing
   about the tibiofemoral compartments. A generic "gonarthrosis" or
   "degenerative changes" without a named compartment → mark the compartments
   "uncertain" (0.4-0.5), do not spread it across all three as "present".

4. DISTINGUISH THESE PAIRS:
   - Effusion (fluid) vs Synovitis (inflamed synovium). Effusion alone is NOT
     synovitis.
   - Contusion (marrow oedema, no fracture line) vs Fracture (a fracture line).
     A report describing oedema AND a fracture line → both "present".

5. DEGENERATION AND OEDEMA ARE NOT INJURY. This applies to BOTH menisci AND
   ligaments, and it is the single most common mistake on these reports:
     - meniscus: "grade I/II degeneration", "intrasubstance signal", "linear
       signal without surfacing" → NOT a tear
     - ligament (ACL/MCL): "signal increase", "myxoid degeneration", "thinning",
       "periligamentous oedema", "fluid around the ligament", "laxity" → NOT a
       tear unless the report says tear / rupture / discontinuity
   Mark these "uncertain" (0.3-0.5), not "present". Only an explicit tear,
   rupture, discontinuity or named tear grade counts as "present".

6. NAMED SIDE BEATS THE HEADING. If the sentence says "lateral collateral
   ligament", that is NOT MCL, even in a report that mostly discusses the medial
   side. Read the structure actually named in the sentence you are quoting.

{RULE_MINIMAL}8. DO NOT INFER ACROSS FINDINGS. An ACL tear does not make an effusion
   "present". Judge each finding only on what the report says about it.

9. "confidence" is 0.0-1.0 and reflects how sure you are about the STATUS you
   chose, not how severe the finding is.

10. "evidence" is a SHORT verbatim quote from the report, in the report's own
   language, that justifies the status. Leave it "" when nothing in the text
   speaks to this finding. Never invent a quote.

OUTPUT
Return only a JSON object, no prose before or after:

{{"exam_completeness": "comprehensive" | "partial",
  "findings": {{"<finding name>": {{"status": "...", "confidence": 0.0, "evidence": "..."}}, ...}}}}

All 12 finding names must be present, spelled exactly as listed above."""


def build_system_prompt(with_minimal_rule: bool = True) -> str:
    """Sistem prompt'unu uret.

    `with_minimal_rule=False` 7. kurali ("minimal/trace/mild -> uncertain")
    tamamen cikarir. Kural numaralarinda bosluk olusur (6'dan 8'e atlar) ama bu
    onemsiz — modelin numaralara degil iceriklerine bakmasi gerekiyor ve
    ablationi kural metinlerini yeniden numaralandirmadan yapmak, iki varyant
    arasindaki TEK farkin o kural olmasini garanti ediyor.
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        LABEL_DEFINITIONS=LABEL_DEFINITIONS,
        RULE_MINIMAL=RULE_MINIMAL if with_minimal_rule else "")


SYSTEM_PROMPT = build_system_prompt(True)


def build_user_message(report: str) -> str:
    return f"REPORT:\n{report.strip()}\n\nExtract the 12 findings as JSON."


def build_few_shot_messages() -> list:
    """Few-shot ornekleri messages dizisine cevir (user/assistant ciftleri)."""
    msgs = []
    for ex in FEW_SHOT:
        msgs.append({"role": "user", "content": build_user_message(ex["report"])})
        msgs.append({"role": "assistant",
                     "content": json.dumps(ex["output"], ensure_ascii=False)})
    return msgs


# --- JSON semasi: structured output destekleyen saglayicilar icin ------------
def output_schema() -> dict:
    finding = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["present", "absent", "uncertain"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "evidence": {"type": "string"},
        },
        "required": ["status", "confidence", "evidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "exam_completeness": {"type": "string",
                                  "enum": ["comprehensive", "partial"]},
            "findings": {
                "type": "object",
                "properties": {k: finding for k in LABELS},
                "required": list(LABELS),
                "additionalProperties": False,
            },
        },
        "required": ["exam_completeness", "findings"],
        "additionalProperties": False,
    }


# --- Ayristirma -------------------------------------------------------------
_STATUS_TO_SOFT = {"present": 1.0, "absent": 0.0, "uncertain": None}


def _extract_json(text: str):
    """Modelin cevabindan JSON nesnesini cikar.

    Structured output kullanildiginda gereksiz, ama her saglayici desteklemiyor
    ve model bazen ```json blogu veya aciklama metni ekliyor. Once duz parse,
    sonra kod blogu, sonra ilk dengeli suslu parantez blogu.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def parse_response(text: str, study_uid: str = None) -> dict:
    """Model cevabini duz bir sozluge cevir.

    Dondurulen sozlukte her etiket icin uc alan olur:
      <etiket>            soft hedef (1.0 / 0.0 / None)  -> egitim hedefi
      <etiket>_status     present/absent/uncertain/missing
      <etiket>_conf       0-1
      <etiket>_evidence   dayanak alinti
    Ayrica: parse_ok, exam_completeness, n_missing, raw_error
    """
    out = {"StudyInstanceUID": study_uid, "parse_ok": False,
           "exam_completeness": None, "n_missing": len(LABELS), "raw_error": None}
    data = _extract_json(text)
    if not isinstance(data, dict):
        out["raw_error"] = "JSON cikarilamadi"
        for k in LABELS:
            out[k], out[f"{k}_status"] = None, "missing"
            out[f"{k}_conf"], out[f"{k}_evidence"] = 0.0, ""
        return out

    out["exam_completeness"] = data.get("exam_completeness")
    findings = data.get("findings") or {}
    missing = 0
    for k in LABELS:
        f = findings.get(k)
        if not isinstance(f, dict):
            out[k], out[f"{k}_status"] = None, "missing"
            out[f"{k}_conf"], out[f"{k}_evidence"] = 0.0, ""
            missing += 1
            continue
        st = str(f.get("status", "")).strip().lower()
        if st not in _STATUS_TO_SOFT:
            st = "uncertain"
        try:
            conf = float(f.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        out[k] = _STATUS_TO_SOFT[st]
        out[f"{k}_status"] = st
        out[f"{k}_conf"] = min(max(conf, 0.0), 1.0)
        out[f"{k}_evidence"] = str(f.get("evidence", ""))[:300]
    out["n_missing"] = missing
    out["parse_ok"] = missing == 0
    return out


# --- Soft label uretimi -----------------------------------------------------
# OLCULDU (Faz 1.8, Qwen2.5-32B-AWQ, 20 gold study):
#     sert 0/1 (uncertain=0.5)   macro AUC = 0.8323
#     0.5 +/- 0.5*conf           macro AUC = 0.8655   <- +0.033, bedava
#
# Mekanizma: sert etiketlerde butun "present"ler 1.0'da esitlenir ve AUC
# esitlikleri 0.5 sayar. Guven skoru o esitlikleri kirar; guven dogrulukla
# korele oldugu icin (AUC_guven 0.819) kirilma dogru yone gider.
#
# Bu, over-calling'i prompt'la kovalamak yerine SIRALAMAYA birakmak demek —
# yarismanin metrigi zaten macro ROC-AUC, mutlak kalibrasyon degil.
SOFT_UNCERTAIN = 0.5


def soft_label(status, confidence, uncertain_value: float = SOFT_UNCERTAIN):
    """(status, confidence) -> [0,1] araliginda soft hedef.

    present  -> 0.5 + 0.5*conf   (guven arttikca 1'e yaklasir)
    absent   -> 0.5 - 0.5*conf   (guven arttikca 0'a yaklasir)
    digeri   -> uncertain_value  (None verilirse loss'ta maskelenir)

    Boylece "present ama emin degil" ile "present ve emin" ayrilir; sert
    esiklemede ikisi de 1.0 olurdu.
    """
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        c = 0.0
    c = min(max(c, 0.0), 1.0)
    if status == "present":
        return 0.5 + 0.5 * c
    if status == "absent":
        return 0.5 - 0.5 * c
    return uncertain_value


def add_soft_labels(df, conf_suffix: str = "_conf", uncertain_value: float = SOFT_UNCERTAIN):
    """Her etiket icin `<etiket>_soft` sutunu ekle (df yerinde degistirilir).

    `conf_suffix` neden parametre: bake-off'ta modelin kendi bildirdigi guven
    (`_conf`, AUC_guven 0.819) logprob'dan turetilenden (`_lpconf`, 0.679) daha
    iyi cikti, ama bu olcum modele bagli — degistirilebilir olsun.
    """
    for k in LABELS:
        st = df.get(f"{k}_status")
        cf = df.get(f"{k}{conf_suffix}")
        if st is None:
            continue
        if cf is None:
            cf = [None] * len(df)
        df[f"{k}_soft"] = [soft_label(s, c, uncertain_value) for s, c in zip(st, cf)]
    return df
