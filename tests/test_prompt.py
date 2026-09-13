"""src/labels/prompt.py ayristiricisini test et.

Neden onemli: model cikisi her zaman temiz JSON olmayacak — kod blogu, aciklama
metni, eksik alan, gecersiz status. 4.400 raporluk bir kosuda bunlardan biri
istisna atarsa kosu coper. Ayristirici asla patlamamali, "missing" demeli.

Kullanim: python tests/test_prompt.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src" / "labels"))
import prompt as P

FAILS = []


def check(name, cond, extra=None):
    print(("  OK   " if cond else "  FAIL ") + name +
          ("" if extra is None else "  " + str(extra)))
    if not cond:
        FAILS.append(name)


def full_output(**over):
    f = {k: {"status": "uncertain", "confidence": 0.2, "evidence": ""} for k in P.LABELS}
    f.update(over)
    return {"exam_completeness": "partial", "findings": f}


JS = json.dumps(full_output(
    ACL={"status": "present", "confidence": 0.95, "evidence": "ACL rupture"},
    Effusion={"status": "absent", "confidence": 0.9, "evidence": "no effusion"},
), ensure_ascii=False)

print("--- temiz JSON ---")
r = P.parse_response(JS, "s1")
check("parse_ok", r["parse_ok"])
check("StudyInstanceUID tasindi", r["StudyInstanceUID"] == "s1")
check("present -> 1.0", r["ACL"] == 1.0)
check("absent -> 0.0", r["Effusion"] == 0.0)
check("uncertain -> None (loss'ta maskelenecek)", r["MCL"] is None)
check("evidence tasindi", r["ACL_evidence"] == "ACL rupture")
check("exam_completeness tasindi", r["exam_completeness"] == "partial")

print("--- model cevabini suslemisse ---")
check("kod blogu icinde", P.parse_response("Here:\n```json\n" + JS + "\n```")["parse_ok"])
check("onde/arkada metin", P.parse_response("Sure! " + JS + " Hope that helps.")["parse_ok"])

print("--- bozuk girdi asla patlamamali ---")
for name, bad in [("bos string", ""), ("None", None), ("duz metin", "I cannot help."),
                  ("kirik JSON", '{"findings": {"ACL": {'),
                  ("JSON dizisi", "[1,2,3]"), ("bos nesne", "{}")]:
    r = P.parse_response(bad)
    check(name + " -> parse_ok False", r["parse_ok"] is False)
    check(name + " -> 12 etiket missing", r["n_missing"] == 12, r["n_missing"])

print("--- eksik ve gecersiz alanlar ---")
part = {"exam_completeness": "comprehensive",
        "findings": {k: {"status": "absent", "confidence": 0.9, "evidence": ""}
                     for k in P.LABELS[:10]}}
r = P.parse_response(json.dumps(part))
check("eksik etiket sayisi 2", r["n_missing"] == 2, r["n_missing"])
check("eksik etiket None", r[P.LABELS[11]] is None)
check("kismi cikti parse_ok False", r["parse_ok"] is False)

r = P.parse_response(json.dumps(full_output(
    ACL={"status": "PRESENT ", "confidence": "1.7", "evidence": "x" * 500})))
check("status normalize (buyuk harf + bosluk)", r["ACL_status"] == "present")
check("confidence [0,1] araligina kirpildi", r["ACL_conf"] == 1.0)
check("evidence 300 karaktere kirpildi", len(r["ACL_evidence"]) == 300)

r = P.parse_response(json.dumps(full_output(
    ACL={"status": "maybe", "confidence": 0.5, "evidence": ""})))
check("bilinmeyen status -> uncertain", r["ACL_status"] == "uncertain")

r = P.parse_response(json.dumps(full_output(ACL="bu bir sozluk degil")))
check("etiket degeri sozluk degilse missing", r["ACL_status"] == "missing")

print("--- prompt saglamasi ---")
check("negation kurali prompt'ta", "NEGATION IS AN ANSWER" in P.SYSTEM_PROMPT)
check("kompartman kurali prompt'ta", "COMPARTMENTS ARE SEPARATE" in P.SYSTEM_PROMPT)
check("exam_completeness kurali prompt'ta", "exam_completeness" in P.SYSTEM_PROMPT)
check("etiketler arasi cikarim yasagi", "DO NOT INFER ACROSS FINDINGS" in P.SYSTEM_PROMPT)
check("12 etiket tanimi", all(k in P.LABEL_DEFINITIONS for k in
                              ["ACL", "MCL", "PF OA", "Baker's", "Contusion"]))
check("few-shot 3 ornek", len(P.FEW_SHOT) == 3)
check("few-shot user/assistant cifti", len(P.build_few_shot_messages()) == 6)
for i, ex in enumerate(P.FEW_SHOT):
    check("few-shot " + str(i) + " kendi ayristiricisindan geciyor",
          P.parse_response(json.dumps(ex["output"]))["parse_ok"])
check("sema 12 etiketi zorunlu kiliyor",
      len(P.output_schema()["properties"]["findings"]["required"]) == 12)
check("sema status enum'u uc degerli",
      len(P.output_schema()["properties"]["findings"]["properties"]["ACL"]
          ["properties"]["status"]["enum"]) == 3)

print()
print("=" * 60)
if FAILS:
    print("BASARISIZ (" + str(len(FAILS)) + "):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("TUM PROMPT/PARSER TESTLERI GECTI")
