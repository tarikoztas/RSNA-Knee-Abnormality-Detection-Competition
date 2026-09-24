"""Katman 1 — acik agirlikli LLM ile rapor cikarimi (vLLM).

Neden vLLM, neden Ollama degil:
  Ollama etkilesimli tek-istek kullanimi icin tasarlandi. vLLM'in continuous
  batching'i onlarca istegi ayni anda GPU'da tutar — ayni donanimda 10-30 kat
  throughput farki. Ayrica vLLM JSON sema gudumlu decoding destekliyor: semaya
  uymayan token'lar decode sirasinda maskelenir, yani model GECERSIZ JSON
  URETEMEZ. 7B bir modelden 12 alanli JSON isterken bu belirleyici.

Tasarim notlari:
  * Guven skoru MODELE SORULMUYOR (kucuk modeller her seye 0.9 der); status
    token'inin logprob'undan turetiliyor. Ikisi de kaydediliyor, hangisinin
    dogrulukla daha iyi korele ettigi dev set'te olculecek (Katman 4 girdisi).
  * Her cevap aninda JSONL'e yazilir -> Kaggle oturumu kesilirse kaldigi yerden
    devam eder. 4.400 raporluk bir kosuyu bastan almak istemiyoruz.
  * vLLM'in gudumlu decoding API'si surumler arasi degisti; uc bilinen sekli
    sirayla denenip calisan kullanilir, hicbiri yoksa gudumsuz moda dusulur
    (ayristiricimiz zaten dayanikli).
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt as P

STATUS_WORDS = ("present", "absent", "uncertain")


# --------------------------------------------------------------------------
# Gudumlu decoding: surum tespiti
# --------------------------------------------------------------------------
def make_sampling_params(schema, max_tokens=700, temperature=0.0, logprobs=None):
    """(SamplingParams, kullanilan_yol) dondur.

    vLLM surum farklari (ilk bake-off'ta hepsi basarisiz oldu -> gudumsuz kaldi,
    parse_ok 0.918'de takildi ve 4 rapor tamamen kayboldu):
      V1 engine   SamplingParams(structured_outputs=StructuredOutputsParams(json=...))
      >=0.6.x     SamplingParams(guided_decoding=GuidedDecodingParams(json=...))
      ~0.5.x      SamplingParams(..., guided_json=schema)
      eski        yok -> gudumsuz + dayanikli ayristirici

    Kullanicinin logunda "EngineCore" satirlari vardi, yani V1 engine calisiyor;
    o surumde alan adi `structured_outputs` olarak degisti. Ilk siraya o eklendi.
    """
    from vllm import SamplingParams

    base = dict(temperature=temperature, max_tokens=max_tokens, logprobs=logprobs)
    attempts = []

    # --- 1) vLLM V1: structured_outputs ---
    for modpath, clsname in [("vllm.sampling_params", "StructuredOutputsParams"),
                             ("vllm", "StructuredOutputsParams")]:
        try:
            mod = __import__(modpath, fromlist=[clsname])
            cls = getattr(mod, clsname)
            sp = SamplingParams(structured_outputs=cls(json=schema), **base)
            return sp, f"structured_outputs={clsname}(json=...)  [V1]"
        except Exception as e:
            attempts.append(f"{clsname}: {type(e).__name__}")

    # --- 2) guided_decoding=GuidedDecodingParams ---
    try:
        from vllm.sampling_params import GuidedDecodingParams
        sp = SamplingParams(guided_decoding=GuidedDecodingParams(json=schema), **base)
        return sp, "guided_decoding=GuidedDecodingParams(json=...)"
    except Exception as e:
        attempts.append(f"GuidedDecodingParams: {type(e).__name__}")

    # --- 3) duz guided_json ---
    try:
        sp = SamplingParams(guided_json=schema, **base)
        return sp, "guided_json=..."
    except Exception as e:
        attempts.append(f"guided_json: {type(e).__name__}")

    print("  [UYARI] GUDUMLU DECODING YOK -> gudumsuz mod.")
    print("          Denenen yollar: " + " | ".join(attempts))
    print("          Ayristirma hatasi orani yukselecek (ilk kosuda %8 idi).")
    return SamplingParams(**base), "GUDUMSUZ"


# --------------------------------------------------------------------------
# Prompt -> metin
# --------------------------------------------------------------------------
def build_chat_text(tokenizer, report: str, few_shot: bool = True,
                    system_prompt: str = None) -> str:
    """Sohbet sablonunu uygulayip tek bir metin uret.

    `enable_thinking=False` neden gerekli (Faz 1.8 bulgusu):
    Qwen3 ailesi varsayilan olarak thinking mode'da calisir ve cevaptan once
    <think>...</think> blogu yazar. Bake-off'ta Qwen3-14B-AWQ'nun parse_ok'u
    0.000 cikti — 700 token'lik butce dusunmeye gitti, JSON'a hic sira gelmedi.
    Bu parametre eski sablonlarda yok, o yuzden TypeError yakalanip atlaniyor.
    """
    msgs = [{"role": "system", "content": system_prompt or P.SYSTEM_PROMPT}]
    if few_shot:
        msgs += P.build_few_shot_messages()
    msgs.append({"role": "user", "content": P.build_user_message(report)})
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


# --------------------------------------------------------------------------
# Logprob -> guven
# --------------------------------------------------------------------------
def status_confidences(output) -> dict:
    """Uretilen token'lar arasinda status kelimelerini bul, logprob'dan guven turet.

    Dondurur: {"present": [olasiliklar], "absent": [...], "uncertain": [...]}
    Sira, JSON'da etiketlerin gecis sirasiyla ayni (sema 'required' sirasini korur),
    boylece i. status token'i i. etikete denk gelir.

    Neden basit bir tarama: gudumlu decoding JSON yapisini garanti ettigi icin
    status alanlarinin sirasi deterministik. Token'i tam konumlandirmak yerine
    "status kelimesine denk gelen token"i yakalamak yeterli ve surume dayanikli.
    """
    found = {w: [] for w in STATUS_WORDS}
    lps = getattr(output, "logprobs", None)
    if not lps:
        return found
    for step in lps:
        if not step:
            continue
        # Bu adimda secilen token (en yuksek logprob'a sahip olan degil, SECILEN)
        try:
            chosen = max(step.values(), key=lambda x: getattr(x, "rank", 99) * -1
                         if hasattr(x, "rank") else x.logprob)
        except Exception:
            continue
        tok = (getattr(chosen, "decoded_token", None) or "").strip().strip('"').lower()
        if tok in STATUS_WORDS:
            found[tok].append(float(math.exp(chosen.logprob)))
    return found


def attach_logprob_conf(rec: dict, output) -> dict:
    """parse_response ciktisina logprob tabanli guveni ekle (<etiket>_lpconf).

    TAMAMI korumali: logprob cikti formati vLLM surumleri arasinda degisiyor
    (0.29'da `flat_logprobs` / `logprob_token_ids` gibi yeni alanlar var). Bir
    format degisikliginin 13 saatlik kosuyu oldurmesine izin veremeyiz —
    cozulemezse `_lpconf` None kalir, baska hicbir sey etkilenmez.

    Zaten OLCULDU (Faz 1.8): modelin kendi bildirdigi guven (AUC 0.819)
    logprob'dan turetilenden (0.679) daha iyi. Yani bu alan artik yedek bilgi.
    """
    try:
        conf = status_confidences(output)
        queues = {w: list(v) for w, v in conf.items()}
    except Exception:
        queues = {}
    for k in P.LABELS:
        try:
            q = queues.get(rec.get(f"{k}_status"))
            rec[f"{k}_lpconf"] = float(q.pop(0)) if q else None
        except Exception:
            rec[f"{k}_lpconf"] = None
    return rec


# --------------------------------------------------------------------------
# Ana kosu
# --------------------------------------------------------------------------
def run(model_id, reports, out_path, tensor_parallel_size=1,
        max_model_len=4096, gpu_memory_utilization=0.90, quantization=None,
        few_shot=True, max_tokens=700, batch_log_every=200, dtype="auto",
        logprobs=None, system_prompt=None, tag=None):
    """reports: [(study_uid, report_text), ...]. Sonuclari out_path'e JSONL yazar.

    Zaten yazilmis study'leri atlar (resume). Dondurur: (kayitlar, meta).
    """
    from transformers import AutoTokenizer
    from vllm import LLM

    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["StudyInstanceUID"])
                except Exception:
                    pass
        print(f"  {len(done):,} rapor zaten islenmis, atlaniyor (resume)")

    todo = [(u, r) for u, r in reports if u not in done]
    if not todo:
        print("  yapilacak is yok")
        return [], {}

    print(f"Model yukleniyor: {model_id}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # ONEK CACHE'I — en buyuk hiz kaldiraci (Faz 1.8 bulgusu).
    # Sistem prompt'u + few-shot ornekleri ~2.180 token ve 4.407 istegin
    # HEPSINDE birebir ayni. Bu acik degilken vLLM o oneki her rapor icin
    # sifirdan yeniden hesapliyor: ~9.6 milyon gereksiz prefill token'i.
    # Ayni kitabin ilk 50 sayfasini her bolumde bastan okumak gibi.
    llm_kwargs = dict(model=model_id, tensor_parallel_size=tensor_parallel_size,
                      max_model_len=max_model_len, dtype=dtype,
                      gpu_memory_utilization=gpu_memory_utilization,
                      enable_prefix_caching=True,
                      trust_remote_code=True)
    if quantization:
        llm_kwargs["quantization"] = quantization
    try:
        llm = LLM(**llm_kwargs)
    except TypeError as e:
        # Eski vLLM surumlerinde parametre adi farkli olabilir
        print(f"  [bilgi] enable_prefix_caching kabul edilmedi ({e}); onek")
        print("          cache'i olmadan devam ediliyor — kosu YAVAS olacak.")
        llm_kwargs.pop("enable_prefix_caching")
        llm = LLM(**llm_kwargs)
    load_s = time.time() - t0
    print(f"  yuklendi ({load_s:.0f}s)")

    # logprobs neden varsayilan KAPALI: her token icin 5 aday dondurmek
    # 4.407 rapor x ~500 token x 5 = ~11M kayit serilestirmek demek ve olcumu
    # zaten yaptik — modelin kendi bildirdigi guven (AUC 0.819) logprob'dan
    # turetilenden (0.679) iyi cikti. Yeniden olcmek isteyen logprobs=5 verir.
    sp, guided_mode = make_sampling_params(P.output_schema(), max_tokens=max_tokens,
                                          logprobs=logprobs)
    print(f"  gudumlu decoding: {guided_mode}")
    print(f"  logprobs: {logprobs if logprobs else 'kapali (hiz icin)'}")

    prompts = [build_chat_text(tok, r, few_shot, system_prompt) for _, r in todo]

    # BUTCE KONTROLU — bu kontrol olmadigi icin Tur 3'te sessizce veri kaybettik.
    # vLLM girdi+cikti <= max_model_len olacak sekilde cikti butcesini KIRPAR ve
    # uyari vermez; JSON yarim kalir, ayristirma basarisiz olur, sebebi de
    # gorunmez. En uzun promptu olcup baştan haber veriyoruz.
    lens = [len(tok(pr).input_ids) for pr in prompts]
    n_in, n_max = sum(lens) / len(lens), max(lens)
    bos = max_model_len - n_max
    print(f"  girdi uzunlugu: ortalama {n_in:.0f}, EN UZUN {n_max} token")
    print(f"  max_model_len={max_model_len} -> en uzun promptta cikti icin"
          f" kalan: {bos} token  (istenen: {max_tokens})")
    if bos < max_tokens:
        kritik = sum(1 for L in lens if max_model_len - L < max_tokens)
        print("  " + "!" * 66)
        print(f"  !! BUTCE YETERSIZ: {kritik}/{len(lens)} promptta cikti butcesi")
        print(f"  !! {max_tokens} token'in ALTINA dusuyor. Bu raporlarda JSON")
        print(f"  !! yarim kalacak ve ayristirilamayacak.")
        print(f"  !! COZUM: max_model_len >= {n_max + max_tokens} yap.")
        print("  " + "!" * 66)
    else:
        print("  >> butce yeterli: tum promptlar tam cikti alabilir.")

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0

    recs = []
    n_out = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for (uid, _), o in zip(todo, outs):
            gen = o.outputs[0]
            n_out += len(gen.token_ids)
            rec = P.parse_response(gen.text, uid)
            rec = attach_logprob_conf(rec, gen)
            rec["n_out_tokens"] = len(gen.token_ids)
            rec["model"] = model_id
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            recs.append(rec)

    ok = sum(r["parse_ok"] for r in recs)
    # Kesilme teshisi: gudumlu decoding modeli 12 bulgunun HEPSINI doldurmaya
    # zorluyor, kisa yoldan cikamiyor. Butce biterse JSON yarim kalir ve
    # ayristirilamaz. Tur 3'te parse_ok gudumlu decoding ACIKKEN dustu — bu
    # sayac o hipotezi dogrudan test ediyor.
    n_capped = sum(1 for r in recs if r.get("n_out_tokens", 0) >= max_tokens)
    n_bad_capped = sum(1 for r in recs
                       if not r["parse_ok"] and r.get("n_out_tokens", 0) >= max_tokens)
    meta = {
        "model": model_id, "tag": tag or model_id, "n": len(recs),
        "parse_ok": ok,
        "parse_ok_rate": ok / max(len(recs), 1),
        "guided_mode": guided_mode, "load_s": round(load_s, 1),
        "gen_s": round(gen_s, 1), "out_tokens": n_out,
        "tok_per_s": round(n_out / max(gen_s, 1e-9), 1),
        "s_per_report": round(gen_s / max(len(recs), 1), 3),
        "mean_out_tokens": round(n_out / max(len(recs), 1), 1),
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "max_prompt_tokens": int(n_max),
        "budget_ok": bool(max_model_len - n_max >= max_tokens),
        "n_capped": n_capped,
        "n_fail_capped": n_bad_capped,
        "n_fail": len(recs) - ok,
    }
    # Teshis DOSYAYA da yazilir: vLLM binlerce INFO satiri basiyor ve Kaggle
    # cikti limitini asinca kirpiyor — ilk kosuda "gudumlu decoding" satiri tam
    # bu yuzden kayboldu ve gudumsuz calistigini fark etmedik.
    try:
        diag = os.path.join(os.path.dirname(out_path) or ".", "run_diagnostics.jsonl")
        with open(diag, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
    except Exception:
        pass
    print(f"  {len(recs):,} rapor / {gen_s:.0f}s  "
          f"({meta['s_per_report']:.2f} s/rapor, {meta['tok_per_s']:.0f} tok/s)")
    print(f"  ayristirma basarisi: {ok}/{len(recs)} ({meta['parse_ok_rate']:.1%})")
    print(f"  token tavanina dayanan: {n_capped}  |  bunlardan ayristirilamayan:"
          f" {n_bad_capped} / {len(recs)-ok}")
    est_h = meta["s_per_report"] * 4407 / 3600
    print(f"  >> 4.407 raporun tamami icin tahmin: {est_h:.2f} saat")
    return recs, meta
