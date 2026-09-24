# %% [markdown]
# # Faz 1.8 — Katman 1: Açık Ağırlıklı LLM ile Rapor → Etiket
#
# Bu notebook **iki modda** çalışır:
#
# | Mod | Ne yapar | Süre |
# |---|---|---|
# | `MODE = "bakeoff"` | Aday modelleri 49 raporluk dev set'te yarıştırır, 20 gold'da per-label F1/AUC ölçer | ~30-60 dk |
# | `MODE = "full"` | Kazanan modeli 4.407 raporun tamamına uygular | ölçülecek |
#
# **Önce `bakeoff` çalıştır.** Hangi modelin yeterli olduğunu tahmin etmek yerine
# ölçüyoruz: dev set'in 20'si gold etiketli, yani elle hiçbir şey işaretlemeden
# anında yer gerçeği var.
#
# ## Üç tasarım kararı
#
# **1. vLLM, Ollama değil.** Ollama etkileşimli tek-istek kullanımı için tasarlandı.
# vLLM'in *continuous batching*'i onlarca isteği aynı anda GPU'da tutar — aynı
# donanımda 10-30 kat throughput farkı. Ayrıca vLLM **JSON şema güdümlü decoding**
# destekliyor: şemaya uymayan token'lar decode sırasında maskelenir, yani model
# geçersiz JSON *üretemez*. 7B bir modelden 12 alanlı JSON isterken belirleyici.
#
# **2. Güven skoru modele sorulmuyor.** Küçük modeller her şeye 0.9 der. Bunun
# yerine `status` token'ının logprob'undan türetiliyor. İkisi de kaydediliyor;
# hangisi doğrulukla daha iyi korele ediyorsa Katman 4'te o kullanılacak.
#
# **3. `transformers` yedeği var.** Kaggle'da `pip install vllm` torch sürüm
# çakışması yüzünden kırılabiliyor. Kırılırsa notebook `transformers` ile devam
# eder — yavaş ama 49 rapor için yeterli, yani bake-off yine yapılır.
#
# ## Runtime ayarları
#
# | Ayar | Değer |
# |---|---|
# | Accelerator | **GPU** (T4×2 varsa onu seç — 32 GB, büyük model için şart) |
# | Internet | **Açık** (model indirilecek) |
# | Persistence | No persistence |

# %% [markdown]
# ## 0. vLLM — EN BAŞTA, torch import edilmeden önce
#
# **Sıra burada kritik.** `pip install vllm` kendi torch sürümünü kurar. Eğer torch
# bu hücreden önce import edilmişse, kurulum diskteki torch'u değiştirir ama
# bellekte eski sürüm yüklü kalır → `import vllm` `ImportError` verir. Motor
# çalışırken yağ değiştirmeye benziyor.
#
# Bu yüzden kurulum **ilk kod hücresi** ve buradan önce hiçbir şey import
# edilmiyor. Kurulum yeni yapıldıysa hücre sana kernel'i yeniden başlatmanı
# söyleyecek — o zaman **Run → Restart & Run All**. İkinci turda vllm zaten
# kurulu olacağı için kurulum atlanır ve import temiz çalışır.

# %%
import importlib.util
import subprocess
import sys


def vllm_status():
    """('yok'|'hazir'|'kirik', bilgi) — hata MESAJINI da dondurur, sadece tipini degil."""
    if importlib.util.find_spec("vllm") is None:
        return "yok", None
    try:
        import vllm
        return "hazir", vllm.__version__
    except Exception as e:
        return "kirik", f"{type(e).__name__}: {e}"


TORCH_ALREADY_LOADED = "torch" in sys.modules
state, info = vllm_status()
print(f"vLLM durumu: {state}" + (f"  ({info})" if info else ""))
if TORCH_ALREADY_LOADED:
    print("  !! torch bu hucreden ONCE import edilmis — kurulum gerekirse")
    print("     kernel yeniden baslatilmali.")

if state == "yok":
    print()
    print("Kuruluyor (5-15 dk, torch'u da indirebilir)...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "vllm"],
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print("  pip HATA (son 2000 karakter):")
        print((r.stderr or r.stdout)[-2000:])
    else:
        print("  pip tamam")
    state, info = vllm_status()
    print(f"vLLM durumu (kurulumdan sonra): {state}" + (f"  ({info})" if info else ""))

HAS_VLLM = state == "hazir"

if state == "kirik":
    print()
    print("!" * 72)
    print("vLLM kurulu ama import edilemiyor. TAM HATA:")
    print("  " + str(info))
    print()
    low = str(info).lower()
    if "torch" in low or "libc" in low or "symbol" in low or "cuda" in low:
        print("Bu bir SURUM UYUSMAZLIGI. Cozum:")
        print("  Run -> Restart & Run All   (kernel yeniden baslasin)")
        print("Yeniden baslattiktan sonra vllm zaten kurulu olacak, kurulum")
        print("atlanacak ve import temiz calisacak.")
    else:
        print("Surum uyusmazligina benzemiyor. Bu mesaji bana gonder —")
        print("kurulum satirini buna gore degistirecegiz.")
    print("!" * 72)
elif not HAS_VLLM:
    print()
    print("vLLM kurulamadi. transformers yoluna dusulecek:")
    print("  - 49 raporluk bake-off YAPILABILIR (yavas ama olur)")
    print("  - 4.407 raporluk tam kosu icin cok yavas — once vLLM cozulmeli")

# %% [markdown]
# ## 0b. Güdümlü decoding sondası — model yüklemeden, 10 saniyede
#
# İlk bake-off'ta güdümlü decoding **sessizce çalışmadı** ve bunu ancak 3 model
# koştuktan ve ham çıktıları analiz ettikten sonra anladık. Bedeli: 4 rapor
# tamamen kayboldu (%8), hepsi İngilizce olmayan.
#
# Bu sonda model yüklemiyor, sadece `SamplingParams`'ın hangi alanı kabul
# ettiğine bakıyor. **Buradaki cevap `HICBIRI CALISMIYOR` ise bake-off'u
# çalıştırma** — çıktıyı gönder, sürüme göre düzeltelim.

# %%
#!writefile tools/vllm_probe.py /kaggle/working/vllm_probe.py

# %%
if HAS_VLLM:
    # Kendi isim alaninda calistir: Jupyter hucresinde __name__ == "__main__"
    # oldugu icin exec, dosyanin sonundaki `if __name__ == "__main__": probe()`
    # satirini de tetikliyor ve cikti IKI KEZ basiliyordu.
    _ns = {"__name__": "vllm_probe"}
    exec(open("/kaggle/working/vllm_probe.py").read(), _ns)
    _ns["probe"]()
else:
    print("vLLM yok — sonda atlandi.")

# %% [markdown]
# ## 1. Yapılandırma

# %%
MODE = "retry"            # "retry" | "ablation" | "bakeoff" | "full"

# Aday modeller: kucukten buyuge. VRAM'e sigmayanlar otomatik atlanir.
# `vram_gb` = kabaca gereken bos VRAM (agirliklar + KV cache + aktivasyon).
# Bake-off 1 sonucuna gore guncellendi:
#  - fp16 modeller tp=2'ye alindi (tek 15 GB T4'e sigmiyorlar, OOM verdiler)
#  - 14B-AWQ tp=2'ye alindi: tp=1'de 32B'den YAVAS cikti (15.2 vs 10.6 s/rapor)
#  - Qwen3 modelleri kaldi ama artik enable_thinking=False ile calisacaklar
CANDIDATES = [
    {"id": "Qwen/Qwen2.5-14B-Instruct-AWQ", "vram_gb": 12, "tp": 2, "quant": "awq"},
    {"id": "Qwen/Qwen2.5-32B-Instruct-AWQ", "vram_gb": 24, "tp": 2, "quant": "awq"},
]

# --- ABLATION (Tur 3 sonucu) -------------------------------------------------
# 14B kazandi: F1_absent 0.6830 vs 32B 0.6805 (n=20, fark gurultu) ama
# 6.42 saat vs 12.41 — tek oturuma sigan tek model.
ABLATION_MODEL = {"id": "Qwen/Qwen2.5-14B-Instruct-AWQ", "vram_gb": 12,
                  "tp": 2, "quant": "awq"}

# Tur 1 -> Tur 3'te DORT sey birden degisti, o yuzden F1_absent dususunu
# (0.7251 -> 0.6805) prompt'a mi decoding'e mi atfedecegimizi bilmiyoruz.
# 14B artik hizli oldugu icin ayiklamak ~12 dakika. Tek degisken: 7. kural.
ABLATION_VARIANTS = [
    ("minimal_VAR", True),     # mevcut prompt
    ("minimal_YOK", False),    # "minimal/trace/mild -> uncertain" kurali cikarildi
]

# Tam kosuda kullanilacak model — bake-off sonucuna gore ELLE doldur.
# Bake-off kazanani: 14B, F1_absent 0.6830 (32B 0.6805) ve YARI surede.
FULL_RUN_MODEL = {"id": "Qwen/Qwen2.5-14B-Instruct-AWQ", "vram_gb": 12,
                  "tp": 2, "quant": "awq"}

# 4096 -> 8192. ONCEKI DEGER HATALIYDI ve olcum bunu acikca gosterdi:
#   prompt oneki  ~3.355 token (sistem 5.720 + few-shot 4.344 karakter)
#   4096 - 3355   =  ~740 token, rapor VE cikti icin TOPLAM
# Yani 1.000 karakterden uzun her rapor cikti butcesini 700'un altina itiyordu.
# Olculen sonuc (14B, Tur 3):
#   rapor  0-800 kar -> parse_ok 1.000      1200-1600 -> 0.727
#        800-1200    -> 0.867               2500-5000 -> 0.000
# Bu kesilme degil BUTCE SIKISMASI; vLLM girdi+cikti <= max_model_len olacak
# sekilde cikti butcesini kirpiyor ve JSON yarim kaliyor.
# 8192 ile: onek 3.355 + en uzun rapor ~1.581 + cikti 1.100 = ~6.036, rahat pay var.
MAX_MODEL_LEN = 8192
# 700 -> 1100: gudumlu decoding modeli 12 bulgunun HEPSINI doldurmaya zorluyor,
# kisa yoldan cikamiyor. Tur 3'te ortalama cikti 463-479 token'di ve parse_ok
# gudumlu decoding ACIKKEN dustu (0.918 -> 0.816/0.857) — kesilme suphesi.
# Kosu `n_capped` sayacini basiyor, hipotezi dogrudan test edecek.
# 1100 -> 1400: tam kosuda 53 rapor (%1.2) tavana dayanip kesildi ve
# ayristirilamadi (n_fail_capped == n_fail == 53, yani TEK sebep buydu).
# En uzun prompt 5.700 token, 5700+1400=7100 < 8192 — hala rahat.
MAX_NEW_TOKENS = 1400
OUT_DIR = "/kaggle/working"

# Token basina 5 aday logprob dondurmek buyuk bir serilestirme yuku ve guven
# olcumu zaten yapildi (kendi bildirdigi 0.819 > logprob 0.679). Yeniden olcmek
# istersen 5 yap — o zaman <etiket>_lpconf sutunlari dolar.
LOGPROBS = None

# %% [markdown]
# ## 2. Modülleri diske yaz
#
# Üç modül gömülü geliyor — repo'daki sürümleri tek doğru sürüm, notebook onları
# `py2ipynb.py` dönüşümünde gömer, yani elle düzenlenmez ve bayatlamaz.

# %%
#!writefile src/labels/prompt.py /kaggle/working/prompt.py

# %%
#!writefile src/labels/evaluate.py /kaggle/working/evaluate.py

# %%
#!writefile src/labels/devset_ids.py /kaggle/working/devset_ids.py

# %%
#!writefile src/labels/retry_ids.py /kaggle/working/retry_ids.py

# %%
#!writefile src/labels/vllm_runner.py /kaggle/working/vllm_runner.py

# %% [markdown]
# ## 3. Ortam kontrolü
#
# GPU ve VRAM'i tespit ediyoruz — hangi adayların sığdığını bu belirliyor.

# %%
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/kaggle/working")

import numpy as np
import pandas as pd
import torch

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

print("torch      :", torch.__version__)
print("CUDA var mi:", torch.cuda.is_available())
N_GPU = torch.cuda.device_count()
print("GPU sayisi :", N_GPU)
VRAM_GB = 0.0
VRAM_PER_GPU = 0.0
for i in range(N_GPU):
    p = torch.cuda.get_device_properties(i)
    gb = p.total_memory / 1024**3
    VRAM_GB += gb
    VRAM_PER_GPU = gb if i == 0 else min(VRAM_PER_GPU, gb)
    print(f"  GPU {i}: {p.name}  {gb:.1f} GB")
print(f"TOPLAM VRAM : {VRAM_GB:.1f} GB")
print(f"GPU BASINA  : {VRAM_PER_GPU:.1f} GB   <- bir model ancak tp x bu kadarini kullanir")

if N_GPU == 0:
    raise RuntimeError("GPU yok. Sag panel -> Accelerator -> GPU secin.")

# %% [markdown]
# ## 4. Veri: dev set
#
# UID listeleri koda gömülü (`devset_ids.py`), ayrı bir Kaggle Dataset gerekmiyor.

# %%
import prompt as P
import devset_ids as D
import evaluate as EV

def find_train_csv(root="/kaggle/input", depth=2):
    """train.csv'yi iki seviyeye kadar ara. Bulamazsa NE MOUNT EDILMIS oldugunu yaz.

    Onceki surum sadece tek seviye ariyordu ve bulamayinca sadece "bulunamadi"
    diyordu — hangi dataset'lerin bagli oldugunu gostermek teshisi cok kolaylastirir.
    """
    if not os.path.isdir(root):
        return None, []
    seen = []
    for a in sorted(os.listdir(root)):
        pa = os.path.join(root, a)
        if not os.path.isdir(pa):
            continue
        seen.append(a)
        if os.path.exists(os.path.join(pa, "train.csv")):
            return pa, seen
        if depth > 1:
            for b in sorted(os.listdir(pa)):
                pb = os.path.join(pa, b)
                if os.path.isdir(pb) and os.path.exists(os.path.join(pb, "train.csv")):
                    return pb, seen
    return None, seen


DATA, mounted = find_train_csv()
if DATA is None:
    print("Mount edilmis dataset'ler:", mounted if mounted else "(HICBIRI)")
    print()
    print("  Sag panel -> '+ Add Input' -> Competitions sekmesi ->")
    print("  'rsna-knee-abnormality-detection' -> Add")
    print()
    print("  NOT: Import edilen notebook input'lari BERABERINDE GETIRMEZ.")
    print("  Her import sonrasi veriyi yeniden baglamak gerekiyor.")
    raise FileNotFoundError("train.csv bulunamadi — yukaridaki adimlari izleyin.")
print("VERI:", DATA)

train = pd.read_csv(f"{DATA}/train.csv")
train["Report"] = train["Report"].fillna("")
by_uid = train.set_index("StudyInstanceUID")

# HATA DUZELTMESI: eskiden `if MODE == "bakeoff"` ... `else: tum veri` yaziyordu.
# "ablation" modu else dalina dustu ve 49 rapor yerine 4.407 raporu isledi —
# 8.4 saatlik kazara bir tam kosu. (Sonuc iyi cikti ama niyet bu degildi.)
# Artik mod ADI ile eslestiriliyor, bilinmeyen mod hata veriyor.
if MODE == "retry":
    # Tam kosuda 1100 token tavanina dayanip kesilen 53 rapor. Sadece bunlari
    # yeniden islemek 4.407'yi bastan almaktan ~80 kat hizli.
    import retry_ids as R
    uids = [u for u in R.RETRY if u in by_uid.index]
elif MODE in ("bakeoff", "ablation"):
    uids = [u for u in D.DEVSET if u in by_uid.index]
elif MODE == "full":
    uids = train["StudyInstanceUID"].tolist()
else:
    raise ValueError(f"bilinmeyen MODE: {MODE!r}. "
                     'Gecerli: "retry" | "ablation" | "bakeoff" | "full"')
_ne = {"retry": "kesilen 53 rapor", "bakeoff": "49 raporluk dev set",
       "ablation": "49 raporluk dev set", "full": "TUM train seti"}
print(f"  (mod '{MODE}' -> {_ne[MODE]})")
reports = [(u, by_uid.loc[u, "Report"]) for u in uids]
print(f"MODE={MODE}  ->  {len(reports):,} rapor")

# Yer gercegi: dev set'in gold olanlari (bake-off degerlendirmesi bunlarda)
gold_cols = ["StudyInstanceUID"] + P.LABELS
gold_dev = train[train["StudyInstanceUID"].isin(D.GOLD_DEV)][gold_cols].copy()
print(f"gold_dev (olcum icin): {len(gold_dev)} study")
print(f"gold_holdout (DOKUNULMAYACAK): {len(D.GOLD_HOLDOUT)} study")

# %% [markdown]
# ## 5. Ön uçuş kontrolü — model var mı, sığıyor mu?
#
# İki eleme birden, **indirmeden önce**:
#
# 1. **Model gerçekten var mı?** Hugging Face'te depo adları değişiyor ve bazı
#    kuvantize varyantlar topluluk tarafından üretiliyor, resmi olmayabilir.
#    20 GB indirdikten sonra 404 almak istemiyoruz — `model_info` çağrısı
#    saniyeler sürer ve hiçbir şey indirmez.
# 2. **VRAM'e sığıyor mu?** `vram_gb` tahminleri kaba; sığmayan model zaten
#    hata verip atlanır ama baştan elemek oturum süresi kazandırır.
#
# Yerel bir yol (`/kaggle/input/...`) verilmişse HF kontrolü atlanır.

# %%
def model_exists(model_id):
    """(var_mi, aciklama). Hicbir sey indirmez."""
    if model_id.startswith("/"):
        return os.path.exists(model_id), "yerel yol"
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        n = len(getattr(info, "siblings", None) or [])
        return True, f"HF'de var ({n} dosya)"
    except Exception as e:
        return False, f"{type(e).__name__}"


print("ON UCUS KONTROLU")
print("-" * 72)
fits = []
for c in CANDIDATES:
    exists, note = model_exists(c["id"])
    # DUZELTME (bake-off 1 bulgusu): bir model tp GPU'ya yayilir, yani
    # kullanabilecegi VRAM = tp x GPU_BASINA_VRAM. Onceki surum toplam VRAM ile
    # karsilastiriyordu; bu yuzden Qwen2.5-7B (fp16, ~15 GB) tp=1 ile "sigar"
    # gorunup tek 15 GB'lik T4'te OOM verdi. Karsilastirma yanlis taraftaydi.
    usable = VRAM_PER_GPU * c["tp"]
    if not exists:
        verdict, why = "ATLA", f"BULUNAMADI ({note})"
    elif c["tp"] > N_GPU:
        verdict, why = "ATLA", f"{c['tp']} GPU gerekiyor, {N_GPU} var"
    elif c["vram_gb"] > usable:
        verdict, why = "ATLA", (f"VRAM yetmez ({c['vram_gb']} > {usable:.0f} = "
                                f"{c['tp']}x{VRAM_PER_GPU:.0f})")
    else:
        verdict, why = "OK  ", f"{note}, sigar ({c['vram_gb']}/{usable:.0f} GB)"
    print(f"  [{verdict}] {c['id']:<40} {why}")
    if verdict.strip() == "OK":
        fits.append(c)

print("-" * 72)
print(f"Yarisacak model: {len(fits)} / {len(CANDIDATES)}")
if not fits:
    raise RuntimeError(
        "Hicbir aday kullanilabilir degil.\n"
        "  - 'BULUNAMADI' ise: HF'de dogru depo adini bulup CANDIDATES'i guncelle\n"
        "    (ornek arama: huggingface.co/models?search=qwen2.5+instruct+awq)\n"
        "  - 'VRAM yetmez' ise: daha kucuk veya kuvantize bir model ekle\n"
        "  - Kaggle Models sekmesinden model ekleyip 'id' yerine\n"
        "    '/kaggle/input/<dataset>/<yol>' yazmak indirme suresini sifirlar")

# %% [markdown]
# ## 6. transformers yedeği
#
# vLLM yoksa bu kullanılır: güdümlü decoding yok, sadece batched `generate`.
# Ayrıştırıcımız dayanıklı olduğu için geçersiz JSON tamamen kaybedilmiyor,
# `missing` olarak işaretleniyor — ama ayrıştırma başarısı oranını izle.

# %%
def run_transformers(model_id, reports, out_path, quant=None, batch_size=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["StudyInstanceUID"])
                except Exception:
                    pass
    todo = [(u, r) for u, r in reports if u not in done]
    if not todo:
        return [], {}

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    load_s = time.time() - t0

    recs, n_out = [], 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            texts = []
            for _, r in chunk:
                msgs = ([{"role": "system", "content": P.SYSTEM_PROMPT}]
                        + P.build_few_shot_messages()
                        + [{"role": "user", "content": P.build_user_message(r)}])
                texts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                     add_generation_prompt=True))
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_MODEL_LEN).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for (uid, _), row in zip(chunk, gen):
                new = row[enc["input_ids"].shape[1]:]
                n_out += int((new != tok.pad_token_id).sum())
                rec = P.parse_response(tok.decode(new, skip_special_tokens=True), uid)
                rec["model"] = model_id
                for k in P.LABELS:
                    rec[f"{k}_lpconf"] = None      # bu yolda logprob toplanmiyor
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                recs.append(rec)
            print(f"    {min(i+batch_size, len(todo))}/{len(todo)}", end="\r")
    gen_s = time.time() - t0
    del model
    torch.cuda.empty_cache()

    ok = sum(r["parse_ok"] for r in recs)
    return recs, {"model": model_id, "n": len(recs), "parse_ok": ok,
                  "parse_ok_rate": ok / max(len(recs), 1),
                  "mean_out_tokens": round(n_out / max(len(recs), 1), 1),
                  "guided_mode": "yok (transformers)", "load_s": round(load_s, 1),
                  "gen_s": round(gen_s, 1), "out_tokens": int(n_out),
                  "tok_per_s": round(n_out / max(gen_s, 1e-9), 1),
                  "s_per_report": round(gen_s / max(len(recs), 1), 3)}

# %% [markdown]
# ## 6b. Ablation — tek değişken: "minimal → uncertain" kuralı
#
# Tur 3'te dört şey birden değişti (güdümlü decoding, önek cache'i, prompt
# kuralları, logprobs) ve `F1_absent` 0.7251 → 0.6805 geriledi. Hangisinin
# yaptığını bilmiyoruz. Bu bölüm **sadece 7. kuralı** açıp kapatıp ölçüyor;
# diğer her şey sabit.
#
# Aynı zamanda `MAX_NEW_TOKENS = 1100` ile kesilme hipotezini test ediyor:
# `n_capped` sayacı sıfıra yakınsa `parse_ok` düşüşünün sebebi kesilmeydi.

# %%
if MODE == "ablation":
    import vllm_runner as VR

    abl_rows, abl_ev = [], {}
    for vtag, with_rule in ABLATION_VARIANTS:
        print("=" * 72)
        print(f"VARYANT: {vtag}   (7. kural {'VAR' if with_rule else 'YOK'})")
        print("=" * 72)
        sys_prompt = P.build_system_prompt(with_minimal_rule=with_rule)
        print(f"  sistem prompt: {len(sys_prompt)} karakter")
        out_path = f"{OUT_DIR}/abl_{vtag}.jsonl"
        try:
            recs, meta = VR.run(
                ABLATION_MODEL["id"], reports, out_path,
                tensor_parallel_size=ABLATION_MODEL["tp"],
                max_model_len=MAX_MODEL_LEN, quantization=ABLATION_MODEL["quant"],
                max_tokens=MAX_NEW_TOKENS, logprobs=LOGPROBS,
                system_prompt=sys_prompt, tag=vtag)
        except Exception as e:
            print(f"  BASARISIZ: {type(e).__name__}: {str(e)[:300]}")
            abl_rows.append({"varyant": vtag, "durum": f"HATA: {type(e).__name__}"})
            continue
        if not recs:
            recs = [json.loads(l) for l in open(out_path, encoding="utf-8")]
        pred = pd.DataFrame(recs)

        row = {"varyant": vtag, "durum": "ok",
               "parse_ok": round(meta.get("parse_ok_rate", np.nan), 3),
               "tavana_dayanan": meta.get("n_capped"),
               "cikti_token": meta.get("mean_out_tokens"),
               "s_rapor": meta.get("s_per_report"),
               "tam_kosu_saat": round(meta.get("s_per_report", 0) * 4407 / 3600, 2)}
        for unc in ("mask", "absent", "half"):
            ev = EV.evaluate(pred, gold_dev, uncertain=unc)
            s = EV.summarise(ev)
            row[f"F1_{unc}"] = s["macro_F1"]
            row[f"AUC_{unc}"] = s["macro_AUC"]
            if unc == "mask":
                row["kapsam"] = s["ort_kapsam"]
                abl_ev[vtag] = ev
        # Guven kalitesi de varyanta gore degisebilir
        cq = EV.confidence_quality(pred, gold_dev, conf_suffix="_conf")
        row["AUC_guven"] = (round(float(cq["AUC_guven"].mean(skipna=True)), 3)
                            if len(cq) and cq["AUC_guven"].notna().any() else None)
        abl_rows.append(row)
        print()

    abl = pd.DataFrame(abl_rows)
    print("=" * 72)
    print("ABLATION SONUCLARI")
    print("=" * 72)
    print(abl.to_string(index=False))
    abl.to_csv(f"{OUT_DIR}/ablation_summary.csv", index=False)

    ok_rows = abl[abl["durum"] == "ok"] if "durum" in abl.columns else abl
    if len(ok_rows) == 2:
        a = ok_rows.iloc[0]
        b = ok_rows.iloc[1]
        print()
        print("--- KARAR ---")
        for k in ("F1_absent", "AUC_half", "kapsam", "parse_ok"):
            print(f"  {k:<12} {a['varyant']}={a[k]}   {b['varyant']}={b[k]}"
                  f"   fark={b[k]-a[k]:+.4f}")
        kazanan = a if a["F1_absent"] >= b["F1_absent"] else b
        print()
        print(f"  >> F1_absent'e gore kazanan: {kazanan['varyant']}")
        print(f"  >> Tur 1 (eski prompt, gudumsuz) referansi: F1_absent 0.7251")

# %% [markdown]
# ## 7. Bake-off
#
# Her model için: çalıştır → gold_dev'de değerlendir → tabloya ekle.
# Sonuçlar JSONL'e anında yazıldığı için oturum kesilirse kaldığı yerden devam eder.

# %%
if MODE == "bakeoff":
    import vllm_runner as VR

    results, all_ev = [], {}
    for c in fits:
        tag = c["id"].split("/")[-1]
        print("=" * 72)
        print(c["id"])
        print("=" * 72)
        out_path = f"{OUT_DIR}/raw_{tag}.jsonl"
        try:
            if HAS_VLLM:
                recs, meta = VR.run(
                    c["id"], reports, out_path,
                    tensor_parallel_size=c["tp"], max_model_len=MAX_MODEL_LEN,
                    quantization=c["quant"], max_tokens=MAX_NEW_TOKENS,
                    logprobs=LOGPROBS)
            else:
                recs, meta = run_transformers(c["id"], reports, out_path, c["quant"])
        except Exception as e:
            print(f"  BASARISIZ: {type(e).__name__}: {str(e)[:300]}")
            results.append({"model": tag, "durum": f"HATA: {type(e).__name__}"})
            continue

        if not recs:
            recs = [json.loads(l) for l in open(out_path, encoding="utf-8")]
        pred = pd.DataFrame(recs)

        # guided_mode TABLOYA girsin: ilk kosuda bu bilgi sadece ekrana basiliyordu
        # ve vLLM log seli icinde kayboldu — gudumsuz calistigini fark etmedik.
        row = {"model": tag, "durum": "ok",
               "gudumlu": meta.get("guided_mode", "?"),
               "parse_ok": round(meta.get("parse_ok_rate", np.nan), 3),
               "cikti_token": meta.get("mean_out_tokens"),
               "s_rapor": meta.get("s_per_report"),
               "tam_kosu_saat": round(meta.get("s_per_report", 0) * 4407 / 3600, 2)}
        for unc in ("mask", "absent", "half"):
            ev = EV.evaluate(pred, gold_dev, uncertain=unc)
            s = EV.summarise(ev)
            row[f"F1_{unc}"] = s["macro_F1"]
            row[f"AUC_{unc}"] = s["macro_AUC"]
            if unc == "mask":
                row["kapsam"] = s["ort_kapsam"]
                row["en_zayif"] = s["en_zayif"]
                all_ev[tag] = ev
        results.append(row)
        print()

    bake = pd.DataFrame(results)
    print("=" * 72)
    print("BAKE-OFF SONUCLARI")
    print("=" * 72)
    print(bake.to_string(index=False))
    bake.to_csv(f"{OUT_DIR}/bakeoff_summary.csv", index=False)

    # Bu kontrol ekrana ayrica basiliyor cunku log seli icinde kaybolmamasi lazim.
    if "gudumlu" in bake.columns:
        bad = bake[bake["gudumlu"].astype(str).str.contains("GUDUMSUZ|yok", na=False)]
        print()
        if len(bad):
            print("!" * 72)
            print("!! GUDUMLU DECODING CALISMADI:", list(bad["model"]))
            print("!! Model semaya uymak zorunda DEGIL -> parse_ok dusuk kalir.")
            print("!! Bu satiri bana gonder, vLLM surumune gore duzeltelim.")
            print("!" * 72)
        else:
            print(">> Gudumlu decoding TUM modellerde aktif:",
                  list(bake["gudumlu"].unique()))

# %% [markdown]
# ### 7b. Kazananın etiket bazında dökümü
#
# `macro_F1` tek bir sayı; hangi etiketin zayıf olduğunu görmek Katman 4'teki
# ağırlıklandırmayı belirleyecek.

# %%
if MODE == "bakeoff" and len(all_ev):
    ok = bake[bake["durum"] == "ok"]
    if len(ok):
        best = ok.loc[ok["F1_absent"].idxmax(), "model"]
        print(f"En iyi (F1_absent'e gore): {best}")
        print()
        print(all_ev[best].to_string(index=False))
        all_ev[best].to_csv(f"{OUT_DIR}/bakeoff_best_per_label.csv", index=False)

        # Guven skoru: modelin kendi bildirdigi mi, logprob mu daha iyi?
        recs = [json.loads(l) for l in
                open(f"{OUT_DIR}/raw_{best}.jsonl", encoding="utf-8")]
        pred = pd.DataFrame(recs)
        print()
        print("--- GUVEN SKORU KALITESI ---")
        for suf, isim in [("_conf", "modelin kendi bildirdigi"),
                          ("_lpconf", "logprob'dan turetilen")]:
            cq = EV.confidence_quality(pred, gold_dev, conf_suffix=suf)
            if len(cq) and cq["AUC_guven"].notna().any():
                m = cq["AUC_guven"].mean(skipna=True)
                print(f"  {isim:<26} ortalama AUC_guven = {m:.3f}")
            else:
                print(f"  {isim:<26} olculemedi (sabit veya eksik)")
        print()
        print("  AUC_guven 0.5 = guven rastgele, dogrulukla ilgisi yok.")
        print("  0.5'e yakinsa Katman 4'te guvene gore agirliklandirma ZARAR verir;")
        print("  o durumda duz agirlik kullanilir.")

# %% [markdown]
# ## 7c. Retry — kesilen raporları yeniden işle
#
# Tam koşuda 53 rapor (%1.2) `max_tokens` tavanına dayanıp kesildi; JSON yarım
# kaldı. Hepsinin sebebi aynıydı (`n_fail_capped == n_fail == 53`), yani
# `MAX_NEW_TOKENS = 1400` ile düzelmesi bekleniyor.
#
# Sadece o 53'ü işliyoruz — 4.407'yi baştan almak ~8.4 saat, bu ~7 dakika.

# %%
if MODE == "retry":
    import vllm_runner as VR

    tag = FULL_RUN_MODEL["id"].split("/")[-1] if FULL_RUN_MODEL else "retry"
    out_path = f"{OUT_DIR}/retry_{tag}.jsonl"
    recs, meta = VR.run(FULL_RUN_MODEL["id"], reports, out_path,
                        tensor_parallel_size=FULL_RUN_MODEL["tp"],
                        max_model_len=MAX_MODEL_LEN,
                        quantization=FULL_RUN_MODEL["quant"],
                        max_tokens=MAX_NEW_TOKENS, logprobs=LOGPROBS, tag="retry")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    df_r = pd.DataFrame(recs if recs else
                        [json.loads(l) for l in open(out_path, encoding="utf-8")])
    print()
    print(f"Yeniden islenen : {len(df_r)}")
    print(f"Basarili        : {int(df_r.parse_ok.sum())} / {len(df_r)}"
          f"  ({df_r.parse_ok.mean():.1%})")
    print(f"Hala tavanda    : {int((df_r.n_out_tokens >= MAX_NEW_TOKENS).sum())}")
    if df_r.parse_ok.mean() == 1.0:
        print("  >> HEPSI KURTARILDI. Faz 1.8 tam kapsam: 4407/4407")
    else:
        kalan = int((~df_r.parse_ok).sum())
        print(f"  >> {kalan} rapor hala basarisiz. Tavanda iseler MAX_NEW_TOKENS")
        print("     daha da artirilabilir; degilse baska bir sebep var.")

# %% [markdown]
# ## 8. Tam koşu
#
# `MODE = "full"` ve `FULL_RUN_MODEL` doldurulduğunda çalışır.
# JSONL'e anında yazdığı için oturum kesilirse aynı notebook'u tekrar
# çalıştırmak kaldığı yerden devam ettirir.

# %%
if MODE == "full":
    if FULL_RUN_MODEL is None:
        raise ValueError("FULL_RUN_MODEL'i bake-off kazananiyla doldurun.")
    if not HAS_VLLM:
        print("!! UYARI: vLLM yok. 4.407 rapor transformers ile cok uzun surer.")
        print("   Yine de devam ediliyor — JSONL resume oldugu icin birden fazla")
        print("   oturumda tamamlanabilir.")
    import vllm_runner as VR

    tag = FULL_RUN_MODEL["id"].split("/")[-1]
    out_path = f"{OUT_DIR}/weak_raw_{tag}.jsonl"
    if HAS_VLLM:
        recs, meta = VR.run(FULL_RUN_MODEL["id"], reports, out_path,
                            tensor_parallel_size=FULL_RUN_MODEL["tp"],
                            max_model_len=MAX_MODEL_LEN,
                            quantization=FULL_RUN_MODEL["quant"],
                            max_tokens=MAX_NEW_TOKENS, logprobs=LOGPROBS)
    else:
        recs, meta = run_transformers(FULL_RUN_MODEL["id"], reports, out_path,
                                      FULL_RUN_MODEL["quant"])
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    allr = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    df = pd.DataFrame(allr)
    print(f"\nToplam islenmis: {len(df):,} / {len(train):,}")
    print(f"Ayristirma basarisi: {df['parse_ok'].mean():.1%}")
    print("\nStatus dagilimi (tum etiketler birlestirilmis):")
    st = pd.concat([df[f"{k}_status"] for k in P.LABELS])
    print((st.value_counts(normalize=True) * 100).round(1).to_string())

    keep = (["StudyInstanceUID", "parse_ok", "exam_completeness", "model"]
            + [c for k in P.LABELS for c in
               (k, f"{k}_status", f"{k}_conf", f"{k}_lpconf", f"{k}_evidence")])
    df[[c for c in keep if c in df.columns]].to_csv(
        f"{OUT_DIR}/weak_labels_layer1.csv", index=False)
    print(f"\nkaydedildi: {OUT_DIR}/weak_labels_layer1.csv")

    # GOLD HOLDOUT SAGLAMASI — bu 38 study prompt ayarinda hic kullanilmadi
    gh = train[train["StudyInstanceUID"].isin(D.GOLD_HOLDOUT)][gold_cols]
    ev = EV.evaluate(df, gh, uncertain="absent")
    print()
    print("=" * 72)
    print("KATMAN 3 — GOLD HOLDOUT UZERINDE NIHAI OLCUM (38 study)")
    print("=" * 72)
    print(ev.to_string(index=False))
    print()
    print(json.dumps(EV.summarise(ev), ensure_ascii=False, indent=2))
    ev.to_csv(f"{OUT_DIR}/layer3_holdout_eval.csv", index=False)
    print()
    print("  Bu tablo Faz 2.8'deki etiket basina loss agirliklandirmasini belirler:")
    print("  F1'i dusuk etiketlerde weak-label agirligi dusurulur.")

# %% [markdown]
# ## 9. TESLİM ÖZETİ — sadece bu hücrenin çıktısını göndermek yeterli
#
# **Neden en sonda ve ayrı bir hücre:** vLLM binlerce `INFO` satırı basıyor ve
# Kaggle uzun çıktıları kırpıyor. Üç turdur kritik satırlar (`gudumlu decoding`,
# özet tablo) o selin içinde kayboldu. Bu hücre en son çalıştığı için önündeki
# log seli onu etkilemiyor — çıktısı kısa ve eksiksiz.
#
# **Çıktıların kalıcı olması için `Save Version → Save & Run All` kullan.**
# Interaktif oturumda `/kaggle/working` oturum bitince uçuyor (No persistence).

# %%
print("=" * 72)
print("TESLIM OZETI")
print("=" * 72)

print()
print("--- /kaggle/working icerigi ---")
for f in sorted(os.listdir(OUT_DIR)):
    try:
        kb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {kb:9.1f} KB  {f}")
    except OSError:
        print(f"  {'?':>9}     {f}")

for ad, yol in [("ABLATION SUMMARY", f"{OUT_DIR}/ablation_summary.csv"),
                ("BAKEOFF SUMMARY", f"{OUT_DIR}/bakeoff_summary.csv"),
                ("KAZANAN — ETIKET BAZINDA", f"{OUT_DIR}/bakeoff_best_per_label.csv"),
                ("KATMAN 3 — GOLD HOLDOUT", f"{OUT_DIR}/layer3_holdout_eval.csv")]:
    print()
    print(f"--- {ad} ---")
    if os.path.exists(yol):
        print(pd.read_csv(yol).to_string(index=False))
    else:
        print("  (yok — bu mod calismadi)")

print()
print("--- RUN DIAGNOSTICS (model basina altyapi) ---")
dp = f"{OUT_DIR}/run_diagnostics.jsonl"
if os.path.exists(dp):
    for line in open(dp, encoding="utf-8"):
        d = json.loads(line)
        print(f"  {d.get('tag') or d.get('model','?')}")
        print(f"      gudumlu={d.get('guided_mode')}")
        print(f"      butce_ok={d.get('budget_ok')}"
              f"  max_model_len={d.get('max_model_len')}"
              f"  en_uzun_prompt={d.get('max_prompt_tokens')}"
              f"  max_tokens={d.get('max_tokens')}")
        print(f"      tavana_dayanan={d.get('n_capped')}/{d.get('n')}"
              f"  (bunlardan ayristirilamayan: {d.get('n_fail_capped')}"
              f" / toplam hata {d.get('n_fail')})")
        print(f"      parse_ok={d.get('parse_ok')}/{d.get('n')}"
              f"  cikti_token_ort={d.get('mean_out_tokens')}"
              f"  {d.get('s_per_report')} s/rapor"
              f"  -> 4407 rapor ~{round(d.get('s_per_report',0)*4407/3600,2)} saat")
else:
    print("  (yok)")

# Tam kosu yapildiysa etiket dagilimini da ozetle
wl = f"{OUT_DIR}/weak_labels_layer1.csv"
if os.path.exists(wl):
    w = pd.read_csv(wl)
    print()
    print("--- TAM KOSU DURUMU ---")
    print(f"  islenmis study: {len(w):,} / 4407   ayristirma: {w['parse_ok'].mean():.1%}")
    st = pd.concat([w[f"{k}_status"] for k in P.LABELS if f"{k}_status" in w.columns])
    print("  status dagilimi: " +
          "  ".join(f"{k}={v:.1%}" for k, v in st.value_counts(normalize=True).items()))

print()
print("=" * 72)
print("Bu blogu oldugu gibi gonder. Ayrica Output sekmesinden indir:")
print("  bakeoff_summary.csv  ·  bakeoff_best_per_label.csv  ·  run_diagnostics.jsonl")
print("=" * 72)
