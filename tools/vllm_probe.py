"""vLLM'in gudumlu decoding API'sini TESPIT ET — model yuklemeden, 10 saniyede.

Neden ayri bir sonda: ilk bake-off'ta gudumlu decoding sessizce calismadi ve
bunu ancak 3 model kostuktan ve ham ciktilari analiz ettikten sonra anladik.
Bu sonda model yuklemedigi icin saniyeler suruyor ve dogrudan cevap veriyor:
hangi API sekli bu vLLM surumunde kabul ediliyor?

Kaggle'da kullanim: vLLM kurulum hucresinden SONRA yeni bir hucreye yapistirip
calistir. Ciktiyi oldugu gibi gonder.
"""
import sys


def probe():
    print("=" * 68)
    print("vLLM GUDUMLU DECODING SONDASI")
    print("=" * 68)
    try:
        import vllm
    except Exception as e:
        print(f"vllm import EDILEMIYOR: {type(e).__name__}: {e}")
        return
    print("vllm surumu:", getattr(vllm, "__version__", "?"))
    print("python     :", sys.version.split()[0])

    try:
        import torch
        print("torch      :", torch.__version__)
    except Exception:
        pass

    # Hangi sinif adlari var?
    print()
    print("--- ilgili siniflar mevcut mu? ---")
    for modpath, name in [("vllm.sampling_params", "StructuredOutputsParams"),
                          ("vllm", "StructuredOutputsParams"),
                          ("vllm.sampling_params", "GuidedDecodingParams"),
                          ("vllm", "GuidedDecodingParams")]:
        try:
            mod = __import__(modpath, fromlist=[name])
            getattr(mod, name)
            print(f"  VAR  {modpath}.{name}")
        except Exception as e:
            print(f"  yok  {modpath}.{name}  ({type(e).__name__})")

    # SamplingParams hangi alanlari kabul ediyor?
    print()
    print("--- SamplingParams alanlari ---")
    from vllm import SamplingParams
    fields = None
    for attr in ("model_fields", "__dataclass_fields__", "__annotations__"):
        f = getattr(SamplingParams, attr, None)
        if f:
            fields = sorted(f.keys())
            print(f"  ({attr} uzerinden, {len(fields)} alan)")
            break
    if fields:
        ilgili = [f for f in fields
                  if any(s in f.lower() for s in
                         ("guid", "struct", "json", "schema", "logprob", "grammar"))]
        print("  ilgili alanlar:", ilgili or "(hicbiri)")
    else:
        print("  alan listesi alinamadi")

    # Gercekten insa edilebiliyor mu? (asil test)
    schema = {"type": "object",
              "properties": {"x": {"type": "string"}},
              "required": ["x"], "additionalProperties": False}
    base = dict(temperature=0.0, max_tokens=64, logprobs=5)

    print()
    print("--- INSA TESTI (asil cevap burada) ---")
    ok_any = False

    for modpath, clsname in [("vllm.sampling_params", "StructuredOutputsParams"),
                             ("vllm", "StructuredOutputsParams")]:
        try:
            mod = __import__(modpath, fromlist=[clsname])
            cls = getattr(mod, clsname)
            SamplingParams(structured_outputs=cls(json=schema), **base)
            print(f"  CALISIYOR  structured_outputs={clsname}(json=...)   [V1 API]")
            ok_any = True
            break
        except Exception as e:
            print(f"  olmadi     {clsname}: {type(e).__name__}: {str(e)[:110]}")

    try:
        from vllm.sampling_params import GuidedDecodingParams
        SamplingParams(guided_decoding=GuidedDecodingParams(json=schema), **base)
        print("  CALISIYOR  guided_decoding=GuidedDecodingParams(json=...)")
        ok_any = True
    except Exception as e:
        print(f"  olmadi     GuidedDecodingParams: {type(e).__name__}: {str(e)[:110]}")

    try:
        SamplingParams(guided_json=schema, **base)
        print("  CALISIYOR  guided_json=...")
        ok_any = True
    except Exception as e:
        print(f"  olmadi     guided_json: {type(e).__name__}: {str(e)[:110]}")

    print()
    print("=" * 68)
    if ok_any:
        print("SONUC: gudumlu decoding KULLANILABILIR -> bake-off'u calistir.")
    else:
        print("SONUC: HICBIRI CALISMIYOR. Yukaridaki hata mesajlarini gonder;")
        print("       vLLM surumune gore baska bir yol (orn. outlines/xgrammar")
        print("       backend'i veya surum sabitleme) gerekecek.")
    print("=" * 68)


if __name__ == "__main__":
    probe()
