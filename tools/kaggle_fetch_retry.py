"""429 (rate limit) karsisinda kademeli geri cekilerek cikti indir.

Neden gerekli: uzun sureli yoklama (45 saniyede bir) Kaggle API'sinin hiz
sinirini tetikledi. Israrla tekrar denemek sinir penceresini uzatiyor;
dogru davranis her denemede daha uzun beklemek.
"""
import subprocess, sys, time

KAGGLE = r"C:/Users/tarik/AppData/Roaming/Python/Python313/Scripts/kaggle.exe"
SLUG, DEST, PAT = sys.argv[1], sys.argv[2], sys.argv[3]
BEKLEME = [0, 600, 900, 1800, 1800]        # saniye: hemen, 10dk, 15dk, 30dk, 30dk

for i, w in enumerate(BEKLEME, 1):
    if w:
        print(f"[{i}/{len(BEKLEME)}] {w//60} dk bekleniyor...", flush=True)
        time.sleep(w)
    r = subprocess.run([KAGGLE, "kernels", "output", SLUG, "-p", DEST,
                        "--file-pattern", PAT],
                       capture_output=True, text=True, timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    if "429" in out or "Too Many Requests" in out:
        print(f"[{i}] hala 429", flush=True)
        continue
    if r.returncode != 0:
        print(f"[{i}] HATA: {out.strip()[:300]}", flush=True)
        continue
    print(f"[{i}] BASARILI")
    print(out.strip()[-600:])
    sys.exit(0)

print("TUM DENEMELER 429 — sinir penceresi uzun, daha sonra tekrar denenmeli")
sys.exit(1)
