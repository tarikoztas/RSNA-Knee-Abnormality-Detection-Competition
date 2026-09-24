"""Bir Kaggle kernel'ini bitene kadar yokla — API hatasini "bitti" sanmadan.

Neden ayri bir arac: ilk yoklama dongumu shell'de `case` ile yazmistim ve
QUEUED/RUNNING disindaki HER SEYI terminal durum sayiyordu. Kimlik dogrulama
suresi dolunca API bir hata METNI dondurdu, dongu onu "is bitti" sandi ve
sessizce cikti. Sonuc: kosunun gercekten bitip bitmedigini bilmiyorduk.

Ders: bir yoklayici, "basarili bitis" ile "sorgulayamadim"i ayirt edemiyorsa
sessizlik basari gibi gorunur.

Kullanim:
  python tools/kaggle_wait.py <owner/slug> [--max-dakika 90] [--aralik 45]

Cikis kodlari:
  0  kernel terminal duruma ulasti (COMPLETE / ERROR / CANCEL...)
  2  kimlik/erisim sorunu — yeniden giris gerekiyor
  3  zaman asimi (hala calisiyor olabilir)
"""
import argparse
import subprocess
import sys
import time

KAGGLE = r"C:/Users/tarik/AppData/Roaming/Python/Python313/Scripts/kaggle.exe"

DEVAM = ("QUEUED", "RUNNING")
BITTI = ("COMPLETE", "ERROR", "CANCEL", "KILLED", "FAILED")
KIMLIK = ("Authentication required", "was denied", "401", "403",
          "credentials", "Unauthorized")


def durum(slug):
    """('devam'|'bitti'|'kimlik'|'bilinmiyor', ham_metin) dondur."""
    try:
        r = subprocess.run([KAGGLE, "kernels", "status", slug],
                           capture_output=True, text=True, timeout=120)
    except Exception as e:
        return "bilinmiyor", f"{type(e).__name__}: {e}"
    txt = ((r.stdout or "") + (r.stderr or "")).strip()
    up = txt.upper()
    if any(k.upper() in up for k in KIMLIK):
        return "kimlik", txt
    if any(k in up for k in DEVAM):
        return "devam", txt
    if any(k in up for k in BITTI):
        return "bitti", txt
    return "bilinmiyor", txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--max-dakika", type=float, default=90)
    ap.add_argument("--aralik", type=float, default=45)
    a = ap.parse_args()

    son = time.time() + a.max_dakika * 60
    bilinmiyor_ustuste = 0
    while time.time() < son:
        kind, txt = durum(a.slug)
        if kind == "bitti":
            print(f"BITTI: {txt}")
            return 0
        if kind == "kimlik":
            print("KIMLIK SORUNU — yoklama durduruldu (sessizce 'bitti' sanmamak icin)")
            print(f"  {txt.splitlines()[0][:160]}")
            print("  Cozum: terminalinde  kaggle.exe auth login")
            return 2
        if kind == "bilinmiyor":
            bilinmiyor_ustuste += 1
            # Gecici ag hatasi olabilir; ust uste 3 kez olursa pes et.
            if bilinmiyor_ustuste >= 3:
                print(f"BILINMEYEN DURUM (3 kez ust uste): {txt[:200]}")
                return 2
        else:
            bilinmiyor_ustuste = 0
        time.sleep(a.aralik)

    print(f"ZAMAN ASIMI: {a.max_dakika:.0f} dakika doldu, kernel hala calisiyor olabilir")
    return 3


if __name__ == "__main__":
    sys.exit(main())
