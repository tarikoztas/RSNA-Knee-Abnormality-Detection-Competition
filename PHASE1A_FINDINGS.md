# FAZ 1A BULGULARI — EDA, Dil, Fold Anahtarı, Lateralite

> Kaynak: `notebooks/02_eda.ipynb`, Kaggle'da çalıştırıldı (13 Eylül 2026).
> Kapsam: **tüm train seti** — 4.407 study / 24.371 seri (örneklem değil).
> Ham çıktılar: `outputs/phase1a/`

---

## 1. Lateralite — ÇÖZÜLDÜ (%58 → %99 kapsam)

Faz 0.7'de katman 3 (`ImagePositionPatient` x-işareti) %86.1 isabetle askıya
alınmıştı. Sebebin köşe-vs-merkez karışıklığı olduğu teşhis edildi ve düzeltildi.

| Aşama | İsabet |
|---|---|
| Faz 0.7 — köşe tabanlı (`IPP[0]`) | 0.861 |
| Faz 1A — görüntü merkezi (`image_center`) | **0.987** |

### 1.1 Neden 30 mm ölü bölge — asıl mekanizma

İsabet, scanner markasına değil **dizin izomerkeze göre konumuna** bağlı:

| \|merkez_x\| | İsabet | Seri payı |
|---|---|---|
| 0–10 mm | **0.507** | 1.7% |
| 10–20 mm | 0.647 | 0.8% |
| 20–30 mm | 0.830 | 0.7% |
| 30–50 mm | 0.966 | 4.1% |
| 50–75 mm | 0.993 | 26.2% |
| 75–100 mm | 0.988 | 41.1% |
| 100+ mm | 0.983 | 25.3% |

Monoton ve keskin: 30 mm'nin altında sinyal pratikte **yok** (0-10 mm kovası tam
yazı-tura). Fiziksel sebep: bazı merkezler dizi izomerkeze ortalıyor; o zaman x
işareti tarafı değil gürültüyü ölçer.

**Bu, marka bazlı kural yazma ihtiyacını ortadan kaldırıyor.** Başta scanner
grubu başına isabet hesaplanmış ve 4/19 grup %95'in altında çıkmıştı
(`GEHC | OPTIMA MR430S 1.5T` → **0.417**, yani rastgeleden kötü). Ama o grupların
tek ortak özelliği medyan |merkez_x|'in küçük olmasıydı (5.9 mm ve 13.0 mm;
sağlıklı gruplarda 74–110 mm). Ölü bölge bu serileri markadan bağımsız ayıklıyor.

### 1.2 Tag'siz scanner'lara genelleme güvenli mi — EVET

`Laterality` tag'inin varlığı neredeyse tamamen bir **scanner/site özelliği**:
n≥40 olan grupların **%77.8**'inde tag oranı ya ~0 ya ~1. Tüm GE grupları 0.000,
Toshiba 0.000, Siemens Healthineers ~1.000. Yani tag'li yarı ile tag'siz yarı
farklı üreticilerden geliyor → isabeti tag'li yarıda ölçüp diğerine genellemek
başta riskli görünüyordu.

Doğrudan ölçemiyoruz (yer gerçeği yok) ama **sinyal büyüklüğünü** ölçebiliyoruz.
Tag'i hiç olmayan 23 grup (1.294 study, %29.4) — medyan |merkez_x| değerleri:

```
GE SIGNA EXPLORER 101  ·  GE SIGNA ARTIST  90  ·  GE SIGNA EXCITE  70
GE OPTIMA MR450W   77  ·  GE ARCHITECT     73  ·  TOSHIBA VANTAGE  61
PHILIPS INGENIA   105  ·  TOSHIBA TITAN    89  ·  CANON GALAN 3T   87
```

Hepsi 61–105 mm bandında, yani **%99 isabet rejiminde**. Hiçbiri
`GEHC | OPTIMA MR430S`in izomerkez desenine benzemiyor. Genelleme savunulabilir.

### 1.3 Tag çelişkileri — tag yanılıyor, iki diz yok

25 study'de `Laterality` tag'i seriler arasında çelişkili. **25'inin 25'inde**
merkez-x tek taraf gösterdi → bilateral çekim yok, tag bazı serilerde yanlış.

Örnek (6 serinin hepsi merkez_x ≈ +111…+136, yani açıkça SOL):

```
T2W_BONDORF  Coronal   tag=R   center_x=119.7   <- tag YANLIS
T1W_TSE      Sagittal  tag=L   center_x=111.5
PDW_         Coronal   tag=L   center_x=119.5
PDW_SPAIR    Sagittal  tag=L   center_x=136.0
PDW_SPAIR    Axial     tag=L   center_x=118.9
PDW_SPAIR    Coronal   tag=L   center_x=119.7
```

Çelişkili study'lerde uyuşmazlık oranı %35.0, çelişkisizlerde %2.1. Ama tüm
hataların yalnızca %16.1'i bu study'lerden geliyor.

### 1.4 Çoğunluk oyu isabeti ARTIRMIYOR — ama gerekli

| Ölü bölge | Seri isabeti | Study çoğunluk oyu |
|---|---|---|
| 0 mm | 0.9751 | 0.9737 |
| 30 mm | 0.9873 | 0.9815 |
| 50 mm | 0.9883 | 0.9864 |

Çoğunluk oyu hatayı düşürmüyor → **hatalar study içinde korele**, kaynak aynı
hatalı konumlandırma. Oy vermek gürültüyü ortalamıyor.

Buna rağmen study düzeyi agregasyon zorunlu: (a) bir study'de tek diz var,
(b) 1.3'teki çelişkili tag'leri düzeltiyor.

### 1.5 Nihai yapılandırma

```python
detect_laterality(dss, use_ipp_layer=True, ipp_deadzone=30.0)   # varsayilan
resolve_study_laterality(series_datasets)                       # study duzeyi
```

| Ölçüt | Değer |
|---|---|
| Study kapsamı | **%99.0** (önceki %58.2) |
| Taraf bilinmeyen study | 43 / 4.407 |
| Katman 3'ün kazandırdığı | 1.801 study (+%40.9 puan) |
| Tahmini yanlış aynalama | **~%0.8** (0.41 × %1.85) |

> **Not:** %99.73'lük "birleşik isabet" rakamı yanıltıcıdır — yer gerçeği olan
> study'lerde katman 1 zaten tag'i okuyor, yani ölçüm dairesel. Katman 3'ün tek
> başına isabeti **%98.2** (study düzeyi, 30 mm), yanlış aynalama tahmini bundan
> türetildi.

Çıktı: `outputs/phase1a/laterality_final.csv` — 4.407 study, L=2.045 R=2.319 bilinmeyen=43.

---

## 2. Dil (Faz 1.3)

`fasttext` lid.176 kullanıldı. **Not:** paketin kendi `predict()`i numpy≥2 ile
kırık (`np.array(probs, copy=False)`); notebook alt seviye `model.f.predict()`
yoluna düşerek çözüyor.

| Dil | Study | Pay |
|---|---|---|
| en | 1.730 | 39.3% |
| es | 682 | 15.5% |
| tr | 546 | 12.4% |
| el | 321 | 7.3% |
| de | 263 | 6.0% |
| sh | 231 | 5.2% |
| bg | 220 | 5.0% |
| nl | 152 | 3.5% |
| hr | 137 | 3.1% |
| fr | 81 | 1.8% |
| ? | 41 | 0.9% |
| bs | 3 | 0.1% |

11 dil + 41 sınıflandırılamayan. **`sh` + `hr` + `bs` aynı dil kümesidir**
(Sırp-Hırvat-Boşnak, 371 study); gruplama ve prompt tasarımında tek dil sayılmalı
→ etkin dil sayısı **9**.

### Gold alt kümede dağılım kayması

`en` gold'da %48.3, tüm veride %39.3 → hafif İngilizce ağırlıklı. Daha önemlisi
**`fr` (81 study), `bs` ve `?` grubunda HİÇ gold yok** — o merkezler için
validation sinyali sıfır.

---

## 3. Fold grup anahtarı (Faz 1.7)

### 3.1 Üretici adı varyantları gerçek ve büyük

```
SIEMENS HEALTHINEERS 1053 | SIEMENS 898        -> ayni firma
PHILIPS MEDICAL SYSTEMS 718 | PHILIPS 492 | PHILIPS HEALTHCARE 91
GE MEDICAL SYSTEMS 868 | GEHC 37
TOSHIBA 181 | CANON_MEC 45                     -> Canon, Toshiba Medical'i satin aldi
FUJIFILM HEALTHCARE 16 | HITACHI MEDICAL 8     -> Fujifilm, Hitachi'yi satin aldi
```

`strip().upper()` bunları birleştirmiyor → **aynı cihaz ailesi birden fazla gruba
bölünüyor**, yani `ARCHITECTURE.md` Bölüm 3'teki "ince anahtar sızıntıyı geri çağırır"
riski fiilen mevcut. Eşleme tablosu 11 ham adı **5 aileye** indiriyor
(SIEMENS 1951, PHILIPS 1301, GE 905, CANON 226, FUJIFILM 24).

Benzer şekilde `sh` + `hr` + `bs` aynı dil kümesi → `hbs` olarak birleştirildi,
dil sayısı 11 → 9.

### 3.2 Aday karşılaştırması (4.407 study, 5-fold)

| Anahtar | Grup | En büyük | Tek study'li | Gold grubu | Fold boyutu | Fold başına gold |
|---|---|---|---|---|---|---|
| Manufacturer (ham) | 11 | 0.239 | 0 | 8 | 788–1053 | 10–16 |
| mfr_ailesi | 5 | 0.443 | 0 | 4 | **24–1951** | **0–22** |
| Manufacturer + Model (ham) | 51 | 0.129 | 8 | 26 | 881–882 | 9–15 |
| mfr_ailesi + Model | 46 | 0.168 | 8 | 23 | 881–882 | 9–14 |
| mfr_ailesi + Model + Field | 51 | 0.115 | 8 | 27 | 881–882 | 7–17 |
| lang_g | 10 | 0.393 | 0 | 8 | 632–1730 | 5–28 |
| **lang_g + mfr_ailesi** | **22** | **0.134** | **1** | **15** | **861–924** | **9–14** |
| lang_g + mfr_ailesi + Model | 68 | 0.059 | 13 | 29 | 881–882 | 8–16 |

**Eleme:**
- `mfr_ailesi` tek başına → bir fold'da **0 gold**, fold boyutları 24 ile 1951 arası.
  Kullanılamaz. (5 grup, 5 fold → bir grup bir fold demek.)
- `lang_g` tek başına → fold boyutları 632–1730, gold 5–28. Çok dengesiz.
- `... + Model` varyantları → 8–13 tek study'li grup; gereğinden ince.

### 3.3 ÖNERİ: `lang_g + mfr_ailesi` (22 grup)

`ARCHITECTURE.md` Bölüm 3 kuralı — *yeterli grup sayısı ve dengeyi sağlayan en KABA
anahtar* — bunu seçtiriyor: `Manufacturer + Model`'den kabası (22 vs 51), tek study'li
grup neredeyse yok (1 vs 8), fold'lar dengeli (861–924), fold başına 9–14 gold.

Kavramsal olarak da daha doğru: **dil hangi hastane** sinyali, **üretici ailesi hangi
cihaz tipi** sinyali. Çaprazı "merkez"i model adından daha iyi yaklaşıklıyor, çünkü
model adı tek bir hastanenin iki scanner'ını ayrı gruplara böler.

**Kabul edilen kalan risk:** iki farklı üreticiyle çalışan bir hastane iki gruba
bölünür ve farklı fold'lara düşebilir. Bunun gerçek bir sızıntı olması için modelin
"hastane"yi cihaz görünümünden bağımsız olarak ezberlemesi gerekir; farklı üreticilerin
görüntüleri zaten farklı göründüğü için bu zayıf bir kanal. Hasta kimliği veride yok,
dolayısıyla hasta örtüşmesini kontrol edemiyoruz.

→ **Faz 1.14 diagnostiği** (rastgele vs gruplu CV farkı) bu kararı ampirik olarak
doğrulayacak. Fark küçükse anahtar kabalaştırılabilir.

---

## 4. Maliyet — Faz 2.2 için DÜZELTME

Seri sayısı **24.371**, daha önce ~13.000 tahmin edilmişti. Faz 0.5'teki
0.88 s/seri ile:

- tek thread: **~6 saat** (önceki tahmin 3.2 saat — yanlıştı)
- 8–16 worker: **~25–45 dakika**

9 saatlik oturum sınırı için hâlâ rahat, ama seri seçimi (Faz 2.1) tüm serileri
değil study başına sabit sayıda seri işleyecek — bu da maliyeti ~%40'a indirir.
