# FAZ 0 BULGULARI — DICOM Envanteri ve Sonuçları

> Kaynak: `notebooks/00_phase0_dicom_survey.ipynb`, Kaggle'da çalıştırıldı.
> Örneklem: **500 study / 1.000 seri / 2.000 dosya** (train setinin ~%11'i, study düzeyinde rastgele).
> Tarih: 12 Eylül 2026.

---

## 1. Transfer Syntax (Faz 0.3) — ÇÖZÜLDÜ

| Syntax | Pay | Decode testi |
|---|---|---|
| Explicit VR Little Endian (sıkıştırılmamış) | %100.00 | 5/5 başarılı |

**Sonuç:** Örneklemde JPEG Lossless / JPEG 2000 / Implicit VR **hiç görülmedi**.

### Kaggle imajındaki decoder durumu

```
Python 3.12.13 · pydicom 3.0.2
pylibjpeg / pylibjpeg-libjpeg / pylibjpeg-openjpeg / pylibjpeg-rle / python-gdcm : HİÇBİRİ KURULU DEĞİL
Pillow 11.3.0 : var
Aktif handler'lar: numpy, pillow, rle   (gdcm, jpeg_ls, pylibjpeg YOK)
```

**Karar:**
- Veri sıkıştırılmamış olduğu için mevcut kurulum **yeterli**. `pylibjpeg` / `gdcm` zorunlu
  bağımlılık değil; Faz 2.13'teki offline wheel paketleme işi büyük ölçüde iptal.
- **Ama emniyet payı yok:** sıkıştırılmış tek bir dosya çıkarsa okunamaz. Ön işleme script'i
  decode hatasını yakalayıp loglayacak. Log boş çıkmazsa eğitim tarafında `pip install`
  (internet açık), submission tarafında wheel'i Kaggle Dataset'e koymak gerekir.
- Sıkıştırılmamış olması ön işlemenin **hızlı** olacağı anlamına gelir (CPU'da decode maliyeti
  yok, darboğaz ağ I/O).

---

## 2. Tag Envanteri (Faz 0.4)

**83 farklı tag** bulundu (yarışma açıklaması "86 allowlisted" diyor — tutarlı).

### 2.1 Site / scanner proxy

| Alan | Doluluk | Tekil | Study içi tutarlı | En büyük grup | Durum |
|---|---|---|---|---|---|
| `SoftwareVersions` | %94.2 | 45 | 1.000 | 0.115 | Var — ama KULLANMA (aşağıya bak) |
| `ManufacturerModelName` | %100 | 33 | 1.000 | 0.196 | **Kullanılacak** |
| `Manufacturer` | %100 | 11 | 1.000 | 0.240 | **Kullanılacak** |
| `MagneticFieldStrength` | %94.2 | 5 | 1.000 | 0.567 | Tek başına dengesiz |
| `PatientPosition` | %94.2 | 1 | 1.000 | 1.000 | Sabit — işe yaramaz |
| `PhotometricInterpretation` | %100 | 1 | 1.000 | 1.000 | Sabit — MONOCHROME1 yolu hiç tetiklenmeyecek |
| `InstitutionName` | %0 | 0 | — | — | **SİLİNMİŞ** |
| `StationName` | %0 | 0 | — | — | **SİLİNMİŞ** |
| `DeviceSerialNumber` | %0 | 0 | — | — | **SİLİNMİŞ** (olsaydı tartışma biterdi) |

**`study_ici_tutarli = 1.000`** — kritik bilinmeyen temiz çıktı: bir study'nin bütün serileri
aynı cihazdan geliyor, yani hiçbir aday anahtar study'yi ikiye bölmüyor.

Grup anahtarı adayları (study düzeyinde):

| Anahtar | Grup sayısı | En büyük grubun payı |
|---|---|---|
| `Manufacturer` | 11 | 0.240 |
| `Manufacturer + MagneticFieldStrength` | 20 | 0.158 |
| `Manufacturer + ManufacturerModelName` | 37 | 0.154 |
| **`Manufacturer + Model + FieldStrength`** | **45** | **0.124** |

**Karar:** Fold grup anahtarı = **`Manufacturer + ManufacturerModelName`** (ikisi de
`.strip().upper()` ile normalize edilerek). 37 grup, en büyük grup %15.4.

### Neden daha ince anahtarlar (SoftwareVersions, FieldStrength) REDDEDİLDİ

Grup anahtarını daha ince yapmak sızıntıya karşı daha güvenli **değil, daha tehlikelidir**:

- **Kaba anahtar** (az grup): iki farklı merkez aynı gruba düşer. Fold'lar gereğinden fazla
  ayrışır → biraz veri verimsizliği. **Sızıntı yok.**
- **İnce anahtar** (çok grup): tek bir fiziksel cihaz birden fazla gruba bölünür — örneğin
  yazılım güncellemesinden sonra `VD13` → `VE11`. GroupKFold bu grupları farklı fold'lara
  atabilir → aynı cihaz hem train hem val'de → **sızıntı geri gelir.**

Kural: *yeterli grup sayısı ve dengeyi sağlayan en KABA anahtarı seç.*

| Anahtar | Grup | En büyük | Bir cihazı bölme riski |
|---|---|---|---|
| `Manufacturer` | 11 | 0.240 | yok |
| **`Manufacturer + ModelName`** | **37** | **0.154** | **yok** |
| `+ MagneticFieldStrength` | 45 | 0.124 | var (37→45: 8 model değişken alan gücü raporluyor) |
| `SoftwareVersions` | 45 | 0.115 | yüksek (güncelleme cihazı böler) |

Açık kalan soru → Faz 1.12: rapor dili eklendiğinde anahtar yine kabalaşmalı mı yoksa
(dil × üretici) çaprazı fazla mı incelir? Faz 1.14'teki rastgele-vs-gruplu CV farkı
diagnostiği bu kararı ampirik olarak çözecek.

**İki uyarı:**
1. Bu bir *site* değil *scanner* proxy'si. Aynı model farklı hastanelerde bulunur → gerçek merkez
   sayısından az grup üretir. Bu **muhafazakâr** yöndeki bir hata: iki merkez aynı gruba düşerse
   fold'lar gereğinden fazla ayrışır, ki bu güvenli taraf. Tehlikeli olan ters durum (aynı merkezin
   hem train hem val'de olması) engellenmiş olur.
2. `Manufacturer` için 11 tekil değer yüksek — muhtemelen yazım varyantları var
   (`SIEMENS` / `Siemens` / sondaki boşluk). **Gruplamadan önce normalize et** (upper + strip),
   yoksa aynı üretici iki ayrı grup sayılır. → Faz 1.12'de kontrol edilecek.
3. Rapor dili (Faz 1.3) bağımsız ikinci bir site sinyali. Nihai anahtar muhtemelen
   `(dil, scanner)` kombinasyonu olacak.

### 2.2 Lateralite

| Alan | Doluluk | Tekil |
|---|---|---|
| `Laterality` | %48.8 | 3 |
| `ImageLaterality` | — | **SİLİNMİŞ** |

Study'lerin yarısında hangi diz olduğu bilinmiyor.

**Karar:** Katmanlı çıkarım + **etiketli yarı üzerinde ampirik doğrulama**.
Sıra: `Laterality` tag → `SeriesDescription` / `ProtocolName` metin araması →
`ImagePositionPatient` x-işareti → `None` (aynalama yapma).
Her katmanın doğruluğu, tag'i dolu olan %48.8'lik kısımda ölçülecek.

### 2.3 Geometri

| Alan | Doluluk | Tekil (2.000 dosyada) |
|---|---|---|
| `ImagePositionPatient` | %100 | 1.995 |
| `ImageOrientationPatient` | %100 | 975 |
| `PixelSpacing` | %100 | 196 |

**Sonuç:** Slice sıralaması için normal-vektör projeksiyonu yöntemi tam kapasite kullanılabilir,
`InstanceNumber` fallback'ine pratikte hiç ihtiyaç olmayacak.

`PixelSpacing` 196 farklı değer → çözünürlük seriden seriye ciddi değişiyor. Basit resize ile
başlıyoruz; `MONAI.Spacing` ile fiziksel normalizasyon Faz 3.6 ablation'ına bırakıldı.

### 2.4 İntensite

| Alan | Doluluk | Tekil |
|---|---|---|
| `RescaleSlope` | %44.6 | 183 |
| `RescaleIntercept` | %44.6 | 2 |

**Karar:** Var olduğunda uygulanacak. Sebep, mutlak değer değil **seri içi tutarlılık**:
aynı serinin slice'ları farklı slope ile kaydedilmişse, slope uygulanmadan yapılan seri-düzeyi
normalizasyonu slice'lar arası sahte parlaklık farkı üretir.
(Tüm slice'lar aynı slope'a sahipse pozitif-afin dönüşüm percentile+min-max normalizasyonu
altında zaten etkisizdir.)

---

## 3. Etiket Durumu — EN KRİTİK BULGU

```
Toplam train study        : 4.407
Gold (12 etiket de dolu)  :    58   (%1.32)
Hiç etiketi olmayan       : 4.349
```

**58 gold study.** Beklenenden çok daha az.

### Sonuçları

1. **Gold-only validation tek başına kullanılamaz.** 12 etiketin her biri için 58 örnek demek;
   nadir bir bulguda (`Fracture`) muhtemelen 1-3 pozitif vaka olacak. Tek bir vakanın sıralaması
   o etiketin AUC'sini 0.4'ten 0.9'a taşıyabilir. Macro AUC'nin 1/12'si tamamen gürültü olur.
2. **Weak-label pipeline'ın doğrulaması da aynı problemle malul.** Katman 3'teki per-label F1
   tablosu 58 örnek üzerinden hesaplanacak → güven aralıkları çok geniş olacak.
3. **Modelin gördüğü etiketlerin >%98'i weak olacak.** Yani model, weak-labeler'ın hatalarını
   da öğrenecek. Weak-label kalitesi artık "önemli bir bileşen" değil, **skorun tavanı**.

### Açık karar (Faz 1'e girerken verilecek)

Validation stratejisi seçenekleri — detaylı tartışma için bkz. oturum notları:
- (A) Weak-label CV birincil + 58 gold sanity-check
- (B) Elle doğrulanmış ~200-300 study'lik kendi gold setimizi üretmek
- (C) İki bağımsız weak-labeler arasındaki uyumu proxy metrik olarak kullanmak
- (D) Public LB'ye normalden fazla ağırlık vermek (5 submission/gün bütçesiyle)

---

## 4. Faz 0.5-0.9 için doğrudan sonuçlar

| Görev | Envanterin söylediği |
|---|---|
| 0.5 slice sıralama | `IPP`+`IOP` %100 dolu → normal-vektör projeksiyonu birincil, fallback neredeyse hiç devreye girmeyecek |
| 0.6 normalizasyon | Percentile clip + min-max; `RescaleSlope/Intercept` varsa önce uygula |
| 0.7 laterality | %48.8 tag + heuristic katmanları; **aynalama ekseni düzleme göre değişir** (bkz. `src/data/dicom_io.py`) |
| 0.9 2.5D | Slice sayısı seri başına yeterli; komşu slice stack'i sorunsuz |

---

## 5. Faz 0.5–0.9 Doğrulama Sonuçları

> Kaynak: `notebooks/01_phase0_pipeline.ipynb`, Kaggle'da çalıştırıldı.
> Örneklem: 300 seri (başlık denetimi) / 12 seri (piksel okuma). Tarih: 12 Eylül 2026.

### 5.1 Slice sıralama (0.5) — KAPANDI

| Kontrol | Sonuç |
|---|---|
| Sıralama yöntemi | `ImagePositionPatient` 300/300 (fallback hiç devreye girmedi) |
| `InstanceNumber` fiziksel sırayla uyumlu | 300/300 — ayrışma yok |
| Düzensiz slice aralığı (cv > 0.10) | 0/300 (cv ortalaması **1.4e-05**) |
| Aynı pozisyonda tekrarlı slice | 0/300 |
| Hesaplanan düzlem vs CSV etiketi | **300/300 (%100)** |
| Aynalama ekseni | Sagittal→0, Coronal/Axial→2 — beklendiği gibi |

Düzlem uyumunun %100 olması çift kayıt kontrolüdür: `ImageOrientationPatient`'tan hesaplanan
düzlem ile `train_series.csv`'nin `Anatomical_Plane` sütunu bağımsız iki kaynak. Tam örtüşme
hem geometri kodunu hem CSV'yi doğruluyor.

Geometri metadata'sı kusursuz. **Not:** fallback kod yolu gerçek veride hiç tetiklenmiyor,
yani yalnızca `tests/test_dicom_io.py`'daki sentetik vakalarla sınanmış durumda.

### 5.2 Lateralite (0.7) — KATMAN 3 ASKIDA

| Katman | Kapsam | İsabet |
|---|---|---|
| `Laterality` tag | %48.3 (145/300, L=70 R=75) | — |
| `SeriesDescription`/`ProtocolName` metni | %33.1 (48/145) | **%97.9** (1 hata) |
| `ImagePositionPatient` x-işareti | %99.3 (144/145) | **%86.1** |

Katman 1+2 birleşik kapsam ≈ **%65 (seri düzeyinde)**.

Tek metin hatası: `RT_t2_tse_fs_tra` → heuristic `R`, tag `L`. `RT` hem "right" hem protokol
öneki olabilir; tag'in kendisinin hatalı olma ihtimali de var. Katman sıralaması tag'i
öncelediği için bu hata yalnızca tag'siz study'lerde etkili.

#### Katman 3'teki %86 bir kod hatası (düzeltilebilir)

```
L: ortalama  +29.2  medyan  +20.5  min  -88.5  max +122.2
R: ortalama -137.2  medyan -156.7  min -218.3  max  +80.3
```

Medyanlar doğru yöne ayrışıyor ama aralıklar örtüşüyor ve tüm dağılım negatife kaymış.
Sebep: kod `ImagePositionPatient[0]`'ı, yani görüntünün **köşesini** kullanıyor
(`dicom_io.py::detect_laterality`), merkezini değil.

- **Sagittal:** hastanın x ekseni slice eksenidir → slice'ların x ortalaması diz merkezine
  denk gelir. Sapma yok.
- **Coronal/Axial:** x ekseni görüntü içindedir → köşe, merkezden ~yarım FOV (50-80 mm)
  uzakta. Gözlenen sistematik negatif kayma bu.

Heuristic iki farklı konvansiyonu tek eşikle karşılaştırıyor.
**Düzeltme:** görüntü merkezinin x'ini hesapla — `IPP + (Columns/2)·sütun_yönü·PixelSpacing`.

### 5.3 KARAR: Yatay flip augmentation YASAK

12 etiketin 4'ü medial/lateral ayrımı (`Medial Meniscus`, `Lateral Meniscus`, `Medial OA`,
`Lateral OA`). Bir dizde medial/lateral ayrımı, o dizin hangi diz olduğuna bağlı bir sol-sağ
ayrımıdır. Rastgele yatay flip uygulanırsa model bu 4 etikette medial ile lateral'i **tanım
olarak** ayırt edemez → macro AUC'nin 4/12'si tavan 0.5.

Lateralite normalizasyonu opsiyonel bir varyans azaltma tekniği DEĞİL, bu 4 etiketin
öğrenilebilirliğinin ön koşulu. Tercih sıralaması:

**doğru aynalama > hiç aynalamama > rastgele flip**

"Yanlış aynalama" o 4 etikete geometrik etiket gürültüsü enjekte eder — bu yüzden %86'lık
katmanı açmak kapalı tutmaktan kötü olabilir. Eşik kuralı (%70-90 → kapat) korunuyor.

### 5.4 Açık iş: study düzeyi agregasyon

Kapsam **seri** düzeyinde ölçüldü. Bir study'de tek diz var → herhangi bir serisi tarafı
biliyorsa bütün study'nin tarafı bilinir. Study başına ~3 seri olduğundan study düzeyi
kapsam %65'in belirgin şekilde üstünde olmalı (bağımsızlık varsayımıyla ~%96, ama tag
varlığı muhtemelen scanner özelliği olduğu için gerçek değer arada).

Yeni heuristic değil, saf agregasyon. Ölçülmesi başlık okumasıyla bitiyor.

### 5.5 Maliyet — Faz 2.2 planlaması

`0.88 s/seri` · `16.4 ms/slice`

~13.000 serilik train seti: tek thread ~3.2 saat, 8-16 worker ile ~20-30 dakika.
9 saatlik oturum sınırının çok altında — ön işleme darboğaz değil.

---

> **GÜNCELLEME (13 Eylül 2026):** Bölüm 5.2'deki "katman 3 askıda" kararı ve
> Bölüm 5.4'teki study düzeyi agregasyon sorusu **kapandı**. Merkez-x düzeltmesi
> isabeti %86.1 → %98.7'ye çıkardı, kapsam %58 → %99'a yükseldi.
> Ayrıca Bölüm 5.5'teki maliyet tahmini yanlıştı: seri sayısı 13.000 değil **24.371**.
> Hepsi için bkz. **`PHASE1A_FINDINGS.md`**.
