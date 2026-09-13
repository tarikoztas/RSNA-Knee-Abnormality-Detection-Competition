# PROJECT OVERVIEW — RSNA Knee Abnormality Detection

> Bu dosya projenin "ne yapıyoruz ve hangi kurallara tabiyiz" referansıdır.
> Teknik mimari için `ARCHITECTURE.md`, faz/görev detayları için `PHASES.md` dosyasına bak.

---

## 1. Yarışma

**Kaggle:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
**Sponsor:** Radiological Society of North America (RSNA)
**Toplam ödül:** $77,000

### Görev

Her diz MR **çalışması (study)** için 12 ikili bulgunun olasılık skorunu tahmin et:

| # | Etiket | Açıklama |
|---|--------|----------|
| 1 | `ACL` | Ön çapraz bağ yaralanması |
| 2 | `MCL` | İç yan bağ yaralanması |
| 3 | `Medial Meniscus` | İç menisküs yırtığı |
| 4 | `Lateral Meniscus` | Dış menisküs yırtığı |
| 5 | `Medial OA` | Medial tibiofemoral kompartman osteoartriti |
| 6 | `Lateral OA` | Lateral tibiofemoral kompartman osteoartriti |
| 7 | `PF OA` | Patellofemoral osteoartrit |
| 8 | `Effusion` | Eklem efüzyonu / sıvı birikimi |
| 9 | `Synovitis` | Sinovyal zar iltihabı |
| 10 | `Baker's` | Baker kisti |
| 11 | `Contusion` | Kemik kontüzyonu / kemik ödemi |
| 12 | `Fracture` | Kırık |

Bu bir **multi-label** problemdir (softmax DEĞİL, 12 bağımsız sigmoid).

### Değerlendirme Metriği

**Macro-averaged ROC-AUC** — 12 hedefin ROC-AUC'lerinin ağırlıksız ortalaması.

Önemli sonuçları:
- Mutlak kalibrasyon skoru etkilemez, sadece **sıralama (ranking)** önemlidir.
- Her etiket eşit ağırlıklıdır → nadir bulgular (örn. `Fracture`) sık bulgular kadar değerlidir.
  Nadir bir etikette AUC 0.5'e düşerse macro ortalama ciddi zarar görür.
- Tek bir etiketi iyileştirmek toplam skorun 1/12'sini etkiler.

### Submission Formatı

Dosya adı **kesinlikle** `submission.csv` olmalı:

```
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
```

Sütun isimleri birebir yukarıdaki gibi olmalı (boşluklar ve kesme işareti dahil).

---

## 2. Kod Yarışması Kısıtları (SERT KURALLAR)

| Kısıt | Değer |
|---|---|
| Submission yöntemi | Sadece Kaggle Notebook (CSV upload YOK) |
| CPU Notebook runtime | ≤ 9 saat |
| GPU Notebook runtime | ≤ 9 saat |
| İnternet erişimi | **KAPALI** (submit sırasında) |
| External data / pretrained model | Serbest — ama **herkese açık ve ücretsiz** olmalı |
| Günlük submission | Maks. 5 |
| Final submission seçimi | Maks. 2 |
| Maksimum takım büyüklüğü | 5 |

### İnternet kapalı olmasının pratik sonuçları

- `pip install` submission notebook'unda **çalışmaz**. İhtiyaç duyulan tüm paketler ya Kaggle imajında hazır olmalı ya da bir Kaggle Dataset olarak yüklenip offline kurulmalı.
- `timm`/HuggingFace model ağırlıkları **indirilemez**. Tüm pretrained ağırlıklar önceden bir Kaggle Dataset'e yüklenip oradan okunmalı.
- Bunu **Faz 2'de ilk submission denemesinde** doğrula, sona bırakma.

---

## 3. Takvim

| Tarih | Aşama |
|---|---|
| 30 Temmuz 2026 | Başlangıç |
| **15 Ekim 2026** | Entry Deadline + Team Merger Deadline |
| **22 Ekim 2026** | **Final Submission Deadline** |
| 5 Kasım 2026 | Kazananların kod/video/açıklama teslimi |

Tüm saatler 23:59 UTC.

---

## 4. Veri Seti

**Toplam:** 819,640 dosya, **569.76 GB**.
**ASLA yerel makineye indirilmeyecek** — Kaggle Notebook'ta `/kaggle/input/` altında hazır mount edilmiş gelir.

### Dosyalar

#### `train.csv` — bir satır = bir eğitim study'si
- `StudyInstanceUID` — study'nin benzersiz kimliği, `train_series/` altındaki klasör adıyla eşleşir
- `Report` — serbest metin radyoloji raporu. **12 farklı dilde olabilir.**
- 12 ikili etiket sütunu (yukarıdaki tabloda listelenen)

#### `train_series.csv` — bir satır = bir seri
- `StudyInstanceUID` — serinin ait olduğu study
- `SeriesInstanceUID` — serinin benzersiz kimliği
- `Fluid_Sensitive` — 1 ise sekans sıvı sinyalini vurguluyor (T2, PD, STIR vb.)
- `Fat_Suppression` — 1 ise yağ baskılama uygulanmış
- `Anatomical_Plane` — `Sagittal` / `Coronal` / `Axial`

> **Not:** `Fluid_Sensitive` ve `Fat_Suppression` sık sık korele ama resmi açıklamada
> "her vaka için eşdeğer değildir" uyarısı var → ikisini AYRI feature olarak tut.

#### `train_series/` — DICOM'lar
```
train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
```
Her `.dcm` tek bir 2D kesit (slice). Seri başına tipik olarak **20-45 slice** (medyan 30),
uzun kuyrukta birkaç yüz slice'a çıkabilir.

#### `test.csv` / `test_series.csv` / `test_series/`
- Örnek dosyalarda sadece 3 study var; skorlama sırasında gerçek veriyle değiştirilir.
- Gerçek test setinde **~1300 study** var.
- **`Report` alanı test aşamasında VERİLMEZ.** → Final model sadece görüntüden çalışmalı.

#### `sample_submission.csv`
Tüm etiketler 0.5. Efficiency track için benchmark görevi görür.

---

## 5. Kritik Veri Sorunları

Bunlar bu yarışmanın asıl zorluklarıdır. Model mimarisinden daha belirleyici olma ihtimalleri yüksek.

### 5.1 Zayıf etiketler (EN KRİTİK)

Resmi veri açıklamasından:
> *"Only a small subset of training studies carry per-condition labels. We also provide the
> original text of the radiology report from which you may wish to derive the labels for the
> remaining studies."*

Yani:
- Eğitim setinin **sadece küçük bir kısmı** gold (doğrulanmış) etikete sahip.
- Geri kalanının etiketleri **rapor metninden bizim çıkarmamız** bekleniyor.
- Elimizde iki kalite katmanı olacak: **gold** (küçük, güvenilir) ve **weak/silver** (büyük, gürültülü).

**Altın kural:** Gold alt kümeyi ASLA weak-label üretiminde kullanma. Onu sadece
validation/kalibrasyon için sakla — aksi halde weak-label kalitesini ölçemezsin.

### 5.2 Site / scanner heterojenliği

19+ katkı veren merkez, 5 kıta, farklı scanner ve protokoller. Risk: model "bu görüntü hangi
merkezden geldi"yi öğrenip bunu bulgu tahmini yerine bir kısayol olarak kullanabilir.

**Çözüm:** Fold'ları hasta/merkeze göre grupla (`GroupKFold`). Rastgele split scanner
ezberlemesini gizler ve CV skorunu yapay olarak şişirir.

**Problem:** Veri şemasında doğrudan bir "site" sütunu YOK. Kullanılabilecek proxy'ler:
- **Rapor dili** — her merkez muhtemelen baskın olarak tek dilde rapor yazıyor
- **DICOM metadata** — `Manufacturer`, `ManufacturerModelName`, `MagneticFieldStrength`,
  `InstitutionName` gibi alanlar 86 allowlisted tag içinde kalmış olabilir
  → **Faz 0'da hangi tag'lerin gerçekten mevcut olduğunu envanterle**

### 5.3 DICOM teknik varyasyonu

Resmi nottan: intensiteler, oryantasyonlar ve çözünürlükler seri ve study arasında değişken.
Transfer syntax'lar karışık:
- Uncompressed Explicit VR Little Endian
- JPEG Lossless
- JPEG 2000
- Implicit VR Little Endian

→ `pydicom` + `pylibjpeg` / `gdcm` plugin'leri gerekebilir. Faz 0'da her transfer syntax'ın
okunabildiğini doğrula.

Metadata **86 allowlisted tag'e** stripped edilmiş (de-identification) → beklediğin bazı
standart alanlar olmayabilir.

### 5.4 Dağılım kayması

Resmi uyarı: bulgu prevalansının train / public LB / private LB arasında aynı olacağı
**garanti edilmiyor**.

→ Public LB'ye aşırı uyum (overfitting) riskli. Final submission seçiminde kendi CV skoruna
public LB kadar ağırlık ver.

### 5.5 Çok dillilik

12 dilde rapor → tek dilli veya kural tabanlı etiket çıkarımı verinin büyük kısmını kaçırır.

---

## 6. Compute Stratejisi

Yerel makinede GPU yok, disk yetersiz (sadece i7 CPU). Bu nedenle:

| İş | Nerede |
|---|---|
| Kod yazma, mantık kurma | Claude Code + Colab editör |
| CSV analizleri / EDA | Colab (CPU yeterli) |
| DICOM prototipleme | Kaggle Notebook (veri orada) |
| Model eğitimi (GPU) | Kaggle Notebook (birincil) → Colab / Lightning AI (ek kota) |
| Weak-label LLM inference | Colab / Kaggle GPU |
| Final submission | Kaggle Notebook (zorunlu) |

**Yarışma kısıtı sadece final submission notebook'una ait.** Eğitim/deney aşamasında
istediğiniz platformu kullanabilirsiniz — tek şart, inference kodunu sonunda internetsiz
çalışan bir Kaggle Notebook'a paketlemek.

### Ön işlenmiş dataset stratejisi (ÖNEMLİ)

570 GB ham DICOM'u her oturumda yeniden decode etmek zaman israfıdır. Bir kere işleyip
kendi küçültülmüş dataset'inizi oluşturun:

1. Her seriden gerekli slice'ları seç (hepsini değil)
2. Düşük çözünürlüğe indir (256×256 veya 384×384)
3. `.npy` veya sıkıştırılmış formatta kaydet
4. Kaggle Dataset olarak yayınla (private)

Bu 570 GB'ı muhtemelen birkaç GB - birkaç on GB seviyesine indirir ve her yeni notebook
oturumunda hazır veriden başlamanızı sağlar.

---

## 7. Takım

2 kişilik takım, ancak şu an **bağımsız/paralel** çalışılıyor. Görevler tek kişilik
varsayımıyla planlandı. Team merger deadline: 15 Ekim 2026.

Kural notu: Takım dışıyla özel kod paylaşımı yasak (private sharing). Public forumda
paylaşım serbest.
