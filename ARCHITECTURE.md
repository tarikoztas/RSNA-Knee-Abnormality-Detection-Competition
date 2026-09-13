# ARCHITECTURE — Teknik Mimari ve Yöntemler

> Bu dosya "nasıl yapıyoruz" referansıdır. Problem tanımı için `PROJECT_OVERVIEW.md`,
> zaman planı ve görev listesi için `PHASES.md` dosyasına bak.

---

## 0. Teknoloji Yığını

| Alan | Araç | Neden |
|---|---|---|
| Framework | **PyTorch** | Medikal görüntüleme ekosisteminin standardı; MONAI + timm + HF aynı ekosistemde |
| DICOM okuma | `pydicom` (+ `pylibjpeg`, `gdcm`) | Karışık transfer syntax desteği için plugin'ler gerekli |
| Resampling | `SimpleITK` (opsiyonel) | Pixel spacing normalizasyonu |
| Görüntü transform | `MONAI` | Medikal görüntü için tasarlanmış hazır transform'lar |
| Vision backbone | `timm` | EfficientNet, ConvNeXt, DINOv2 tek satırda |
| Mixed precision | `torch.cuda.amp` | Kaggle GPU belleği/süresi için şart |
| Metrik | `torchmetrics`, `scikit-learn` | Per-label ve macro ROC-AUC |
| NLP (weak-label) | HuggingFace `transformers` | Multilingual LLM + XLM-R fine-tuning |
| Dil tespiti | `fasttext` (lid.176) veya `langdetect` | Site proxy'si ve dil dağılımı analizi |
| Fold stratejisi | `sklearn.model_selection.GroupKFold` | Scanner/site sızıntısını önlemek |
| Deney takibi | CSV log (basit) veya Weights & Biases | 2 kişi paralel çalışıyor, karşılaştırma gerekir |

---

## 1. Veri Pipeline'ı

### 1.1 DICOM Okuma Katmanı

```
DICOM dosyaları
  → pydicom.dcmread()
  → transfer syntax handling (pylibjpeg/gdcm plugin)
  → pixel_array çıkarımı
  → slice sıralama
  → intensite normalizasyonu
  → laterality normalizasyonu
  → resize
  → 2.5D stack
  → tensor
```

#### Slice sıralama

`InstanceNumber`'a tek başına güvenme — bazı serilerde tutarsız olabilir.
Daha güvenilir yöntem: `ImagePositionPatient` + `ImageOrientationPatient`'tan slice
normal vektörünü hesapla, slice'ları bu eksen boyunca projeksiyona göre sırala.

```python
# Kavramsal
normal = np.cross(orientation[:3], orientation[3:])
sort_key = np.dot(position, normal)
```

Fallback: `ImagePositionPatient` yoksa `InstanceNumber` kullan.

#### Intensite normalizasyonu

MR'da CT'deki gibi sabit Hounsfield birimleri **yoktur** — her seri kendi ölçeğinde.

- **Yöntem:** Percentile-based clipping (%1–%99) + min-max ölçekleme,
  veya seri bazında z-score normalizasyon.
- MONAI karşılığı: `ScaleIntensityRangePercentiles`
- Bit derinliği: MR genelde 12-bit ama 16-bit container'da saklanır — okurken kontrol et.
- `RescaleSlope` / `RescaleIntercept` varsa uygula.

#### Laterality normalizasyonu

Sol dizleri sağ dize göre aynala. Model iki yönü ayrı ayrı öğrenmek zorunda kalmaz,
varyans azalır.

> **DÜZELTME (Faz 0.7 bulgusu):** Aynalama **her zaman horizontal flip DEĞİLDİR** —
> ekseni düzleme göre değişir. Sol dizi sağa çevirmek hastanın x ekseni boyunca bir
> yansımadır; bu yansımanın görüntüdeki karşılığı:
>
> | Düzlem | Görüntüdeki karşılığı | Eksen |
> |---|---|---|
> | Coronal / Axial | yatay çevirme | `axis=2` (genişlik) |
> | **Sagittal** | **slice sırasını ters çevir** (medial↔lateral) | `axis=0` |
>
> Sagittal'de yatay flip yapmak ön-arka eksenini ters çevirir — patellayı arkaya taşır,
> yani sessizce anatomik olarak yanlış veri üretir. Genel çözüm: hacmin üç ekseninden
> (slice / yükseklik / genişlik) hangisinin hasta-x bileşeni en büyükse o eksende çevir.
> Uygulama: `src/data/dicom_io.py::lr_flip_axis`.

**Taraf tespiti:** `Laterality` tag'i train setinin sadece **%48.8**'inde dolu,
`ImageLaterality` ise tamamen silinmiş (Faz 0.4 envanteri). Katmanlı çıkarım gerekiyor:
tag → `SeriesDescription`/`ProtocolName` metni → `ImagePositionPatient` x-işareti.
Her katmanın isabeti, tag'i dolu olan yarıda ölçülüyor
(`notebooks/01_phase0_pipeline.ipynb` Bölüm 4).

> **Dikkat:** Bu yapıldıktan sonra augmentation'da rastgele horizontal flip kullanmak
> normalizasyonu bozar — ya flip'i kapat ya da bilinçli olarak tekrar aç.

#### Boyutlandırma

Sabit matrise resize (256×256 başlangıç, DINOv2 için 224 veya 384). Pixel spacing
farklarını `MONAI.Spacing` ile normalize etmek daha doğru sonuç verir ama daha yavaştır —
önce basit resize ile başla, ablation'da karşılaştır.

### 1.2 2.5D Temsil

Tam 3D CNN pahalı, tek 2D slice bağlamsız. Ortası: **2.5D**.

- Merkez slice ± k komşu slice'ı **kanal boyutunda** stack'le (örn. k=2 → 5 kanal).
- Pretrained backbone 3 kanal bekler → ilk conv katmanının ağırlıklarını kanal sayısına
  göre çoğalt/ortala (timm `in_chans` parametresi bunu otomatik yapar).
- Bir seriden birden fazla 2.5D pencere örneklenebilir (slice ekseninde kayarak).

### 1.3 Seri Seçimi

Bir study'de tipik olarak 3-6 seri var. Hepsini işlemek pahalı, rastgele seçmek bilgi kaybı.

**Strateji: metadata güdümlü öncelikli seçim**

`train_series.csv`'deki üç alanı kullanarak her study'den sabit sayıda seri seç:

| Bulgu grubu | Öncelikli seri tipi |
|---|---|
| Effusion, Synovitis, Contusion, Baker's | Fluid-sensitive + fat-suppressed |
| Meniscus (medial/lateral) | Sagittal ve coronal düzlem |
| ACL | Sagittal (özellikle oblik sagittal) |
| MCL | Coronal |
| OA (medial/lateral/PF) | Coronal + axial (PF için axial) |

Pratik başlangıç: her study'den **her düzlemden en az bir seri** (sagittal + coronal + axial),
fluid-sensitive olanları önceliklendirerek. Böylece hem kapsayıcı hem sabit boyutlu bir
girdi elde edilir.

---

## 2. Weak-Label Pipeline (En Kritik Bileşen)

Basit regex 12 dilde ifade çeşitliliğini yakalayamaz. Katmanlı bir sistem kuruyoruz.

```
Raporlar (çok dilli, ~5000)
  │
  ├─→ [Katman 1] Multilingual LLM ile yapılandırılmış çıkarım
  │       → present / absent / uncertain + confidence
  │
  ├─→ [Katman 2] XLM-R distillation (LLM çıktısı + gold ile fine-tune)
  │       → hızlı, tekrarlanabilir etiketleyici
  │
  ├─→ [Katman 3] Gold alt kümede K-fold doğrulama
  │       → per-label F1 / precision / recall / AUC
  │
  └─→ [Katman 4] Confidence-weighted soft label üretimi
          → görüntü modelinin eğitim hedefi
```

### Katman 1 — LLM tabanlı yapılandırılmış çıkarım

**Model:** Açık kaynaklı, multilingual yeteneği güçlü bir LLM (Qwen2.5 veya Llama-3.1 ailesi).
Boyut Colab/Kaggle GPU belleğine göre seçilir (7B-8B sınıfı quantized olarak sığar).

**Prompt tasarımı ilkeleri:**
- Her rapor için 12 bulgunun her biri hakkında **üç durumlu** çıktı iste:
  `present` / `absent` / `uncertain`
- Yapılandırılmış çıktı (JSON) iste, serbest metin değil
- **Negation'ı açıkça talep et:** "Bir bulgunun yokluğu belirtilmişse `absent` işaretle"
  ("no evidence of tear", "yırtık izlenmedi" vb.)
- Few-shot örnekler ekle — mümkünse farklı dillerden
- Rapor dilini belirtme zorunluluğu koyma; model kendi anlasın (dil tespiti ayrı bir adım)
- Güven skoru iste (0-1) veya logprob'lardan türet

**Neden bu yaklaşım:**
- Regex'in yakalayamayacağı ifade çeşitliliğini doğal dil anlama ile yakalar
- Negation'ı ayrı bir NegEx modülü kurmadan model kendi handle eder
- ~5000 kısa rapor → batch inference ile makul sürede işlenir

### Katman 2 — Distillation

LLM'i her seferinde çalıştırmak yavaş ve maliyetli. LLM çıktısı + gold veriyi birleştirip
daha hafif bir multilingual encoder'ı fine-tune et:

- **Model:** `xlm-roberta-base` (veya klinik metne yakın multilingual bir varyant)
- **Head:** 12 çıkışlı multi-label sigmoid
- **Loss:** BCE (LLM'in soft/uncertain çıktıları soft target olarak kullanılabilir)
- **Fayda:** Hızlı, tekrarlanabilir; ayrıca LLM gürültüsünü bir miktar filtreleyip
  genelleşen sinyali öğrenir

### Katman 3 — Doğrulama (ATLAMA)

Gold alt kümeyi K-fold'a böl. Weak-labeler'ı (hem LLM hem distillation modelini) her
fold'da farklı gold alt kümesiyle test et.

Ölç: **per-label precision / recall / F1 / ROC-AUC**

Bu sana hangi bulgular için weak-label'ların güvenilir olduğunu **nicel olarak** gösterir.
Beklenti: `Fracture`, `Baker's` gibi net ifadeli bulgular yüksek F1; `Synovitis` gibi
daha subjektif olanlar düşük.

Bu tablo, Katman 4'teki ağırlıklandırmayı belirler.

### Katman 4 — Confidence-weighted soft label

Görüntü modelini eğitirken weak-label'ları sert 0/1 yerine **soft target** kullan:

- LLM confidence veya distillation modelinin sigmoid çıktısı → soft label
- `uncertain` işaretli örnekler → ya loss'tan maskele ya da düşük ağırlıkla dahil et
- Katman 3'te düşük F1 alan etiketler → o etiket için weak-label ağırlığını düşür
- Gold örnekler → yüksek ağırlık (örn. weak'in 5-10 katı)

**Loss örneği (kavramsal):**
```python
loss = BCE(pred, target, reduction='none')
loss = loss * sample_weight * label_confidence_mask
loss = loss.mean()
```

---

## 3. Fold Stratejisi

**Rastgele K-fold KULLANMA.** Scanner/site ezberlemesi CV skorunu yapay olarak şişirir.

### Grup anahtarı bulma — KARAR VERİLDİ (Faz 0.4)

Veride doğrudan "site" sütunu yok. Faz 0.4 envanteri adayları şu şekilde eledi
(tam tablo: `PHASE0_FINDINGS.md` Bölüm 2.1):

| Aday | Durum |
|---|---|
| `InstitutionName`, `StationName`, `DeviceSerialNumber` | **%0 — tamamen silinmiş.** Kullanılamaz |
| `Manufacturer` | %100 dolu, 11 tekil |
| `ManufacturerModelName` | %100 dolu, 33 tekil |
| `MagneticFieldStrength` | %94.2, 5 tekil — tek başına dengesiz (en büyük grup %56.7) |
| `SoftwareVersions` | %94.2, 45 tekil — **kullanma** (aşağıya bak) |

**Seçilen anahtar:** `Manufacturer + ManufacturerModelName`, ikisi de `.strip().upper()`
ile normalize edilerek → **37 grup, en büyük grubun payı %15.4.**

`study_ici_tutarli = 1.000` ölçüldü: bir study'nin bütün serileri aynı cihazdan geliyor,
yani hiçbir aday anahtar bir study'yi iki gruba bölmüyor.

#### Neden daha İNCE anahtar seçilmedi

Sezgiye aykırı ama: grup anahtarını incelt­mek sızıntıya karşı daha güvenli **değil,
daha tehlikelidir.**

- **Kaba anahtar** (az grup): iki farklı merkez aynı gruba düşer → fold'lar gereğinden
  fazla ayrışır, biraz veri verimsizliği. **Sızıntı yok.**
- **İnce anahtar** (çok grup): tek bir fiziksel cihaz birden fazla gruba bölünür —
  örneğin yazılım güncellemesinden sonra `VD13` → `VE11`. GroupKFold bu iki grubu farklı
  fold'lara atabilir → aynı cihaz hem train hem val'de → **sızıntı geri gelir.**

Kural: *yeterli grup sayısı ve dengeyi sağlayan en KABA anahtarı seç.*
`SoftwareVersions` ve `+ MagneticFieldStrength` tam bu gerekçeyle reddedildi.

**İki sınır:**
1. Bu bir *site* değil **scanner** proxy'si. Aynı model farklı hastanelerde bulunur →
   gerçek merkez sayısından az grup üretir. Bu **muhafazakâr** yöndeki hata: iki merkezin
   aynı gruba düşmesi güvenli taraftır, tehlikeli olan tersi (aynı merkezin hem train hem
   val'de olması) engellenmiş olur.
2. **Rapor dili** (Faz 1.3) bağımsız ikinci bir site sinyali. Nihai anahtar `(dil, scanner)`
   kombinasyonu olabilir — ama çapraz almak anahtarı inceltir, yukarıdaki kural gereği
   otomatik iyileştirme sayılmaz. Faz 1.14'teki rastgele-vs-gruplu CV farkı diagnostiği
   bu kararı ampirik olarak çözecek.

### Uygulama

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=site_proxy):
    ...
```

Multi-label + gruplu + stratifiye ideal olurdu ama zor; grup önceliklidir.
Alternatif: `StratifiedGroupKFold` (sklearn ≥1.0) tek bir etikete göre stratify eder.

### Doğrulama testi

Aynı modeli hem rastgele hem gruplu fold ile eğitip CV skorlarını karşılaştır.
Aradaki fark ne kadar büyükse, site sızıntısı o kadar güçlü demektir. Bu farkı ölçmek,
CV'nin ne kadar güvenilir olduğunu anlamak için değerli bir teşhis adımıdır.

---

## 4. Görüntü Modeli Mimarisi

### 4.1 Baseline (Faz 2)

```
Study
 └─ Seçilmiş seriler (N adet)
      └─ Her seri → 2.5D stack (C×H×W)
           └─ Paylaşılan ağırlıklı backbone (EfficientNet-B0)
                └─ Seri embedding (D boyutlu)

 Seri embedding'leri → agregasyon (mean / attention pooling)
                     → study embedding
                     → Linear head
                     → 12 sigmoid çıkış
```

**Anahtar noktalar:**
- Backbone ağırlıkları tüm seriler arasında **paylaşılır** (shared weights)
- Agregasyon başlangıçta `mean pooling`, sonra `attention pooling`'e yükselt
- Çıkış: 12 bağımsız sigmoid (multi-label, softmax değil)

### 4.2 Attention Pooling (Faz 3 iyileştirmesi)

Mean pooling tüm serileri eşit ağırlıklar. Ama meniskus için sagittal, MCL için coronal
daha bilgilendirici. **Target-specific attention:** her bulgu için ayrı attention ağırlığı
öğren.

```
Her etiket t için:
  α_t = softmax(w_t · seri_embeddings)   # bu etiket için hangi seri önemli
  z_t = Σ α_t,i · embedding_i
  logit_t = v_t · z_t
```

Bu, modelin "hangi bulgu için hangi görüntüye bakmalıyım"ı öğrenmesini sağlar.

### 4.3 Seri Metadata Enjeksiyonu

`Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` bilgisini modele ver:
- One-hot / küçük embedding olarak seri embedding'ine concat et
- Model "bu görüntü ne tür bir sekans" bilgisini sıfırdan öğrenmek zorunda kalmaz

### 4.4 Multi-View Füzyon Seçenekleri (Faz 3)

| Yaklaşım | Nasıl | Artı / Eksi |
|---|---|---|
| Erken füzyon | Farklı düzlemlerin slice'larını kanal olarak stack'le | Basit; ama düzlemler arası oryantasyon farkını yok sayar |
| **Geç füzyon (önerilen)** | Her düzlem ayrı forward pass → embedding concat/attention | Esnek, düzlem başına farklı çözünürlük mümkün |
| Cross-attention | Düzlemler arası transformer attention | En güçlü; en pahalı, overfit riski |

Başlangıç: **geç füzyon**. Baseline stabil olduktan sonra cross-attention denenebilir.

### 4.5 Backbone Adayları

| Backbone | Çözünürlük | Not |
|---|---|---|
| EfficientNet-B0 | 256 | Hızlı baseline, düşük bellek |
| EfficientNet-B3 / ConvNeXt-tiny | 320-384 | Orta seviye kapasite artışı |
| DINOv2-small (ViT-S/14) | 224 (14'ün katı) | Self-supervised pretrain, güçlü genel özellikler |

Ablation planı: B0 → (B3 veya ConvNeXt-tiny) → DINOv2-small. Her adımda CV skoru ve
eğitim süresini kaydet.

> **Offline kısıtı hatırlatması:** Tüm pretrained ağırlıklar önceden bir Kaggle Dataset'e
> yüklenmeli — submission notebook'unda internet yok.

---

## 5. Eğitim Yapılandırması

### Loss

```python
# Temel
criterion = nn.BCEWithLogitsLoss(reduction='none')  # per-element, ağırlıklandırma için

# Dengesizlik için seçenekler:
# 1. pos_weight (etiket başına ters frekans)
pos_weight = (n_neg / n_pos).clamp(max=20)  # aşırı değerleri sınırla
# 2. Focal loss (gamma=2 tipik başlangıç)
```

Multi-label olduğu için klasik oversampling zor uygulanır → **sample-level loss
ağırlıklandırması** tercih edilir (weak/gold ayrımı zaten bunu gerektiriyor).

### Optimizer & Schedule

- `AdamW`, lr ≈ 1e-4 (backbone için daha düşük, head için daha yüksek — discriminative lr)
- Weight decay ≈ 1e-2
- Scheduler: `CosineAnnealingLR` veya `OneCycleLR`
- Warmup: ilk 1-2 epoch

### Mixed Precision

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    out = model(x)
    loss = criterion(out, y)
scaler.scale(loss).backward()
scaler.step(optimizer); scaler.update()
```

Kaggle GPU süresi/belleği için pratikte zorunlu.

### Augmentation

MONAI transform'ları:
- Küçük açılı rotasyon (±10°)
- Random crop / random resized crop
- Intensite jitter (brightness/contrast)
- Gaussian noise (hafif)
- **Sol-sağ flip: KULLANMA.** Gerekçesi aşağıda — bu bir tercih değil, sert kural.

#### Sol-sağ flip augmentation YASAK (Faz 0.7 kararı)

12 etiketin **4'ü** medial/lateral ayrımıdır: `Medial Meniscus`, `Lateral Meniscus`,
`Medial OA`, `Lateral OA`. Bir dizde medial/lateral ayrımı, o dizin *hangi diz olduğuna*
bağlı bir sol-sağ ayrımıdır — sağ dizin medial tarafı, aynalandığında lateral tarafa düşer.

Rastgele sol-sağ flip uygulanırsa model bu 4 etikette medial ile lateral'i **tanım olarak**
ayırt edemez hale gelir → macro AUC'nin 4/12'si tavan 0.5. Metrik ağırlıksız ortalama
olduğu için bu tek başına toplam skorda ~0.17 kayıp demektir.

Bu yüzden lateralite normalizasyonu opsiyonel bir varyans azaltma tekniği **değil**,
bu 4 etiketin öğrenilebilirliğinin ön koşuludur. Tercih sıralaması:

**doğru aynalama > hiç aynalamama > rastgele flip**

- *Hiç aynalamama:* öğrenilebilir ama zor — model önce hangi diz olduğunu çıkarmalı,
  sonra bulguyu yerelleştirmeli.
- *Yanlış aynalama:* o 4 etikete geometrik etiket gürültüsü enjekte eder. Bu yüzden
  taraf tespiti heuristic'i %90'ın altında isabet veriyorsa **kapalı** tutulur
  (`dicom_io.detect_laterality(use_ipp_layer=...)`).

> Dikkat: "flip" burada **hasta x ekseni** boyunca yansımayı ifade eder. Bunun görüntüdeki
> karşılığı düzleme göre değişir (Bölüm 1.1): coronal/axial'de `axis=2`, sagittal'de
> `axis=0`. Sagittal bir hacme yatay flip uygulamak sol-sağ değil ön-arka eksenini çevirir —
> farklı ama yine anatomik olarak yanlış bir işlem.

Üst-alt (vertical) flip de kullanılmaz: diz MR'ında üst-alt ekseni anatomik olarak
anlamlıdır (femur üstte, tibia altta).

### Checkpoint & Resume (ÖNEMLİ)

Kaggle notebook oturumları kesilebilir. Her epoch sonunda:
- Model state, optimizer state, scaler state, epoch numarası, RNG state kaydet
- Kaldığı yerden devam edebilen bir eğitim döngüsü kur
- Ayrıca **süre korumalı** (time-guarded) olsun: kalan süreyi izleyip zaman dolmadan
  temiz şekilde checkpoint'e kaydedip çıksın

### Metrik Takibi

Her epoch sonunda validation'da:
- **Per-label ROC-AUC** (12 ayrı değer)
- **Macro ROC-AUC** (asıl metrik)
- Tahmin dağılımı std'si (aşağıdaki teşhis notuna bak)

`torchmetrics.classification.MultilabelAUROC(num_labels=12, average=None)` per-label verir.

---

## 6. Teşhis ve Sağlık Kontrolleri

Bu adımlar sessiz başarısızlıkları erken yakalar.

### 6.1 Tahmin dağılımı çöküşü

Model tüm örneklere neredeyse aynı skoru veriyorsa (per-label tahmin std'si çok düşük,
örn. <0.05), model gerçekte ayırt edici bir şey öğrenmemiş, sadece base rate'i öğrenmiş
demektir. Her epoch sonunda per-label tahmin std'sini logla.

### 6.2 Gruplu vs rastgele fold farkı

Bölüm 3'te anlatıldı. Fark büyükse CV'ye güven azalır, gruplu fold zorunlu hale gelir.

### 6.3 Gold-only validation

Weak-label'larla eğitirken bile, **validation'ı mutlaka gold örneklerde yap**. Weak-label
üzerinde validation, weak-labeler'ın hatalarını "doğru" sayar ve seni yanıltır.

### 6.4 Submission sanity check

- Sütun isimleri birebir doğru mu?
- Satır sayısı test study sayısıyla eşleşiyor mu?
- NaN / sonsuz değer var mı?
- Tüm değerler [0,1] aralığında mı?
- Baseline karşılaştırması: sample_submission (hepsi 0.5) AUC = 0.5 verir; modelin bunu
  net şekilde geçmeli

---

## 7. Ensemble Stratejisi

1. **Fold ensemble:** Her fold'un modelinin tahminlerini ortala (en basit, en güvenilir kazanç)
2. **Mimari ensemble:** EfficientNet + DINOv2 çıktılarını ağırlıklı ortala
3. **Stacking:** OOF (out-of-fold) tahminler üzerine per-label logistic regression meta-learner
4. **Rank averaging:** Metrik AUC olduğu için, ham olasılık yerine **rank** ortalaması
   genelde daha stabil sonuç verir (farklı modellerin kalibrasyonu farklı olabilir)

---

## 8. İleri Seviye Fikirler (Zaman kalırsa)

### 8.1 Contrastive pretraining (privileged information)

Rapor metnini test'te kullanamıyoruz. Ama **eğitim aşamasında** görüntü ve metin
embedding'lerini birbirine yaklaştıran CLIP-benzeri bir contrastive pretraining, görüntü
encoder'ının daha zengin temsiller öğrenmesine yardımcı olabilir.

Bu, "privileged information" tekniği olarak bilinir: yardımcı modalite sadece eğitimde
mevcut olsa da, ana modalitenin temsil kalitesini artırabilir.

**Ana plan öğesi değil** — baseline'lar oturduktan sonra denenecek bir iyileştirme.

### 8.2 Self-supervised pretraining

570 GB etiketsiz DICOM var. Domain-specific SSL (SimCLR/DINO tarzı) ile backbone'u knee MRI
dağılımına adapte etmek mümkün. Ancak compute maliyeti yüksek — Kaggle kotasıyla zor.
Düşük öncelik.

### 8.3 Slice-level attention (MIL)

Bir study'yi "slice bag"i olarak modelleyip Multiple Instance Learning yaklaşımı uygula.
Hangi slice'ın bulguyu içerdiğini model kendi öğrenir. 2.5D + attention pooling zaten bunun
basitleştirilmiş bir versiyonu.

---

## 9. Repo Yapısı Önerisi

```
rsna-knee/
├── PROJECT_OVERVIEW.md      # problem, kurallar, veri
├── ARCHITECTURE.md          # bu dosya
├── PHASES.md                # faz/görev listesi
├── EXPERIMENTS.md           # deney log'u (her koşu: config, CV, LB, süre)
├── src/
│   ├── data/
│   │   ├── dicom_io.py      # DICOM okuma, sıralama, normalizasyon
│   │   ├── series_select.py # metadata güdümlü seri seçimi
│   │   ├── dataset.py       # PyTorch Dataset/DataLoader
│   │   └── folds.py         # GroupKFold + site proxy
│   ├── labels/
│   │   ├── llm_extract.py   # Katman 1: LLM ile rapor→etiket
│   │   ├── distill.py       # Katman 2: XLM-R fine-tune
│   │   └── validate.py      # Katman 3: gold üzerinde doğrulama
│   ├── models/
│   │   ├── backbone.py      # timm wrapper
│   │   ├── pooling.py       # mean / attention pooling
│   │   └── head.py          # 12-label head
│   ├── train.py             # checkpoint/resume'lu eğitim döngüsü
│   ├── infer.py             # submission üretimi
│   └── metrics.py           # per-label + macro AUC, teşhis
├── notebooks/
│   ├── 00_env_dicom_basics.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_weak_labels.ipynb
│   ├── 03_preprocess_dataset.ipynb   # 570GB → küçültülmüş Kaggle Dataset
│   └── 04_train_baseline.ipynb
└── kaggle/
    └── submission_notebook.py        # offline çalışan final inference
```

**EXPERIMENTS.md disiplini:** Her eğitim koşusu için tek satır kaydet —
`tarih | config özeti | CV macro AUC | LB skoru | süre | not`.
2 kişi paralel çalıştığı için bu, neyin işe yaradığını izlemenin tek yolu.
