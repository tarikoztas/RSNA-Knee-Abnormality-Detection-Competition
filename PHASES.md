# PHASES — Faz Planı ve Görev Listesi

> Problem tanımı: `PROJECT_OVERVIEW.md` · Teknik mimari: `ARCHITECTURE.md`
>
> **Kalan süre:** ~6.5 hafta (bugün 6 Eylül 2026 → final deadline 22 Ekim 2026)
> **Öncelik:** Önce çalışan bir baseline ve süreci öğrenmek, sonra ana leaderboard skorunu
> iyileştirmek. Efficiency track öncelikli değil.

---

## Faz 0 — Ortam ve DICOM Temelleri
**Süre:** 2-3 gün (Hafta 1 başı)
**Amaç:** Veriyi doğru okuyabildiğinden emin olmak. Bu adım atlanırsa sonraki her şeyin
hata ayıklaması zorlaşır.

### Görevler

- [x] **0.1** Kaggle / Colab / Lightning AI hesaplarını hazırla, GPU kotalarını not et
- [x] **0.2** Kaggle Notebook'ta veri yolunu doğrula: `/kaggle/input/...` altındaki yapıyı gez
- [x] **0.3** `pydicom` ile tek bir seriyi oku; **dört transfer syntax'ın da** okunabildiğini
      doğrula (uncompressed, JPEG Lossless, JPEG 2000, Implicit VR).
      Gerekiyorsa `pylibjpeg` / `gdcm` plugin'lerini kur
- [x] **0.4** **86 allowlisted DICOM tag'inin envanterini çıkar** — hangi alanlar gerçekten
      mevcut? Özellikle şunları ara: `Manufacturer`, `ManufacturerModelName`,
      `MagneticFieldStrength`, `InstitutionName`, `StationName`, `Laterality`,
      `ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`,
      `RescaleSlope/Intercept`
      → **Bu envanter Faz 1'deki fold stratejisini belirleyecek, kritik.**
- [x] **0.5** Slice sıralama fonksiyonunu yaz ve doğrula
      (`ImagePositionPatient` + normal vektör, fallback `InstanceNumber`)
- [x] **0.6** Intensite normalizasyonu fonksiyonunu yaz (percentile clip + min-max)
- [x] **0.7** Laterality tespiti ve mirror fonksiyonu
      → Aynalama ekseni düzleme göre değişiyor (sagittal=slice sırası); `lr_flip_axis`.
      Katman 1+2 kapsam %65 / isabet %97.9. **Katman 3 (x-işareti) askıda** —
      merkez-x düzeltmesi yapıldı, yeni isabeti `02_eda`'da ölçülecek.
      **Sol-sağ flip augmentation YASAK** (4 etiket medial/lateral ayrımı)
- [x] **0.8** Birkaç study'yi görselleştir (matplotlib grid) — farklı düzlem ve sekansların
      neye benzediğine dair gözle sezgi kazan
- [x] **0.9** 2.5D stack oluşturma fonksiyonunu yaz ve görsel olarak doğrula

### Çıktı — TAMAMLANDI (12 Eylül 2026)
`src/data/dicom_io.py` + `tests/test_dicom_io.py` (hepsi geçiyor) +
`notebooks/00_phase0_dicom_survey` (envanter) + `notebooks/01_phase0_pipeline`
(görsel/sayısal doğrulama). Bulgular: **`PHASE0_FINDINGS.md`**.

**Faz 1'e taşınan iki açık iş:**
- Lateralite katman 3'ün yeni isabeti + **study düzeyi kapsam** ölçümü → `02_eda`
- Gold etiketli study sayısı sadece **58** (%1.32) çıktı → validation stratejisi
  kararı Faz 1'e girerken verilecek (`PHASE0_FINDINGS.md` Bölüm 3)

---

## Faz 1 — EDA ve Weak-Label Pipeline
**Süre:** Hafta 1 (kalanı) + Hafta 2 başı
**Amaç:** Bu yarışmanın en kritik problemini çözmek: eksik etiketleri raporlardan üretmek.

### 1A — Keşifsel Veri Analizi

- [x] **1.1** `train.csv`: kaç study gold-labeled, kaçı etiketsiz? Oranı çıkar
- [x] **1.2** Gold alt kümede **per-label pozitif/negatif dağılımı** — hangi bulgular nadir?
- [x] **1.3** Rapor dili tespiti (`fasttext lid.176` veya `langdetect`) → dil başına study sayısı
- [x] **1.4** Rapor uzunluk dağılımı (çok kısa/telegrafik raporlar extraction'ı zorlaştırır)
- [x] **1.5** Bulgular arası ko-okürans matrisi (örn. Effusion ↔ Synovitis birlikte mi görülüyor)
- [x] **1.6** `train_series.csv`: study başına seri sayısı, düzlem dağılımı,
      Fluid_Sensitive / Fat_Suppression kombinasyonlarının frekansı
- [x] **1.7** Faz 0.4'teki tag envanterinden **site proxy** adaylarını değerlendir:
      hangi kombinasyon anlamlı gruplar üretiyor?

### 1B — Weak-Label Pipeline

> Detaylı tasarım: `ARCHITECTURE.md` Bölüm 2

      → Öneri: `lang_g + mfr_ailesi` (22 grup). Üretici adı eşleme tablosu
      ZORUNLU — 11 ham ad 5 aileye iniyor. Bkz. `PHASE1A_FINDINGS.md` Böl. 3
- [~] **1.8** **Katman 1:** Multilingual LLM ile yapılandırılmış çıkarım
      → Prompt + ayrıştırıcı + bake-off notebook'u HAZIR (`03_weak_labels_llm`).
      Karar: ÜCRETSİZ açık ağırlıklı model (Kaggle GPU), vLLM + JSON güdümlü decoding.
      Güven skoru modele sorulmuyor, logprob'dan türetiliyor. Bake-off bekleniyor.
      - Model seçimi (Qwen2.5 / Llama-3.1 ailesi, GPU belleğine göre)
      - Prompt tasarımı: 12 bulgu × {present/absent/uncertain} + confidence, JSON çıktı
      - Negation talimatı açıkça prompt'ta olmalı
      - Few-shot örnekler (farklı dillerden)
      - Küçük bir örneklem (50-100 rapor) üzerinde manuel gözden geçirerek prompt'u iterasyonla
      - Tüm raporlarda batch inference
- [ ] **1.9** **Katman 2:** XLM-RoBERTa distillation
      - LLM çıktısı + gold veriyi birleştir
      - 12-etiketli multi-label fine-tuning
      - Soft target kullan (LLM confidence'ı)
- [ ] **1.10** **Katman 3:** Gold alt kümede K-fold doğrulama
      - Per-label precision / recall / F1 / ROC-AUC tablosu üret
      - **Bu tablo Faz 2'deki loss ağırlıklandırmasını belirleyecek**
- [ ] **1.11** **Katman 4:** Confidence-weighted soft label dosyası üret
      - Her study için 12 soft label + güven ağırlığı
      - Gold örnekler işaretli kalsın (validation için ayrılacak)

### 1C — Fold Stratejisi

- [ ] **1.12** Site proxy'yi belirle (DICOM metadata ve/veya rapor dili)
- [ ] **1.13** `GroupKFold` ile 5-fold split üret, fold ataması dosyaya kaydet
- [ ] **1.14** **Teşhis:** Aynı basit modeli rastgele ve gruplu fold ile eğitip CV farkını ölç
      → Sızıntının büyüklüğünü nicel olarak bil

### Çıktı
- `weak_labels.csv` (study başına 12 soft label + confidence)
- `folds.csv` (study → fold ataması)
- Weak-labeler performans tablosu (per-label F1/AUC)

---

## Faz 2 — Görüntü Baseline ve İlk Submission
**Süre:** Hafta 2 (kalanı) + Hafta 3 başı
**Amaç:** Uçtan uca çalışan bir pipeline. Skor önemli değil, **süreç** önemli.

### 2A — Ön İşlenmiş Dataset

- [ ] **2.1** Metadata güdümlü seri seçimi mantığını uygula
      (her düzlemden en az bir seri, fluid-sensitive öncelikli)
- [ ] **2.2** Preprocessing script'i: DICOM → normalize → resize (256px) → `.npy`
- [ ] **2.3** Tüm train setini işle, sonucu **Kaggle Dataset olarak yayınla** (private)
      → 570 GB'ı birkaç GB'a indir, her oturumda yeniden decode etmeyi bitir
- [ ] **2.4** Aynı işlemi test seti için de yapabilen bir versiyon (submission'da kullanılacak)

### 2B — Model ve Eğitim

- [ ] **2.5** `timm` EfficientNet-B0 + 2.5D input (`in_chans` ayarlı) backbone wrapper
- [ ] **2.6** Seri embedding → mean pooling → study embedding → 12-sigmoid head
- [ ] **2.7** Dataset/DataLoader: study başına seri yükleme, augmentation
- [ ] **2.8** Loss: weighted BCE (gold ağırlığı > weak ağırlığı, per-label pos_weight)
- [ ] **2.9** **Checkpoint/resume + time-guard'lı** eğitim döngüsü
- [ ] **2.10** Metrik: per-label + macro ROC-AUC, **validation sadece gold örneklerde**
- [ ] **2.11** Fold 0'da eğit, CV skorunu al

### 2C — İlk Submission (KRİTİK DOĞRULAMA)

- [ ] **2.12** Pretrained ağırlıkları Kaggle Dataset'e yükle (offline erişim için)
- [ ] **2.13** Gerekli paketleri offline kurulabilir hale getir (Kaggle imajında yoksa)
- [ ] **2.14** Inference notebook'u yaz: test DICOM → preprocessing → model → `submission.csv`
- [ ] **2.15** **İnternet KAPALI olarak** notebook'u çalıştır ve submit et
      → Bu adım en yaygın hata kaynağıdır, erken test et
- [ ] **2.16** Submission sanity check (sütun isimleri, satır sayısı, NaN, [0,1] aralığı)
- [ ] **2.17** Runtime'ı ölç: ~1300 study için ne kadar sürüyor? 9 saat sınırına oranla?

### Çıktı
Leaderboard'da bir skor. Pipeline uçtan uca doğrulanmış.

---

## Faz 3 — Model İyileştirme
**Süre:** Hafta 3-4
**Amaç:** Baseline'ı sistematik olarak iyileştirmek. Her değişikliği ayrı ölç.

- [ ] **3.1** 5-fold tam eğitim + fold ensemble → ilk gerçek CV/LB karşılaştırması
- [ ] **3.2** **Attention pooling** (mean pooling yerine) — target-specific tercih edilir
- [ ] **3.3** Seri metadata enjeksiyonu (Fluid_Sensitive / Fat_Suppression / Plane embedding)
- [ ] **3.4** Multi-view geç füzyon (sagittal + coronal + axial ayrı forward pass)
- [ ] **3.5** Backbone ablation: B0 → ConvNeXt-tiny veya EfficientNet-B3 → DINOv2-small
- [ ] **3.6** Çözünürlük ablation (256 → 384)
- [ ] **3.7** Loss ablation: pos_weight vs focal loss
- [ ] **3.8** Weak-label ağırlık ablation: gold/weak oranı, confidence eşiği
- [ ] **3.9** Augmentation ablation

> **Disiplin:** Her deneyi `EXPERIMENTS.md`'ye kaydet:
> `tarih | config | CV macro AUC | LB | süre | not`
> Aynı anda birden fazla değişken değiştirme — neyin işe yaradığını ayırt edemezsin.

---

## Faz 4 — Ensemble ve İnce Ayar
**Süre:** Hafta 5

- [ ] **4.1** En iyi 2-3 mimarinin fold ensemble'ı
- [ ] **4.2** Rank averaging vs olasılık ortalaması karşılaştırması
- [ ] **4.3** OOF tahminler üzerine per-label stacking (logistic regression meta-learner)
- [ ] **4.4** Per-label analiz: hangi bulgular zayıf? Onlara özel iyileştirme denemesi
      (macro ortalama olduğu için en zayıf etiketi iyileştirmek en yüksek getiriyi sağlar)
- [ ] **4.5** Tahmin dağılımı teşhisi (per-label std kontrolü)
- [ ] **4.6** Inference runtime optimizasyonu (9 saat sınırı güvenli mi?)

---

## Faz 5 — Toparlama ve Final
**Süre:** Hafta 6 (15-22 Ekim)

- [ ] **5.1** **15 Ekim:** Entry deadline — kuralların kabul edildiğinden emin ol
      (takım birleşmesi düşünülüyorsa da son gün)
- [ ] **5.2** Final 2 submission'ı seç
      → Sadece public LB'ye bakma; kendi CV skoruna da ağırlık ver
      (prevalans farkı garantisi yok, public LB'ye overfit riskli)
- [ ] **5.3** Kod temizliği ve tekrarlanabilirlik kontrolü
- [ ] **5.4** Model ağırlıklarını Kaggle Dataset olarak yayınla
- [ ] **5.5** Son kez runtime ve submission format doğrulaması
- [ ] **5.6** (Ödül alınırsa) 5 Kasım'a kadar: training kod + video + metod açıklaması

---

## Öncelik Sıralaması (zaman daralırsa)

Zaman yetmezse şu sırayla feda et:

1. ~~Faz 3.6-3.9 ablation'ları~~ (nice-to-have)
2. ~~Faz 4.3 stacking~~ (fold ensemble zaten çoğu kazancı verir)
3. ~~Multi-view füzyon~~ (tek düzlem + attention pooling makul sonuç verir)

**Asla feda edilmeyecekler:**
- Faz 0.4 (tag envanteri) — fold stratejisinin temeli
- Faz 1B (weak-label pipeline) — verinin çoğunu kullanabilmenin tek yolu
- Faz 1C (gruplu fold) — CV'nin güvenilirliği buna bağlı
- Faz 2.15 (offline submission testi) — en yaygın son dakika felaketi
- Gold-only validation — yanlış yapılırsa tüm CV anlamsızlaşır
