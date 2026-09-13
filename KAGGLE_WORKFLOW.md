# KAGGLE WORKFLOW — Çalışma Düzeni

> "Bunu yeni notebook'ta mı yapayım?" sorusunun cevabı burada. Her sorduğunda
> bakacağın yer.

---

## 1. Temel kural

**Bir notebook = bir iş = bir çıktı kümesi.**

Bir notebook'un ne ürettiğini tek cümleyle söyleyemiyorsan, muhtemelen ikiye bölünmeli.

Sebebi Kaggle'a özgü: `/kaggle/working` altındaki dosyalar ancak **Save Version** ile
kalıcı olur ve o sürüme bağlanır. İçeriği değiştirip yeniden commit edersen yeni sürüm
farklı bir işin çıktısını üretir; eski çıktılara ulaşmak için sürüm geçmişinde geriye
gitmen gerekir. Ayrıca bir notebook'un çıktısını başka bir notebook'a **input olarak**
bağlayabilirsin (Faz 2.3'te ön işlenmiş dataset tam olarak böyle çalışacak) — o kurguda
"bir notebook = bir iş" olmazsa bağımlılıklar çorbaya döner.

---

## 2. YENİ notebook aç

- [ ] **Farklı bir iş.** Envanter çıkarmak ile ön işleme yapmak ayrı işler.
      (`00_phase0_dicom_survey` vs `01_phase0_pipeline`)
- [ ] **Farklı çalışma zamanı tipi.** CPU analiz ile GPU eğitim ayrılmalı. GPU kotası
      haftalık ve sınırlı; CPU işini GPU oturumunda çalıştırmak kota yakar.
- [ ] **Çıktısını başka yerde kullanacaksan.** Ön işlenmiş dataset, model ağırlıkları,
      weak-label CSV'si → hepsi kendi notebook'unda üretilmeli ki input olarak bağlanabilsin.
- [ ] **Uzun süreli iş.** Saatler süren bir eğitimi, hızlı iterasyon yaptığın bir analiz
      notebook'una karıştırma. Küçük bir değişiklik için 6 saatlik commit beklemezsin.
- [ ] **Submission notebook'u.** Her zaman ayrı, her zaman internet kapalı test edilmiş.

## 3. MEVCUT notebook'u düzenle

- [ ] Aynı işin **hata düzeltmesi** (bizim `NotADirectoryError` düzeltmesi gibi)
- [ ] Aynı işin **parametre değişikliği** (`N_STUDIES` 500 → 200)
- [ ] Aynı işe **ek bir analiz bölümü** ekleme
- [ ] Çıktı formatını iyileştirme

Şüphedeysen: *"eski çıktıya bir daha bakacak mıyım?"* Evetse yeni notebook.

---

## 3b. Yeni notebook açtığında AYARLAR

Sağ paneldeki **Session options** / **Notebook options** bölümü. Import ettikten sonra
Run All'a basmadan önce bunları ayarla — yanlış ayar ya kota yakar ya da hata verir.

| Ayar | Değer | Neden |
|---|---|---|
| **Language** | Python | — |
| **Accelerator** | işe göre (aşağıdaki tablo) | GPU kotası haftalık ve sınırlı |
| **Internet** | işe göre (aşağıdaki tablo) | Submission'da KAPALI olmak zorunda |
| **Persistence** | No persistence | Çıktıyı `Save Version` kalıcılaştırır; persistence bayat değişken taşır ve "bende çalışıyordu" hatalarının kaynağıdır |
| **Environment** | Always use latest | Tek istisna submission notebook'u — orada sabitlemek tekrarlanabilirliği garantiler |

### İş tipine göre

| Notebook tipi | Accelerator | Internet | Not |
|---|---|---|---|
| Envanter / EDA / metadata denetimi | **None (CPU)** | Açık | Piksel okumuyor ya da az okuyor. GPU kotasını harcama |
| Ön işleme (DICOM → .npy) | **None (CPU)** | Açık | Darboğaz ağ I/O, GPU boşta bekler |
| LLM / weak-label çıkarımı | **GPU** | Açık | Model ağırlığı indirilecek |
| Model eğitimi | **GPU** | Açık | Pretrained ağırlık indirilecek |
| **Submission (inference)** | **GPU** | **KAPALI** | Yarışma kuralı. Kapalıyken test et, sona bırakma (Faz 2.15) |

### Kurallar

- **Şüphedeysen CPU seç.** GPU'ya ihtiyacın olduğunu `torch.cuda.is_available()` ile
  değil, işin ne olduğuna bakarak bil: matris çarpımı yoksa GPU gereksiz.
- **Internet'i submission dışında kapatma.** Kapalıyken `pip install` ve model indirme
  çalışmaz; gereksiz yere kendini kısıtlarsın.
- **Accelerator'ı çalışma ortasında değiştirme.** Kernel yeniden başlar, bellekteki
  her şey gider.

---

## 4. Repo'dan Kaggle'a akış

Kod repo'da yazılır, Kaggle'da çalışır. Sıra:

```
1. Kaynağı düzenle          notebooks/XX_isim.py   (# %% hücre işaretli)
2. Notebook üret            python tools/py2ipynb.py notebooks/XX_isim.py
3. Testleri çalıştır        python tests/test_dicom_io.py
4. Kaggle'a import et       New Notebook -> File -> Import Notebook -> .ipynb yükle
5. Veriyi bağla             + Add Input -> Competitions -> rsna-knee-abnormality-detection
6. Runtime seç              Accelerator: None (CPU) / GPU
7. Run All
8. Save Version -> Save & Run All (Commit)
9. Sonucu geri getir        Output sekmesinden CSV indir -> D:\RSNA Kaggle\
```

### Sık yapılan hata

**Import edilen notebook input'ları beraberinde getirmez.** 5. adımı atlarsan ilk
hücrede `FileNotFoundError` alırsın. Her import sonrası veriyi yeniden bağla.

### Modül güncellendiğinde

`src/data/dicom_io.py` değişirse, notebook'a **gömülü** kopya bayatlar. Sıra:

```
python tools/py2ipynb.py notebooks/01_phase0_pipeline.py   # yeniden göm
```

sonra `.ipynb`'yi Kaggle'a **tekrar import et**. Kaggle'daki hücreyi elle düzenleme —
repo ile notebook arasında sürüm ayrışması, sonradan bulması en zahmetli hata türüdür.

---

## 4b. Notebook teslim brifingi

Claude bir notebook hazırladığında, **sen çalıştırmadan önce** şunları yazmalı:

1. Ne yapıyor
2. Ne işe yarıyor — hangi karara girdi üretiyor
3. Ne üretiyor — çıktı dosyaları ve ekrandaki tablolar
4. Runtime ayarları (yukarıdaki tabloyla tutarlı)
5. Ne kadar sürer
6. Neye bakmalısın — hangi sayı hangi kararı değiştirir
7. Başarısızlık işaretleri — neyi görürsen durup haber vermelisin

Sebebi: bulutta her başlatma bir oturum (GPU ise kota) harcıyor. Ne beklediğini
bilmeden çalıştırmak, sonucu yorumlayamamak demek.

---

## 5. İsimlendirme

| Katman | Biçim | Örnek |
|---|---|---|
| Repo dosyası | `NN_konu.py` | `01_phase0_pipeline.py` |
| Kaggle notebook | `rsna-<konu>` | `rsna-phase0-pipeline` |
| Kaggle Dataset | `rsna-<içerik>` | `rsna-preprocessed-256` |

"Untitled Notebook (3)" ile üç hafta sonra karşılaşmak istemezsin.

---

## 6. Save Version'ın iki modu

| Mod | Ne yapar | Ne zaman |
|---|---|---|
| **Quick Save** | Kodu kaydeder, **çalıştırmaz**, mevcut çıktıyı saklar | Sadece kod yedeklemek |
| **Save & Run All** | Temiz bir ortamda baştan çalıştırır, çıktıyı kalıcılaştırır | Çıktı üretecekse — **varsayılan tercih** |

`Save & Run All` aynı zamanda bir doğrulamadır: notebook'un baştan sona, temiz bir
kernel'de çalıştığını kanıtlar. İnteraktif oturumda elle çalıştırdığın hücre sırası
yanıltıcı olabilir (silinmiş bir hücrenin değişkeni bellekte kalır).

---

## 7. Kota ve limit hatırlatmaları

- **GPU kotası haftalık ve sınırlı.** CPU ile yapılabilecek hiçbir işi GPU oturumunda
  çalıştırma. Envanter, EDA, metadata denetimi → hep CPU.
- **Yarışma kuralı: submission notebook'u ≤ 9 saat** (bkz. `PROJECT_OVERVIEW.md`).
  Bu sınır sadece submission'a ait; eğitim/deney notebook'ların için geçerli değil.
- **İnteraktif oturum boşta kalırsa kapanır.** Uzun işi `Save & Run All` ile arka planda
  çalıştır, tarayıcıyı kapatabilirsin.
- **`/kaggle/working` commit edilmeden kalıcı olmaz.** Oturum bitince uçar.
- **Günlük 5 submission.** Boşa harcama; her submission'ı bir hipotez testi olarak gör.

---

## 8. Faz bazlı notebook planı

| Notebook | İş | Runtime |
|---|---|---|
| `00_phase0_dicom_survey` | Transfer syntax + tag envanteri | CPU |
| `01_phase0_pipeline` | Ön işleme katmanı doğrulaması | CPU |
| `02_eda` | Faz 1A — etiket/dil/seri dağılımları | CPU |
| `03_weak_labels_llm` | Faz 1B Katman 1 — LLM çıkarımı | GPU |
| `04_weak_labels_distill` | Faz 1B Katman 2 — XLM-R | GPU |
| `05_preprocess_dataset` | Faz 2A — 570 GB → küçültülmüş dataset | CPU (I/O bound) |
| `06_train_baseline` | Faz 2B — eğitim | GPU |
| `07_submission` | Faz 2C — **internet KAPALI** inference | GPU |

`05` ve `06`'nın çıktıları Kaggle Dataset olarak yayınlanır ve sonraki notebook'lara
input olarak bağlanır.
