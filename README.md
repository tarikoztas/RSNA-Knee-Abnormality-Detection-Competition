# RSNA Knee Abnormality Detection

Diz MR çalışması başına **12 klinik bulgunun** olasılığını tahmin eden model —
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle yarışması için. Metrik: 12 hedefin macro-averaged ROC-AUC'si.

Yarışmanın en belirleyici zorluğu modelin kendisi değil, **etiket kıtlığı**: 4.407
study'nin yalnızca **58'i** (%1,3) gold etiketli. Geri kalanının elinde sadece 12 dilde
yazılmış serbest metin radyoloji raporu var. Bu depo, o raporlardan çok dilli bir LLM ile
yapılandırılmış zayıf etiket (weak label) üretip görüntü modelini onunla eğitme
yaklaşımını izliyor.

> Proje notları ve kod yorumları Türkçe.

---

## Durum

| Faz | Konu | Durum |
|---|---|---|
| 0 | DICOM okuma, slice sıralama, normalizasyon, lateralite, 2.5D stack | ✅ Tamam |
| 1A | EDA, dil dağılımı, site proxy, fold stratejisi | ✅ Tamam |
| 1B | Weak-label pipeline (LLM çıkarımı) | 🔄 Devam — prompt + ayrıştırıcı + bake-off hazır |
| 1C | GroupKFold split ve sızıntı teşhisi | ⬜ Sırada |
| 2 | Görüntü baseline ve ilk submission | ⬜ |
| 3+ | Model iyileştirme | ⬜ |

Ayrıntılı görev listesi: [`PHASES.md`](PHASES.md)

---

## Depoda ne YOK — ve neden

Bu depo bilinçli olarak **veri içermiyor**. `.gitignore` "veri varsayılan kapalı"
mantığıyla yazıldı; aşağıdakiler dışarıda tutuluyor:

| Dışarıda kalan | Neden |
|---|---|
| `outputs/train.csv`, `devset_50.csv`, `raw_*.jsonl` | **Ham radyoloji raporu metni içeriyor.** Kaggle kuralları yarışma verisinin yeniden dağıtımını yasaklar; ayrıca de-identified olsa da hasta metnidir. |
| `outputs/*_metadata.csv`, `folds.csv`, `laterality_*.csv` | Yarışma verisinden türetilmiş DICOM metadata ve UID listeleri. |
| `src/labels/devset_ids.py` | 87 benzersiz `StudyInstanceUID` — türetilmiş veri. `tools/make_devset.py` yeniden üretir. |
| `notebooks/*.ipynb` | **Üretilen dosya.** Kaynak `.py`'ler depoda; `tools/py2ipynb.py` çevirir. |
| `*.dcm`, model ağırlıkları, `kaggle.json`, `.env` | Veri seti 570 GB ve asla yerele inmez; sırlar hiç commit'lenmez. |

Depoda tutulan tek sayısal çıktı, **hasta düzeyinde hiçbir bilgi içermeyen** toplu
özetler: `label_distribution.csv`, `phase1a_summary.txt`, `bakeoff_*.csv`.

### `.ipynb` neden yok?

Notebook'lar `# %%` hücre işaretli `.py` dosyalarından üretiliyor. `tools/py2ipynb.py`
ayrıca `#!writefile` direktifini işleyip bir kaynak modülün **tüm içeriğini** bir
`%%writefile` hücresine gömüyor — böylece Kaggle'da ayrı bir dataset bağlamadan modül
diske yazılıyor. Pratik sonucu: üretilen `.ipynb`, `devset_ids.py`'nin kopyasını da
taşıyor. Kural bu yüzden basit tutuldu — **kaynağı sakla, türevi saklama.**

---

## Yeniden üretme

```bash
# Notebook üret (Kaggle'a bu .ipynb yüklenir)
python tools/py2ipynb.py notebooks/03_weak_labels_llm.py

# Dev set + gold ayrımını yeniden üret (train.csv erişimi gerekir)
python tools/make_devset.py

# Testler
python tests/test_dicom_io.py
python tests/test_prompt.py
```

Veri `/kaggle/input/` altında mount edilmiş olarak gelir — **yerele indirilmez.**
Ağır işler (DICOM okuma, LLM çıkarımı, eğitim) Kaggle Notebook'ta çalışır;
bu depodaki modüller oraya `%%writefile` ile taşınır.

### Bağımlılıklar

Çekirdek: `pydicom`, `numpy`, `pandas`, `opencv-python`, `matplotlib`, `Pillow`
Weak-label katmanı: `vllm`, `transformers`, `torch`, `huggingface_hub`, `fasttext`

Hepsi Kaggle imajında hazır ya da oturum başında kurulabilir. **Submission
notebook'unda internet kapalıdır** — o aşamada her paket ve ağırlık önceden bir Kaggle
Dataset'ine yüklenmiş olmalı.

---

## Yapı

```
src/
  data/
    dicom_io.py      # DICOM → tensor: slice sıralama, normalizasyon, lateralite, 2.5D stack
    folds.py         # Site proxy normalizasyonu + GroupKFold ataması
  labels/
    prompt.py        # 12 bulgu × {present/absent/uncertain} JSON şeması, prompt, ayrıştırıcı
    vllm_runner.py   # vLLM ile güdümlü (guided) decoding — sağlayıcıdan bağımsız tutuldu
    evaluate.py      # Weak-labeler'ı 58 gold study üzerinde ölç (per-label F1/AUC)
notebooks/           # `# %%` hücre işaretli .py kaynakları (Kaggle'da çalışan asıl kod)
tools/
  py2ipynb.py        # .py → .ipynb, `#!writefile` direktifi dahil
  make_devset.py     # Dev set / gold ayrımı üretici
  vllm_probe.py      # vLLM guided-decoding API'sini model yüklemeden tespit et
tests/               # Bağımlılıksız test script'leri (python tests/<dosya>.py)
```

## Dokümanlar

| Dosya | İçerik |
|---|---|
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | Yarışma kuralları, metrik, veri şeması, kısıtlar |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Teknik mimari ve tasarım kararlarının gerekçeleri |
| [`PHASES.md`](PHASES.md) | Faz planı ve görev listesi |
| [`KAGGLE_WORKFLOW.md`](KAGGLE_WORKFLOW.md) | "Yeni notebook mu açayım?" — çalışma düzeni |
| [`PHASE0_FINDINGS.md`](PHASE0_FINDINGS.md) · [`PHASE1A_FINDINGS.md`](PHASE1A_FINDINGS.md) | Ölçüm sonuçları ve çıkarımlar |

---

## Veri kullanımı

Yarışma verisi RSNA/Kaggle koşullarına tabidir ve bu depoda **yeniden dağıtılmaz**.
Kodu çalıştırmak için veriye [yarışma sayfasından](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
kendin erişmelisin.
