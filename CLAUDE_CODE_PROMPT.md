# Claude Code Başlangıç Prompt'u

Aşağıdaki prompt'u Claude Code'a (Claude Desktop → Code sekmesi) ilk mesaj olarak yapıştır.
`PROJECT_OVERVIEW.md`, `ARCHITECTURE.md` ve `PHASES.md` dosyalarının aynı klasörde olduğundan
emin ol.

---

## Ana Başlangıç Prompt'u (kopyala-yapıştır)

```
RSNA Knee Abnormality Detection adlı bir Kaggle yarışması üzerinde çalışıyorum.
Bu klasörde üç referans dosyası var — başlamadan önce üçünü de oku:

- PROJECT_OVERVIEW.md — problem tanımı, yarışma kuralları, veri seti, kritik veri sorunları
- ARCHITECTURE.md — teknik mimari, model tasarımı, weak-label pipeline, teknoloji seçimleri
- PHASES.md — fazlara bölünmüş görev listesi ve öncelikler

Bağlamım:
- Yerel makinemde GPU yok ve 570 GB'lık veri seti için disk yok. Sadece i7 CPU var.
- Geliştirmeyi tamamen bulutta yapacağım: Kaggle Notebook (birincil, veri orada),
  Colab ve Lightning AI (ek GPU kotası için).
- Bu yüzden ürettiğin kodlar ya Kaggle/Colab notebook'unda çalışacak şekilde,
  ya da bu ortamlara kopyalanabilir modüler Python dosyaları olarak olmalı.
- DICOM ve tıbbi görüntüleme ile ilk kez çalışıyorum. Yeni kavramları açıklarken
  günlük hayattan analojiler kullanarak, sohbet havasında anlat.
- Framework: PyTorch.
- 2 kişilik bir takımız ama bağımsız çalışıyoruz, yani görevleri tek kişilik varsay.

Nasıl çalışmanı istiyorum:
- Faz sırasıyla ilerle (PHASES.md). Bir fazı bitirmeden diğerine geçme.
- Her adımda önce ne yapacağını kısaca açıkla, sonra kodu yaz.
- Kod yazarken neden o yaklaşımı seçtiğini de belirt — öğrenmek istiyorum,
  sadece çalışan kod değil.
- Bir şey belirsizse varsayım yapıp devam etme, bana sor.
- Kritik uyarıları (PHASES.md'deki "asla feda edilmeyecekler" listesi) atlama.

Faz 0 ile başlayalım: ortam kurulumu ve DICOM temelleri.
İlk olarak Faz 0.3 ve 0.4'ü (transfer syntax doğrulama + 86 allowlisted tag envanteri)
yapacak bir Kaggle Notebook kodu yaz.
```

---

## Faz Geçiş Prompt'ları

Her fazın başında kullanabileceğin kısa prompt'lar:

### Faz 1'e geçerken
```
Faz 0'ı tamamladık. Şu bulguları elde ettik:
[Faz 0'daki tag envanteri sonuçlarını buraya yapıştır — özellikle hangi
site-proxy alanları mevcut: Manufacturer, InstitutionName, MagneticFieldStrength vb.]

Şimdi Faz 1'e geçelim. Önce 1A (EDA) görevlerini yapan bir notebook yaz.
```

### Weak-label pipeline'a geçerken
```
EDA sonuçları:
- Gold-labeled study sayısı: [X]
- Etiketsiz study sayısı: [Y]
- Tespit edilen diller ve dağılımı: [liste]
- En nadir 3 etiket: [liste]

ARCHITECTURE.md Bölüm 2'deki weak-label pipeline'ının Katman 1'ini (LLM tabanlı
yapılandırılmış çıkarım) uygulayalım. Önce prompt tasarımını yapalım, sonra
küçük bir örneklemde (50 rapor) test edip iterasyonla iyileştirelim.
```

### Faz 2'ye geçerken
```
Weak-label pipeline hazır. Doğrulama sonuçları (gold üzerinde per-label F1):
[tabloyu yapıştır]

Faz 2'ye geçelim. Önce 2A: ön işlenmiş dataset oluşturma script'ini yaz
(DICOM → normalize → resize 256 → .npy → Kaggle Dataset).
```

### İlk submission öncesi
```
Baseline model eğitildi. Fold 0 CV macro AUC: [X]

Şimdi Faz 2C'ye geçelim — offline çalışan inference notebook'u.
PROJECT_OVERVIEW.md Bölüm 2'deki internet-kapalı kısıtlarına dikkat et:
pretrained ağırlıklar Kaggle Dataset'ten okunmalı, pip install çalışmaz.
```

---

## Yararlı Ara Prompt'lar

### Bir kavramı anlamadığında
```
[Kavram] hakkında daha fazla açıklama yapar mısın? DICOM/tıbbi görüntüleme
konusunda yeniyim, günlük hayattan bir analojiyle anlatabilirsen daha iyi olur.
```

### Deney sonucu geldiğinde
```
Şu deneyi çalıştırdım:
- Config: [özet]
- CV macro AUC: [X]
- LB skoru: [Y]
- Süre: [Z]

EXPERIMENTS.md'ye kaydet ve sonuca göre bir sonraki adımı öner.
CV ve LB arasında fark varsa nedenini tartışalım.
```

### Bir şey beklenenden kötü çalıştığında
```
[Problem açıklaması]

ARCHITECTURE.md Bölüm 6'daki teşhis kontrollerini (tahmin dağılımı çöküşü,
gruplu vs rastgele fold farkı, gold-only validation) uygulayarak nedeni
bulmaya çalışalım.
```

---

## Hatırlatmalar

Claude Code ile çalışırken sık unutulan noktalar:

1. **Uzun oturumlarda bağlam kaybı** — Claude Code uzun bir oturumda dosyaları
   yeniden okumaya ihtiyaç duyabilir. "PHASES.md'yi tekrar oku ve nerede kaldığımızı
   söyle" demek işe yarar.

2. **EXPERIMENTS.md'yi güncel tut** — her deney sonrası. İki kişi paralel
   çalıştığınız için bu, neyin denenip neyin denenmediğini bilmenin tek yolu.

3. **Offline submission testini erken yap** — Faz 2.15. Bu en yaygın son dakika
   felaket kaynağı: internet kapalıyken `pip install` veya model indirme çalışmaz.

4. **Gold-only validation** — weak-label üzerinde validation yaparsan kendi kendini
   kandırırsın. Bu kuralı Claude Code'a hatırlatmaktan çekinme.
