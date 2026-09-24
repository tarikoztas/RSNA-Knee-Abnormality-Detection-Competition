"""Faz 2B — study duzeyinde coklu-seri, coklu-etiket model.

Mimari (ARCHITECTURE.md Bolum 4.1):

    Study
     └─ N seri (Sagittal / Coronal / Axial)
          └─ her seri -> 2.5D yigin (C kanal)
               └─ PAYLASILAN backbone -> seri embedding (D)
      seri embedding'leri -> havuzlama -> study embedding -> 12 sigmoid

Uc tasarim karari:

1. AGIRLIKLAR PAYLASILIR. Her duzlem icin ayri backbone egitmek parametreyi
   uce katlar ve her biri verinin ucte birini gorur. Paylasilan backbone tum
   veriden ogrenir; duzlem bilgisi ayrica embedding olarak veriliyor (3.3).

2. 12 BAGIMSIZ SIGMOID, softmax DEGIL. Bir dizde hem ACL yirtigi hem efuzyon
   olabilir — siniflar birbirini dislamiyor.

3. HAVUZLAMA DEGISTIRILEBILIR. Mean ile basliyoruz (Faz 2.6); attention
   (Faz 3.2) ayni arayuzu kullanacak sekilde yazildi, model degistirmeden
   takilabilir.
"""
import torch
import torch.nn as nn

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
PLANES = ["Sagittal", "Coronal", "Axial"]


class MeanPool(nn.Module):
    """Seri embedding'lerinin maskeli ortalamasi.

    Maske neden gerekli: bir study'de eksik duzlem olabilir (Faz 1A'da hepsi
    tamdi ama test setinde garanti yok). Eksik seriyi sifir vektorle doldurup
    duz ortalama almak embedding'i sulandirir; maskeli ortalama sadece GERCEK
    serileri sayar.
    """

    def forward(self, emb, mask):
        # emb: (B, S, D)   mask: (B, S) 1=gercek seri
        m = mask.unsqueeze(-1).to(emb.dtype)
        return (emb * m).sum(1) / m.sum(1).clamp(min=1.0)


class AttentionPool(nn.Module):
    """Etikete ozel attention havuzlama (Faz 3.2).

    Mean pooling tum serileri esit agirliklar. Ama menisküs icin sagittal,
    MCL icin coronal daha bilgilendirici. Burada her ETIKET kendi attention
    agirligini ogrenir: alpha_t = softmax(w_t . emb), z_t = sum(alpha_t * emb).

    Ciktisi (B, n_labels, D) — mean pooling'in (B, D) ciktisindan farkli,
    bu yuzden head iki durumu da ele aliyor.
    """

    def __init__(self, dim, n_labels=len(LABELS)):
        super().__init__()
        self.score = nn.Linear(dim, n_labels)

    def forward(self, emb, mask):
        s = self.score(emb)                                   # (B, S, L)
        s = s.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))
        a = torch.softmax(s, dim=1)                           # seriler uzerinde
        a = torch.nan_to_num(a)                               # tum seriler maskeliyse
        return torch.einsum("bsl,bsd->bld", a, emb)           # (B, L, D)


class KneeModel(nn.Module):
    """timm backbone + seri metadata + havuzlama + 12 sigmoid.

    Girdi: x (B, S, C, H, W)  — B study, S seri, C kanal (2.5D yigin)
           mask (B, S)        — 1 = gercek seri
           meta (B, S, F)     — seri metadata (opsiyonel, Faz 3.3)
    Cikti: logits (B, 12)     — sigmoid UYGULANMAZ; BCEWithLogitsLoss bekliyor
    """

    def __init__(self, backbone="efficientnet_b0", in_chans=5, pretrained=True,
                 pooling="mean", meta_dim=0, meta_emb=16, dropout=0.2,
                 n_labels=len(LABELS)):
        super().__init__()
        import timm
        # num_classes=0 -> siniflandirma katmani yok, havuzlanmis ozellik doner.
        # in_chans=C -> timm ilk conv agirligini kanal sayisina gore uyarlar
        # (pretrained agirliklari ortalayip cogaltarak), bu yuzden 2.5D yigini
        # dogrudan verebiliyoruz.
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          in_chans=in_chans, num_classes=0)
        d = self.backbone.num_features
        self.meta_dim = meta_dim
        if meta_dim:
            # Fluid_Sensitive / Fat_Suppression / duzlem one-hot gibi bilgiler.
            # Model "bu goruntu ne tur bir sekans" bilgisini sifirdan ogrenmek
            # zorunda kalmasin (ARCHITECTURE 4.3).
            self.meta = nn.Sequential(nn.Linear(meta_dim, meta_emb), nn.ReLU())
            d += meta_emb
        self.pooling_kind = pooling
        self.pool = MeanPool() if pooling == "mean" else AttentionPool(d, n_labels)
        self.drop = nn.Dropout(dropout)
        if pooling == "mean":
            self.head = nn.Linear(d, n_labels)
        else:
            # Etikete ozel havuzlamada her etiketin kendi vektoru var:
            # logit_t = v_t . z_t  -> (L, D) agirlik, satir bazinda carpim
            self.head = nn.Parameter(torch.zeros(n_labels, d))
            nn.init.normal_(self.head, std=0.02)
            self.bias = nn.Parameter(torch.zeros(n_labels))

    def forward(self, x, mask, meta=None):
        B, S = x.shape[:2]
        # Serileri batch boyutuna katlayip TEK forward pass — S kez cagirmaktan
        # cok daha hizli ve BatchNorm istatistikleri tum serilerden gelir.
        emb = self.backbone(x.flatten(0, 1)).view(B, S, -1)
        if self.meta_dim and meta is not None:
            emb = torch.cat([emb, self.meta(meta)], dim=-1)
        z = self.drop(self.pool(emb, mask))
        if self.pooling_kind == "mean":
            return self.head(z)                               # (B, L)
        return (z * self.head).sum(-1) + self.bias            # (B, L)


def masked_bce(logits, target, weight, pos_weight=None):
    """Agirlikli, maskeli BCE — Katman 4'un ciktisini dogrudan tuketir.

    target : (B, L) soft hedef [0,1]   (weak_labels.csv <etiket>_soft)
    weight : (B, L) ornek-etiket agirligi (<etiket>_w); 0 = MASKELI

    Neden ayri bir fonksiyon: agirlik SIFIR olan cift loss'a hic girmemeli.
    `reduction='mean'` kullanip agirlikla carpmak yanlis olur — sifirlar
    paydada kalir ve ogrenme sinyalini sulandirir. Payda GERCEKTEN katkida
    bulunan ciftlerin toplam agirligi olmali.
    """
    l = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight)
    w = weight.to(l.dtype)
    denom = w.sum().clamp(min=1e-8)
    return (l * w).sum() / denom
