# T-01: TRSAv1 Veri Keşfi ve Tekrar Analizi

> Durum: taslak. Rakamlar `scripts/01_veri_kesfi.py` çıktısından (`reports/t01_rapor.md`) gelir,
> yorumlar ve kararlar bu dokümanda toplanır.

## 1. Veri seti
- Kaynak: Hugging Face, `maydogan/Turkish_SentimentAnalysis_TRSAv1` (tek dosya: `TRSAv1.csv`, ~16 MB)
- Makale: Aydoğan M, Kocaman V. *TRSAv1: A new benchmark dataset for classifying user reviews
  on Turkish e-commerce websites.* Journal of Information Science, 2022. doi:10.1177/01655515221074328
- Lisans: Veri kartında lisans alanı **yok**. Kart yalnızca ilgili çalışmaya atıf yapılarak
  metin sınıflandırma ve duygu analizi çalışmalarında kullanılabileceğini söylüyor.
  - Sonuç: portfolyo/araştırma amaçlı kullanım + atıf uygun; ticari kullanım belirsiz.

## 2. Temel tablo
<!-- Satır sayısı, boş değerler, sınıf dağılımı, uzunluk dağılımı -->

## 3. Tekrar analizi
<!-- Birebir aynı metinler / normalleştirilmiş tekrarlar.
     Türkçe küçük harf sorunu nasıl çözüldü? (bkz. src/text_utils.py) -->

## 4. Çelişkili etiketler
<!-- Aynı metin, farklı sınıf: kaç grup, en çok hangi sınıf çiftleri arasında? -->

## 5. Temizlik sonrası sınıf dengesi
<!-- Üç senaryonun karşılaştırması -->

## 6. Örnekleri okuyunca
<!-- Her sınıftan 5 örnek: etiketler mantıklı mı? Nötr sınıf neye benziyor? -->

## 7. Önerilen temizlik kuralı
<!-- Karar + gerekçe -->

## Açık sorular
-
