### 2026-10-31 — T-01: Veri keşfi
- Ne yaptım: TRSAv1'i inceledim (150 bin yorum, 3 sınıf × 50 bin). Tekrarları ve
  çelişkili etiketleri ölçtüm: %8,2 tekrar, 1.049 metin farklı sınıflarla etiketlenmiş.
- Ne öğrendim: Python'un lower() fonksiyonu Türkçeyi bilmiyor ("I"→"i", "İ"→"i"+görünmez nokta).
  Bu yüzden aynı yorumlar farklı metin sayılıyordu; kendi fonksiyonumu yazdım ve testlerle doğruladım.
  Tekrarları train/test bölmesinden önce atmanın sebebi overfitting değil, data leakage:
  aynı yorum iki tarafta olursa test skoru şişer ve overfitting'i göremem.
  Makaleden etiketlerin elle verildiğini buldum; "güzel ürün" gibi hafif övgülerin
  sistematik olarak Neutral etiketlenmesi bir etiketleme kuralına işaret ediyor.
- Aldığım karar: Temizlikte C senaryosu (çoğunluk etiketi). Elle etiketlenmiş veride çoğunluk oyu
  ortak karara en yakın etiketi verir ve zor örnekleri atmaz. Çelişkili metinler ayrıca saklanacak.