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
### 2026-11-05 — T-02: Temizlik, zor örnekler ve bölme
- Ne yaptım: Çelişkili 1.049 metni ana veriden ayırıp "zor örnekler" setine aldım; kalan
  136.632 tekil metni stratified 80/10/10 böldüm (train 109.305 / val 13.663 / test 13.664).
  Testten LLM için 1.500'lük dengeli bir alt küme ayırdım. Sızıntı kontrolü geçti.
- Ne öğrendim: En sık tekrar eden kısa ifadeler ("idare eder", "güzel ürün") aynı zamanda
  en çok anlaşmazlık yaratanlar. Anlaşmazlığın iki türü var: gerçek belirsizlik ("rengi çok
  koyu") ve etiketleme kuralı + gürültü ("Süper bir ürün." 7 Neutral / 7 Positive / 2 Negative).
  Neutral ile "belirsiz" aynı şey değil: Neutral bir duygu sınıfı, belirsizlik ise
  etiketleyicilerin anlaşamaması. Zor örneklerin çoğunluk etiketi en çok Neutral çıktı (tahminim tuttu).
- Aldığım karar: Yumuşak etiketi eğitimde değil değerlendirmede kullanmak (sentez). Önce c'yi
  seçmiştim, ama yumuşak etiket verinin yalnızca %0,8'ine uygulanabildiği için etkisi küçük
  kalacaktı. Bölmede 80/10/10: belirsizlik 70/15/15'ten sadece 0,1 puan fazla, train ise daha büyük.
  Ayrıca kararların tek kaynağının repodaki dokümanlar olması gerektiğini öğrendim:
  dosyaya yazılmayan karar, terminaldeki Claude için hiç verilmemiş sayılıyor.

### 2026-11-05 — T-03: Baseline (TF-IDF + lojistik regresyon)
- Ne yaptım: 3 ön işleme (ham/hafif/tam) × 2 temsil (kelime/karakter) = 6 modeli val setinde
  karşılaştırdım, kazananı (tam + kelime) testte bir kez ölçtüm: macro-F1 0,803,
  Negative recall 0,826 / precision 0,859, ECE 0,033. "basarili" gibi yazımlar yalnızca %1,6
  çıktı, düzeltmeye gerek görmedim.
- Tahminlerim: Ham + karakter kazanır dedim, dördüncü oldu. Zor örneklerde cross-entropy 1/3
  yaklaşımından iyi olur dedim, kıl payı kötü çıktı (1,104'e karşı 1,099).
- Ne öğrendim: Karakter n-gram'ları (char_wb) kelime sınırında kesildiği için "fena değil",
  "tavsiye etmiyorum" gibi iki kelimelik olumsuzlamaları göremiyor; kelime 2-gram'ları görüyor.
  sklearn'ün TfidfVectorizer'ı varsayılan olarak Python lower() kullanıyor ve İ/I hatasını
  sessizce geri getiriyor; lowercase=False ile kapattık.
  Aşırı güven ile data leakage farklı şeyler: leakage veri hazırlama hatası, aşırı güven
  kalibrasyon sorunu. Model ortalamada iyi kalibre (düşük ECE) ama belirsiz örneklerde
  belirsizliği sezmiyor: "Süper bir ürün." için insanlar yarı yarıya bölünmüşken model %92 Positive diyor.
  Ortalama metrikler bunu gizliyor; zor örnekleri ayrı ölçmenin değeri burada.
- Aldığım karar: Ana metrik macro-F1, iş metriği Negative precision/recall. Baseline'ı daha fazla
  iyileştirmek yerine BERTurk'e geçmek; baseline'ın görevi çıta koymak.
