# T-05: LLM ile Sınıflandırma

> Durum: kod yazıldı (`scripts/05_llm.py`); önce pilot, sonra tam koşu.

## 1. Kararlar (2026-11-06)
1. **Model:** Gemini flash-lite, ücretsiz katman (API anahtarı `.env` içinde, repoya girmez).
   Varsayılan `gemini-3.5-flash-lite`. (İlk deneme `gemini-2.5-flash-lite` ileydi; Google bu modeli yeni
   kullanıcılara kapatmış, 404 hatası ve 3.5'e geçme önerisi geldi.) Pilot, kullanılabilir flash-lite modellerini listeler.
2. **Neden sohbetteki Claude değil?** (a) Bu sohbetteki model veri setini ve etiketleme kuralını
   konuşmalardan "biliyor"; onun zero-shot'ı gerçek bir zero-shot olmazdı. Bu, sızıntının LLM'deki karşılığı.
   (b) Model sürümü, sıcaklık ve sistem talimatı kontrol edilemez, yani deney tekrarlanamaz. (c) Olasılık alınamaz.
3. **İstem:** Zero-shot ve few-shot, ikisi birden. Zero-shot'ta etiketleme kuralından hiç bahsedilmez.
   Few-shot'ta **yalnızca train setinden**, sınıf başına 5 rastgele (elle seçilmemiş) kısa yorum gösterilir.
4. **Olasılık:** k kez örnekleme (sıcaklık 1) ile etiket sıklığı. Aynı cevaplardan sözel olasılık da alınır.
   Böylece "LLM'in söylediği güven" ile "örneklemedeki kararsızlığı" karşılaştırılabilir.
5. **Kota için toplu gönderim:** Bir istekte 20 yorum. Yorumların sırası ve hangi yorumlarla birlikte
   gönderileceği her örneklemede değişir. Pilot, toplu gönderimin sonucu bozup bozmadığını ölçer.
7. **Üçüncü istem: "kural" (2026-11-07'de eklendi).** Ara sonuçlarda few-shot, zero-shot'tan farksız çıktı
   (Neutral recall 0,518 → 0,544). Rastgele seçilen 15 örnek incelendiğinde kuralın örneklerde göründüğü
   ("güzeldiii" → Neutral) ama örneklerin birbiriyle çeliştiği görüldü ("... idare eder" → Positive).
   Bu yüzden kuralı örneklere bırakmak yerine açıkça yazan, örneksiz bir istem eklendi (`src/llm.py` → `RULE`).
   **Dürüstlük notu:** Kural T-01'de, bölme yapılmadan önce tüm veriye bakılarak çıkarıldı; istem, test
   dağılımından dolaylı bilgi taşıyor olabilir. Kuraldaki ifadeler genel kalıplardır, testten tek tek seçilmedi.
6. **Ölçüm yeri:** Testin 1.500'lük dengeli LLM alt kümesi ve zor örnekler (1.049). Baseline ve BERTurk'le
   aynı metrikler.

## 2. Uygulama ayrıntıları
- Her cevap `data/llm_cache/` altına yazılıyor. Kota dolarsa ya da bağlantı koparsa aynı komut kaldığı
  yerden devam ediyor; hiçbir istek iki kez atılmıyor.
- Dakikalık kota (429) gelince istenen süre kadar bekleniyor; günlük kota dolunca betik düzgünce duruyor.
- Bozuk JSON cevabı iki kez denenip hâlâ bozuksa o gruptaki yorumlar tek tek soruluyor.
- Örnekleme olasılığında toplamsal düzeltme var: (sayı + 0,5) / (k + 1,5). Yoksa hiç çıkmayan sınıfa 0
  olasılık düşer ve cross-entropy sonsuza gider.

## 3. Pilot sonuçları
<!-- Pilot çalıştırılınca doldurulacak -->

## 4. Sonuçlar
<!-- Tam koşu bitince doldurulacak -->

## 5. Kararlar ve alternatifler
| Karar | Seçtiğimiz | Yerine | Alternatif ne zaman daha iyi olurdu? |
|---|---|---|---|
| Model | Gemini flash-lite (ücretsiz) | Claude Haiku / GPT-4o-mini (birkaç sent) / büyük modeller ($5–15) | Kota sorun olursa ücretli küçük modeller; "model büyüdükçe fark kapanıyor mu?" sorusu için büyük model |
| Olasılık | Örnekleme sıklığı (+ sözel) | Logprobs | Logprobs destekleyen bir modelde: tek istekte gerçek olasılık, örneklemeden ucuz |
| Few-shot örnekleri | Train'den rastgele, sınıf başına 5 | Elle seçilmiş örnekler (ör. hafif övgü → Neutral) | Amaç "kuralı en iyi nasıl öğretiriz?" ise. Ama o zaman ölçülen şey bizim istem mühendisliğimiz olur |
| Gönderim | 20 yorum / istek | 1 yorum / istek | Kota bolsa: her yorum bağımsız değerlendirilir, komşu yorumların etkisi olmaz |
| Kural istemi | Kural açıkça yazılı, örnek yok | Kural + elle seçilmiş tutarlı örnekler | Kural tek başına yetmezse; ama elle seçim istem mühendisliğini ölçmeye kayar |
| İstem dili | Türkçe talimat, İngilizce etiket adları | Tamamen İngilizce istem | Model İngilizce talimatlarda belirgin şekilde daha iyiyse (pilotta denenebilir) |
