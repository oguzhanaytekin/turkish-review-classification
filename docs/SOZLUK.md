# Proje Sözlüğü

Terimler İngilizce bırakıldı (ilanlarda ve mülakatlarda böyle geçiyor); her birinin yanında Türkçe
açıklama ve **bu projedeki karşılığı** var. Unuttuğun bir terim olursa önce buraya bak, burada yoksa
araştır ve ekle.

---

## 1. Veri hazırlama

| Terim | Türkçe | Ne demek? | Bu projede |
|---|---|---|---|
| **Train / val / test** | Eğitim / doğrulama / test seti | Model train'de öğrenir; ayarlar ve seçimler val'de yapılır; test yalnızca en sonda, bir kez ölçülür | 109.305 / 13.663 / 13.664 yorum (T-02) |
| **Data leakage** | Veri sızıntısı | Modelin testte, eğitimde görmemesi gereken bir bilgiyi görmüş olması. Test skoru gerçekte olduğundan iyi görünür | Aynı yorum hem train'de hem test'te olsaydı. Bu yüzden bölmeden önce tekrarları attık (T-01, T-02) |
| **Stratified split** | Tabakalı bölme | Bölerken sınıf oranlarını her parçada aynı tutmak | Her parçada %33,1 Negative / %32,3 Neutral / %34,7 Positive |
| **Seed** | Tohum değeri | Rastgele işlemlerin her çalıştırmada aynı sonucu vermesini sağlayan sayı | Hep 42. Colab'daki bölme bu sayede yereldekiyle birebir aynı çıktı |
| **Normalization** | Normalleştirme | Aynı anlama gelen farklı yazımları tek biçime indirmek | "Güzel......", "güzel ☺️" → "güzel" (`normalize_text`) |
| **Deduplication** | Tekilleştirme | Tekrar eden kayıtları tek kayda indirmek | %8,2 tekrar bulundu (T-01) |
| **Label noise** | Etiket gürültüsü | Yanlış ya da tutarsız etiketler | "Süper bir ürün." 2 kez Negative etiketlenmiş |
| **Soft label** | Yumuşak etiket | Tek bir sınıf yerine olasılık dağılımı olarak verilen etiket | "Güzel ürün": %8 Neg / %81 Neu / %11 Pos |
| **Hard examples** | Zor örnekler | Bizim projemizde: etiketleyicilerin anlaşamadığı metinler | 1.049 metin, eğitimde hiç kullanılmadı, sadece değerlendirmede |
| **Ablation** | (Parça parça deneme) | Bir sistemin parçalarını değiştirip her parçanın etkisini ölçmek | 3 ön işleme × 2 temsil = 6 model (T-03) |

## 2. Modeller

| Terim | Türkçe | Ne demek? | Bu projede |
|---|---|---|---|
| **Baseline** | Temel model / çıta | Basit, hızlı bir model; daha karmaşık modellerin geçmesi gereken çıta | TF-IDF + lojistik regresyon, macro-F1 0,803 |
| **TF-IDF** | Terim sıklığı × ters belge sıklığı | Bir kelimenin o yorumda sık, ama tüm yorumlarda nadir olmasına göre ağırlık vermek | "beğenmedim" yüksek ağırlık alır, "ve" düşük |
| **n-gram** | n'li parça | Art arda gelen n kelime ya da n harf | Kelime 2-gram'ı "fena değil"; karakter 3-gram'ı "güz", "üze" |
| **char_wb** | Kelime sınırlı karakter n-gram | Harf parçaları kelime sınırını aşmaz | "fena değil"i göremediği için kelimeye yenildi (T-03) |
| **Logistic regression** | Lojistik regresyon | Özelliklere ağırlık verip olasılık üreten doğrusal sınıflandırıcı | Baseline modeli |
| **Fine-tune** | İnce ayar | Önceden eğitilmiş bir modeli kendi görevimiz için biraz daha eğitmek | BERTurk'ü 3 sınıflı duygu görevine uyarladık (T-04) |
| **BERT / BERTurk** | — | Metni bağlamıyla birlikte okuyan transformer modeli; BERTurk Türkçe metinle önceden eğitilmiş sürümü | `dbmdz/bert-base-turkish-cased` |
| **Cased / uncased** | Büyük-küçük harf duyarlı / duyarsız | Cased model "İyi" ile "iyi"yi ayırır | Cased seçtik: İ/I riski yok |
| **Token / tokenizer** | Parça / parçalayıcı | Modelin okuduğu birimler (kelime ya da kelime parçası) ve metni bunlara bölen araç | Medyan yorum 16 token |
| **max_len / truncation** | En fazla uzunluk / kesme | Bu uzunluktan uzun metinler kesilir | 128 token; kesilen yorum %0,08 |
| **Epoch** | Tur | Eğitim verisinin tamamının bir kez modelden geçmesi | 3 epoch; 2.'de doydu |
| **Batch** | Yığın | Model güncellenmeden önce birlikte işlenen örnek sayısı | 32 yorum |
| **Learning rate** | Öğrenme oranı | Her güncellemede ağırlıkların ne kadar değişeceği | 2e-5 (0,00002) |
| **Warmup** | Isınma | Eğitimin başında öğrenme oranını sıfırdan yavaşça artırmak | İlk %6 adım |
| **Checkpoint** | Ara kayıt | Eğitimin o anki halinin diske kaydedilmesi; kopunca devam etmeye yarar | Her epoch sonunda Drive'a |
| **Overfitting** | Aşırı uyum / ezberleme | Eğitim verisini ezberleyip yeni veride kötüleşmek | 3. epoch'ta val loss arttı: ilk işaret |
| **fp16** | Yarı hassasiyet | Sayıları daha az bitle tutarak GPU'da hızlanmak | Eğitim 21,8 dk |
| **Logit** | Ham skor | Modelin softmax'tan önceki ham çıktısı; olasılık değil | Sıcaklık logit'lere uygulanıyor |
| **Softmax** | — | Ham skorları toplamı 1 olan olasılıklara çeviren fonksiyon | `src/calibration.py` |

## 3. Metrikler

| Terim | Türkçe | Ne demek? | Bu projede |
|---|---|---|---|
| **Accuracy** | Doğruluk | Doğru tahminlerin oranı | BERTurk 0,844 |
| **Precision** | Kesinlik | "Olumsuz" dediklerimizin ne kadarı gerçekten olumsuz? | Negative: baseline 0,859 → BERTurk 0,889 |
| **Recall** | Duyarlılık / yakalama oranı | Gerçek olumsuzların ne kadarını yakaladık? | Negative: baseline 0,826 → BERTurk 0,861 |
| **F1** | — | Precision ve recall'un dengeli ortalaması (harmonik ortalama) | Neutral F1: 0,72 → 0,76 |
| **Macro-F1** | Makro F1 | Her sınıfın F1'inin eşit ağırlıklı ortalaması; sınıf dengesinden etkilenmez | **Ana metrik.** 0,803 → 0,842 |
| **Confusion matrix** | Karışıklık matrisi | Hangi sınıfın hangisiyle karıştırıldığını gösteren tablo | Negative→Positive: 149 → 71 |
| **Log loss / cross-entropy** | Çapraz entropi | Olasılıkların gerçeğe ne kadar uzak olduğu; emin olup yanılmayı ağır cezalandırır | Zor örneklerde: baseline 1,10, BERTurk kalibre 0,88 |
| **Entropy** | Entropi | Bir dağılımın belirsizliği | İnsanların kendi entropisi 0,64 = cross-entropy'nin alt sınırı |
| **Calibration** | Kalibrasyon | "%80 eminim" diyen modelin gerçekten %80 oranında haklı olması | — |
| **ECE** | Beklenen kalibrasyon hatası | Güven ile doğruluk arasındaki ortalama fark; 0 = mükemmel | Baseline 0,033 · BERTurk ham 0,057 → kalibre 0,011 |
| **Overconfidence** | Aşırı güven | Modelin haklı olduğundan daha emin konuşması. Sızıntıyla karıştırılmamalı | BERTurk: güven 0,90, doğruluk 0,84 |
| **Temperature scaling** | Sıcaklıkla ölçekleme | Logit'leri T'ye bölerek olasılıkları yumuşatmak (T>1) ya da keskinleştirmek (T<1). Tahmin edilen sınıf değişmez | T = 1,43 (val'de öğrenildi) |
| **Gürültü sınırı** | (Güven aralığı) | Bu kadar küçük farklar şansla da oluşabilir | Test ±0,6 puan, LLM alt kümesi ±1,9 puan |

## 4. Sıradaki aşamalar için

| Terim | Türkçe | Ne demek? |
|---|---|---|
| **Zero-shot** | Örneksiz | LLM'e hiç örnek göstermeden sadece talimatla görevi yaptırmak |
| **Few-shot** | Az örnekli | Talimatla birlikte birkaç etiketli örnek göstermek |
| **Prompt** | İstem | LLM'e verilen talimat metni |
| **McNemar testi** | — | İki modelin aynı örneklerdeki hatalarını karşılaştırıp farkın anlamlı olup olmadığını söyleyen istatistik testi |
| **API anahtarı** | — | Bir servisi kod içinden kullanmak için kimlik. Şifre gibi saklanır: `.env` dosyasında, git'e girmez |
| **Rate limit / kota** | Hız sınırı | Dakikada / günde atılabilecek istek sayısı. Aşınca 429 hatası gelir |
| **Batching** | Toplu gönderim | Birden çok yorumu tek istekte göndermek. Bizde 20 yorum / istek |
| **Örnekleme (sampling)** | — | Aynı soruyu sıcaklık > 0 ile k kez sormak; cevapların dağılımı modelin kararsızlığını gösterir |
| **Sözel olasılık (verbalized confidence)** | — | LLM'e "olasılıkları yaz" deyip aldığımız sayılar. Modelin gerçek iç olasılığı değil, söylediği bir şey |
| **Logprobs** | Log-olasılıklar | Modelin ürettiği token'lara verdiği gerçek olasılıkların logaritması; her API sunmaz |
| **Contamination** | Kirlenme | Değerlendirilen modelin test verisini ya da kuralını önceden görmüş olması. Sızıntının LLM'deki karşılığı |
