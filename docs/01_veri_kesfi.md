# T-01: TRSAv1 Veri Keşfi ve Tekrar Analizi

## 1. Veri seti
- Kaynak: Hugging Face, maydogan/Turkish_SentimentAnalysis_TRSAv1 (TRSAv1.csv)
- Makale: Aydoğan M, Kocaman V. TRSAv1: A new benchmark dataset for classifying user
  reviews on Turkish e-commerce websites. Journal of Information Science, 2022.
  doi:10.1177/01655515221074328
- Lisans: Veri kartında lisans alanı yok; atıf yapılarak kullanılabileceği belirtiliyor.
  Portfolyo/araştırma amaçlı kullanım uygun, ticari kullanım belirsiz.
- Etiketleme: Yorumlar, araştırmacıların geliştirdiği web tabanlı bir etiketleme aracı
  (DLT) üzerinden okunarak elle Positive / Negative / Neutral olarak etiketlenmiş.
- Kapsam: Örneklere bakılırsa yorumların büyük kısmı kozmetik/kişisel bakım ürünlerine
  ait. Başka ürün kategorilerine genelleme belirsiz.

## 2. Temel tablo
- 150.000 satır, boş değer yok, 3 sınıf × 50.000.
- Medyan uzunluk: Positive 10 kelime, Negative ve Neutral 8 kelime.
- 2 kelime veya daha kısa yorum: Neutral 3.947, Negative 3.228, Positive 747.

## 3. Tekrar analizi
- Birebir tekrar: 9.020 fazlalık satır (%6,0).
- Normalleştirilmiş tekrar (Türkçe küçük harf + noktalama/emoji silme): 12.259 (%8,2).
  Normalleştirme 3.239 ek tekrar yakaladı (ör. "güzel" kelimesinin 27 farklı yazımı).
- Python'un lower() fonksiyonu Türkçe I/İ'yi yanlış çeviriyor; src/text_utils.py
  içinde düzeltildi ve testlerle doğrulandı.
- 60 yorum normalleştirince tamamen boşalıyor (yalnızca noktalama/emoji).
- Bilinen sınırlama: "hiç bir" / "hiçbir" gibi yazım farkları ve Türkçe karakter
  kullanılmadan yazılmış yorumlar ("basarili") ayrı metin sayılıyor.

## 4. Çelişkili etiketler
- 1.049 normalleştirilmiş metin birden fazla sınıfla etiketlenmiş (9.593 satır, %6,4).
- Çelişkilerin yaklaşık %85'i komşu sınıflar arasında: Negative+Neutral 497,
  Neutral+Positive 390. Negative+Positive yalnızca 39.
- Çelişkili metinler kısa: medyan 3 kelime (tüm metinlerde 10).
- Sistematik gürültü: Hafif övgü çoğunlukla Neutral ("güzel ürün" 230 Neutral / 30 Positive).
  Pekiştirme gelince Positive'e kayıyor ("çok güzel bir ürün" 35 Positive / 30 Neutral).
- Rastgele gürültü: "beğenmedim" 372 Negative / 21 Neutral gibi küçük sapmalar.

## 5. Temizlik senaryoları
| Senaryo | Toplam | min/max |
|---|---|---|
| Ham | 149.940 | 0,999 |
| A: tekilleştir, ilkini tut | 137.681 | 0,935 |
| B: çelişkileri at + tekilleştir | 136.632 | 0,930 |
| C: çoğunluk etiketi | 137.272 | 0,933 |

## 6. Örnekleri okuyunca
- Bazı Neutral örnekler okununca olumsuz görünüyor ("Ürün çok minik olduğundan iade ettim").

## 7. Karar
- Senaryo C: Normalleştirilmiş metne göre tekilleştir, çoğunluk etiketini kullan,
  eşitlik varsa at.
- Gerekçe: Elle etiketlenmiş veride çoğunluk oyu, etiketleyicilerin ortak kararına en
  yakın etiketi verir. Zor örnekleri atmaz (B'nin aksine), etiketi rastgele seçmez
  (A'nın aksine).
- Çelişkili 1.049 metin (eşitlik yüzünden atılan 409 dahil) ayrı bir "zor örnekler"
  setinde saklanacak; modeller bu sette ayrıca değerlendirilecek.
- Tekrarlar train/test bölmesinden önce atılıyor; aynı yorum iki tarafta olursa test
  skoru şişer (data leakage).
- **T-02 güncellemesi:** Zor örnekler ana veriden tamamen çıkarılıp ayrı bir set yapıldı ve
  yumuşak etiketleri değerlendirmede kullanılacak. Ayrıntı: docs/02_temizlik_bolme.md.

## Kararlar ve alternatifler
| Karar | Seçtiğimiz | Yerine | Alternatif ne zaman daha iyi olurdu? |
|---|---|---|---|
| Veri seti | TRSAv1: dengeli 3 sınıf, yayınlanmış referans skor | Turkish Product Reviews (235 bin, 2 sınıf, %94 olumlu) | Lisans önemliyse (CC-BY-SA, ticari kullanım net) ya da dengesiz sınıflandırma çalışılıyorsa |
| Tekrar tespiti | Normalleştirerek: küçük harf + noktalama silme | Birebir eşleşme / bulanık eşleşme | Birebir: veri zaten temizse. Bulanık: çok yazım hatası veya şablon metin varsa; ama farklı anlamlı metinleri birleştirme riski taşır |
| Türkçe küçük harf | Kendi fonksiyonumuz (2 harf eşleme + test) | PyICU gibi bir kütüphane | Birden çok dille çalışılıyorsa ya da karmaşık harf kuralları varsa |
| Çelişkili etiketler | C: çoğunluk etiketi | A: ilkini tut / B: çelişkileri at | A: çelişki çok azsa. B: etiket saflığı her şeyden önemliyse; ama zor örnekler ayrı ölçülmeli |
| Sızıntıyı önleme | Bölmeden önce tekilleştirme | Tekrarları tutup aynı metni hep aynı parçaya koymak (group split) | Tekrar sıklığı bilgi taşıyorsa (ör. modelin "idare eder"in çok sık geçtiğini bilmesi isteniyorsa) |

## LLM karşılaştırması için çıkarım
- Zero-shot LLM, "güzel ürün" gibi hafif övgülere büyük ihtimalle Positive diyecek ve
  bu, veri setinin kuralına göre hata sayılacak. Few-shot örneklerle bu kuralın LLM'e
  aktarılıp aktarılamadığı ölçülecek.

## Açık sorular
- Makalede Neutral sınıfının tanımı / etiketleme yönergesi var mı?
- Kaç etiketleyici çalışmış, etiketleyiciler arası uyum (ör. Cohen's kappa) raporlanmış mı?
- 50.000 / 50.000 / 50.000'lik tam denge, etiketlemeden sonra örnekleme yapıldığını mı gösteriyor?
