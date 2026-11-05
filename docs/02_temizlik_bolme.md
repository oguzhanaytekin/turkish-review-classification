# T-02: Temizlik, Zor Örnekler ve Bölme

> Rakamlar `scripts/02_temizlik_bolme.py` çıktısından (`reports/t02_rapor.md`) gelir.
> Kodda neyin değiştiği: `docs/KOD_GUNLUGU.md`.

## 1. Kararlar
1. **Zor örnekler (sentez):** Birden fazla etiket almış 1.049 metin ana veriden (train/val/test)
   tamamen çıkarılıp ayrı bir set olarak saklanıyor. Her biri için etiketleyici dağılımı
   (yumuşak etiket, ör. "güzel ürün": %10 Negative / %79 Neutral / %11 Positive) ve çoğunluk
   etiketi tutuluyor. Bu set yalnızca değerlendirmede kullanılacak: Modellerin tahmin ettiği
   olasılıklar insanların dağılımıyla karşılaştırılacak.
2. **Bölme:** Stratified 80/10/10, seed 42.
3. **LLM alt kümesi:** Testten 1.500 yorum (sınıf başına 500). Bu boyutta skor belirsizliği
   yaklaşık ±1,9 puan; tüm test setinde yaklaşık ±0,6 puan.

## 2. T-01'deki C kararı nasıl değişti?
T-01'de C (çoğunluk etiketi) seçilmişti. T-02'de zor örnekleri ana veriden tamamen çıkarma
kararı alınınca ana veri, pratikte B senaryosuna (çelişkileri at) dönüştü. B'ye itirazımız
"zor örnekleri attığı için test skorunu şişirir" idi. Zor örnekler artık ayrı bir sette
ölçüleceği için bu itiraz ortadan kalkıyor. C'nin çoğunluk etiketi ise zor örnekler setinde
`cogunluk_etiketi` kolonunda kullanılmaya devam ediyor.

## 3. Sonuçlar
<!-- Betik çalıştırılınca doldurulacak. Beklenen: ana veri ~136.600, zor örnekler 1.049 (409 eşitlik) -->

## 4. Kararlar ve alternatifler
| Karar | Seçtiğimiz | Yerine | Alternatif ne zaman daha iyi olurdu? |
|---|---|---|---|
| Zor örneklerin yeri | Ayrı set, eğitimde yok | Ana veride kalıp işaretlenmesi | Amaç ürüne en yakın modeli çıkarmaksa ve sık geçen belirsiz ifadelerin eğitimde görülmesi isteniyorsa |
| Yumuşak etiket | Değerlendirmede: model olasılıkları ile insan dağılımının karşılaştırılması | Eğitimde yumuşak etiket | Gürültü verinin büyük kısmındaysa, ör. her yorum 5 kişi tarafından etiketlenmişse. Bizde yalnızca %0,8 |
| Kalibrasyon | Eğitim sonrası düzeltme (temperature scaling) + güvenilirlik grafiği (T-03 ve sonrası) | Yumuşak etiketle eğitim | Model eğitilirken belirsizliği doğrudan öğrenmesi isteniyorsa |
| Bölme oranı | 80/10/10 | 70/15/15 · 90/5/5 | 70/15/15: veri az, test güvenilirliği öncelikliyse. 90/5/5: milyonlarca satır varsa |
| Doğrulama | Sabit val seti | K-fold cross-validation | Veri az ve model eğitimi ucuzsa (ör. yalnızca baseline için) |
| Bölme yöntemi | Stratified rastgele | Zamana göre bölme | Yorumların tarihi olsaydı ve model gelecekteki yorumlarda kullanılacak olsaydı (bizde tarih kolonu yok) |
| LLM değerlendirmesi | 1.500'lük sabit alt küme | Testin tamamı (~13.700) | Bütçe yetiyorsa; belirsizlik ±1,9 puandan ±0,6 puana iner |
| Grubu temsil eden metin | Grubun ilk geçen ham metni | En sık görülen yazım biçimi | Yazım biçimi bilgi taşıyorsa (ör. tamamı büyük harf = öfke); bizde etkisi küçük |

## 5. Bilinen sınırlamalar
- Yazım farkı olan neredeyse aynı metinler ("hiç bir" / "hiçbir") farklı parçalara düşebilir.
  Küçük bir sızıntı riski olarak kalıyor.
- Tekilleştirme sonrası sınıflar tam eşit değil; Neutral en az temsil edilen sınıf. Stratified
  bölme bu oranları her parçada koruyor.
