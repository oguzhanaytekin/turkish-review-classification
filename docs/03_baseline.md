# T-03: Baseline (TF-IDF + Lojistik Regresyon)

> Rakamlar `scripts/03_baseline.py` çıktısından (`reports/t03_rapor.md`) gelir.
> Kodda neyin değiştiği: `docs/KOD_GUNLUGU.md`.

## 1. Kararlar
1. **Metrikler:**
   - Ana metrik (model seçimi): **macro-F1**
   - İş metriği: **Negative recall ve precision** (problem tanımı: olumsuz yorumları ayıklamak)
   - Kalibrasyon: **ECE**
   - Zor örnekler: **yumuşak etiketle cross-entropy** (modelin olasılıkları insan dağılımına ne kadar yakın?)
2. **Ön işleme veriyle seçilecek (ablation):** {ham, hafif, tam} × {kelime, karakter n-gram}.
   - hafif = Türkçe küçük harf + boşluk temizliği; noktalama ve emoji korunur.
   - tam = T-01'deki `normalize_text`; noktalama ve emoji silinir.
   - Seçim **val** setinde yapılır. Test setinde yalnızca seçilen model, **bir kez** ölçülür.
3. **Türkçe karaktersiz yazım ("basarili"):** Önce yaygınlığı ölçülecek. Düzeltme (deasciification)
   ancak oran anlamlıysa sonraki adımda eklenecek. Karakter n-gram'ları bu sorunu kısmen
   kendiliğinden yumuşatıyor.
4. **Ön işleme yöntemin bir parçası:** Bu ablation yalnızca baseline için. BERTurk ve LLM ham
   metinle eğitildikleri için onlara ham metin verilecek.

## 2. Sonuçlar
<!-- Betik çalıştırılınca doldurulacak -->

## 3. Kararlar ve alternatifler
| Karar | Seçtiğimiz | Yerine | Alternatif ne zaman daha iyi olurdu? |
|---|---|---|---|
| Ana metrik | Macro-F1 | Accuracy / weighted-F1 | Sınıflar tam dengeliyse accuracy aynı bilgiyi verir ve anlatması daha kolaydır. Weighted-F1: gerçek dağılımda sık sınıfın önemi daha fazlaysa |
| İş metriği | Negative precision/recall | Tek bir eşikte F1 / maliyet tabanlı skor | Kaçırılan olumsuz yorumun ve yanlış alarmın parasal maliyeti biliniyorsa, maliyet fonksiyonu doğrudan optimize edilir |
| Seçim yeri | Val seti | Test seti / cross-validation | Test seti asla; sızıntıdır. CV: veri az ve model ucuzsa (baseline için yapılabilirdi, ama 6 kombinasyon × 5 kat = 30 eğitim) |
| Ön işleme | Veriyle seçim (ablation) | Tek bir yöntemi baştan seçmek | Zaman/hesap kısıtı varsa ve alan bilgisi güçlüyse |
| Küçük harf | Kendimiz (`lowercase=False`) | sklearn'ün varsayılanı (`lowercase=True`) | Hiçbir zaman, Türkçede: sklearn Python'un `lower()` fonksiyonunu kullanır ve İ/I'yı bozar |
| Temsil | Kelime + karakter n-gram karşılaştırması | Yalnızca kelime | İngilizce gibi eklemeli olmayan ve yazım hatası az olan dillerde kelime genelde yeterli |
| Model | Lojistik regresyon | Linear SVM / Naive Bayes / gradient boosting | SVM: biraz daha yüksek doğruluk ama olasılık vermez (kalibrasyon için ek adım gerekir). NB: çok küçük veride. Boosting: seyrek metin özelliklerinde genelde fayda sağlamaz |
| Hiperparametre | C=1 sabit | C için arama (ör. 0,25 / 1 / 4) | Ablation sonucu kesinleştikten sonra seçilen kombinasyon için yapılabilir; küçük bir kazanç beklenir |
| Final model | Yalnızca train'de eğitilmiş | Train + val'de yeniden eğitmek | Val setine sonradan ihtiyaç yoksa. Bizde val, kalibrasyon (temperature scaling) için ayrılıyor |
