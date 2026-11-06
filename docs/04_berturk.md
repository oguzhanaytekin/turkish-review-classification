# T-04: BERTurk Fine-Tune

> Durum: kod yazıldı (`scripts/04_berturk.py`, `notebooks/04_berturk_colab.ipynb`); Colab'da çalıştırılmayı bekliyor.

## 1. Kararlar (2026-11-06)
1. **Model:** BERTurk cased (`dbmdz/bert-base-turkish-cased`). Ham metin verilecek (ön işleme yok);
   büyük/küçük harfi koruduğu için İ/I riski yok.
2. **Eğitim ortamı:** Google Colab (ücretsiz T4 GPU). Oturum kopabileceği için model her epoch
   sonunda Google Drive'a kaydedilecek; kopma olursa kaldığı yerden devam edilecek.
3. **Tekrar sayısı:** 1 seed. Fark baseline'a göre 1 puan civarında çıkarsa (sınırda) 2 seed daha
   eklenecek.
4. **Kod Colab'a GitHub'dan gelecek:** Repo public olacak; Colab repoyu indirip `02` betiğiyle
   işlenmiş veriyi yeniden üretecek (ham veri Hugging Face'ten iniyor).
5. **Karşılaştırma çıtası (T-03 baseline):** test macro-F1 0,803 · LLM alt kümesi 0,796 ·
   Negative recall 0,826 / precision 0,859 · ECE 0,033 · zor örneklerde cross-entropy 1,104.

## 2. Uygulama ayrıntıları
- **Hiperparametreler:** 3 epoch, öğrenme oranı 2e-5, batch 32, max_len 128, weight decay 0,01,
  ilk %6 adımda ısınma (warmup). BERT fine-tune'u için yaygın başlangıç değerleri; arama yapılmadı.
- **En iyi epoch:** Her epoch sonunda val macro-F1 ölçülür; en iyisi geri yüklenir (load_best_model_at_end).
- **Kalibrasyon:** Val setinde tek bir sıcaklık (T) öğrenilir. Tahmin edilen sınıf değişmez, sadece güven
  skorları düzelir. ECE ve zor örnek cross-entropy'si kalibrasyon öncesi ve sonrası raporlanır.
- **Hız:** Test setinde yorum başına çıkarım süresi (ms) ölçülür. LLM karşılaştırmasındaki "hız" ekseni için.
- **Güvenlik önlemleri:** Colab'da yeniden üretilen bölmenin sayıları T-02 ile karşılaştırılır; farklıysa betik
  durur. Önce 2–3 dakikalık bir smoke testi çalıştırılır.

## 3. Sonuçlar
<!-- Eğitim sonrası doldurulacak -->

## 4. Kararlar ve alternatifler
| Karar | Seçtiğimiz | Yerine | Alternatif ne zaman daha iyi olurdu? |
|---|---|---|---|
| Model | BERTurk cased | DistilBERTurk / Türkçe ELECTRA / uncased BERTurk | Distil: hız ve maliyet öncelikliyse (genelde 1–2 puan kayıp). ELECTRA/büyük modeller: GPU bütçesi varsa. Uncased: büyük/küçük harf bilgi taşımıyorsa, ama küçük harf tokenizer'a kalır |
| Ortam | Colab | Kaggle Notebooks | Uzun eğitimlerde: Kaggle arka planda çalışır, haftalık 30 saat GPU garanti |
| Seed | 1 | 3 | Modeller arası fark küçükse ve "bu fark gerçek mi?" sorusu önemliyse: sonuç "ortalama ± sapma" diye raporlanır |
| Eğitim kodu | Hugging Face `Trainer` | Elle yazılmış PyTorch eğitim döngüsü | Eğitimin her adımını kontrol etmek ya da öğrenmek istiyorsan (ör. özel kayıp fonksiyonu). Trainer; checkpoint, devam etme ve fp16'yı hazır veriyor |
| Kalibrasyon yöntemi | Temperature scaling (tek parametre) | Platt scaling / isotonic regression | Isotonic: çok fazla val verisi varsa ve hata deseni düzensizse. Temperature: az parametreli olduğu için aşırı uyum riski düşük, sınıf tahminini değiştirmez |
| max_len | 128 token | 256 / 512 | Uzun yorumların payı yüksekse. Yorumlarımızın medyanı ~10 kelime; 128 token neredeyse hepsini kapsıyor ve 2–4 kat daha hızlı |
| Epoch sayısı | 3 sabit + en iyi epoch'u seçme | Erken durdurma (early stopping) / daha fazla epoch | Veri küçükse ya da val skoru dalgalıysa; büyük veride BERT genelde 2–3 epoch'ta doyuyor |
| Hiperparametre arama | Yok (standart değerler) | Öğrenme oranı / batch araması | GPU bütçesi varsa; genelde 0,5–1 puan getirir. Önce baseline ile farkı görmek istiyoruz |
