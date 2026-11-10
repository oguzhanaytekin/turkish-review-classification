# T-05 LLM Raporu (otomatik üretildi)

## 1. Kurulum

Model: gemini-3.5-flash-lite | k=5 | batch=20 | sıcaklık 1.0 | few-shot: 15 örnek (train'den)
Bu çalıştırmadaki istekler: {'istek': 129, 'hata': 19, 'kota_bekleme': 0, 'girdi_token': 75544, 'cikti_token': 166421, 'sure_sn': 722.7}

## 2.1 zero: sınıf bazında ve karışıklık matrisi (LLM alt kümesi)

```
          precision  recall      f1  destek
Negative     0.7724   0.896  0.8296     500
Neutral      0.7920   0.518  0.6264     500
Positive     0.7943   0.942  0.8618     500
```
```
tahmin    Negative  Neutral  Positive
label                                
Negative       448       42        10
Neutral        129      259       112
Positive         3       26       471
```

## 2.2 few: sınıf bazında ve karışıklık matrisi (LLM alt kümesi)

```
          precision  recall      f1  destek
Negative     0.7816   0.902  0.8375     500
Neutral      0.7884   0.544  0.6438     500
Positive     0.8010   0.926  0.8590     500
```
```
tahmin    Negative  Neutral  Positive
label                                
Negative       451       39        10
Neutral        123      272       105
Positive         3       34       463
```

## 2.3 kural: sınıf bazında ve karışıklık matrisi (LLM alt kümesi)

```
          precision  recall      f1  destek
Negative     0.8071   0.862  0.8337     500
Neutral      0.6422   0.736  0.6859     500
Positive     0.9059   0.712  0.7973     500
```
```
tahmin    Negative  Neutral  Positive
label                                
Negative       431       64         5
Neutral        100      368        32
Positive         3      141       356
```

## 3. Üç istem türünün özeti (zero = örneksiz, few = rastgele örnekli, kural = kural açıkça yazılı)

```
                                  zero        few      kural
cevaplanan (test)            1500/1500  1500/1500  1500/1500
macro-F1                        0.7726     0.7801     0.7723
accuracy                        0.7853     0.7907       0.77
Negative recall                  0.896      0.902      0.862
Negative precision              0.7724     0.7816     0.8071
ECE (örnekleme)                 0.0378     0.0353     0.0458
ECE (sözel)                     0.0847     0.0748     0.0572
zor CE (örnekleme)              1.3337     1.3261      1.185
zor CE (sözel)                  1.3434     1.2608     1.0509
çoğunlukla uyum (zor)           0.6047     0.6188     0.6562
ort. güven test (örnekleme)     0.8232     0.8242     0.8158
ort. güven zor (örnekleme)      0.8252     0.8278     0.7992
ort. güven test (sözel)         0.7917     0.7808     0.7607
ort. güven zor (sözel)          0.8272     0.8042     0.7359
```

Zor örneklerde alt sınır (insanların kendi belirsizliği): 0.6402 | eşit olasılık: 1.0986

## 4. Üç yaklaşımın karşılaştırması (LLM alt kümesi, 1.500 yorum)

```
                           baseline  BERTurk  LLM zero  LLM few  LLM kural
macro-F1 (LLM alt kümesi)    0.7960   0.8311    0.7726   0.7801     0.7723
zor CE (en iyi olasılık)     1.1039   0.8795    1.3337   1.2608     1.0509
```

Not: 1.500 yorumda ±1,9 puanın altındaki macro-F1 farkları gürültü sınırında.

## 5. Zor örnekler: insanlar ve her istem türü (örnekleme olasılığı)

```
                        n_kopya  insan_Neg  insan_Neu  insan_Pos  zero_Neg  zero_Neu  zero_Pos  few_Neg  few_Neu  few_Pos  kural_Neg  kural_Neu  kural_Pos
text                                                                                                                                                      
idare eder                  532       0.15       0.85       0.00      0.08      0.85      0.08     0.08     0.85     0.08       0.08       0.85       0.08
Fena değil                  470       0.09       0.91       0.00      0.08      0.85      0.08     0.08     0.85     0.08       0.08       0.85       0.08
beğenmedim                  394       0.94       0.05       0.00      0.85      0.08      0.08     0.85     0.08     0.08       0.85       0.08       0.08
Güzel ürün                  284       0.08       0.81       0.11      0.08      0.08      0.85     0.08     0.08     0.85       0.08       0.85       0.08
güzel bir ürün              280       0.07       0.87       0.06      0.08      0.08      0.85     0.08     0.08     0.85       0.08       0.85       0.08
Uygun fiyat                 151       0.05       0.22       0.73      0.08      0.08      0.85     0.08     0.08     0.85       0.08       0.85       0.08
Hiç beğenmedim              151       0.98       0.02       0.00      0.85      0.08      0.08     0.85     0.08     0.08       0.85       0.08       0.08
gayet güzel                 126       0.11       0.70       0.19      0.08      0.08      0.85     0.08     0.08     0.85       0.08       0.85       0.08
Güzel beğendim              119       0.04       0.86       0.10      0.08      0.08      0.85     0.08     0.08     0.85       0.08       0.54       0.38
Kokusu çok güzel             94       0.11       0.59       0.31      0.08      0.08      0.85     0.08     0.08     0.85       0.08       0.54       0.38
Hiç bir işe yaramıyor…       84       0.96       0.04       0.00      0.85      0.08      0.08     0.85     0.08     0.08       0.85       0.08       0.08
Çok kalitesiz                78       0.99       0.01       0.00      0.85      0.08      0.08     0.85     0.08     0.08       0.85       0.08       0.08
```

## 6. Maliyet ve hız

Bu çalıştırmada: 129 istek, girdi 75,544 / çıktı 166,421 token.
Ücretli katman fiyatıyla ($0.3/$2.5 / 1M token) yaklaşık maliyet: $0.439 (ücretsiz katmanda 0).
Not: Önbellekten gelen cevaplar bu sayılara dahil değil; koşu birden çok güne bölündüyse toplam daha yüksek. Değerlendirilen yorum×istem sayısı: 7,647.
