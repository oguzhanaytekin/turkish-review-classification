# T-04 BERTurk Raporu (otomatik üretildi)

## 1. Kurulum

Model: dbmdz/bert-base-turkish-cased | GPU: Tesla T4 | seed 42 | epoch 3 | lr 2e-05 | batch 32 | max_len 128
Veri: train 109,305 / val 13,663 / test 13,664 / zor 1,049 | test kimlik özeti ec122295137a
128 token'dan uzun test yorumu (kesildi): 11 (%0.08) | medyan uzunluk 16 token
Eğitim süresi: 21.8 dk | çıkarım hızı: 1.37 ms/yorum

## 2. Epoch'lar (val)

```
       eval_loss  eval_macro_f1  eval_accuracy
epoch                                         
1.0       0.4170         0.8322         0.8362
2.0       0.4041         0.8413         0.8421
3.0       0.4299         0.8413         0.8431
```
En iyi checkpoint: /content/drive/MyDrive/trc_runs/berturk_seed42/checkpoints/checkpoint-10248

## 3. Test seti (bir kez)

ANA METRİK  macro-F1 : 0.8416
accuracy             : 0.8436
İŞ METRİĞİ  Negative recall    : 0.8610
            Negative precision : 0.8891
Öğrenilen sıcaklık T = 1.426  (>1: model aşırı emin, olasılıklar yumuşatıldı)
ECE      : ham 0.0565 -> kalibre 0.0109
log loss : ham 0.4275 -> kalibre 0.3987

Sınıf bazında:
```
          precision  recall      f1  destek
Negative     0.8891  0.8610  0.8748    4518
Neutral      0.7773  0.7522  0.7645    4407
Positive     0.8603  0.9120  0.8854    4739
```

Karışıklık matrisi (satır = gerçek, sütun = tahmin):
```
tahmin    Negative  Neutral  Positive
label                                
Negative      3890      557        71
Neutral        461     3315       631
Positive        24      393      4322
```

LLM alt kümesinde macro-F1: 0.8311

## 4. Zor örnekler

```
alt sınır (insanların kendi belirsizliği)    0.6402
BERTurk (ham olasılık)                       0.9901
BERTurk (kalibre)                            0.8795
her şeye eşit olasılık (1/3)                 1.0986
```

Çoğunluk etiketiyle uyum: 0.7969
Ortalama güven (ham): test 0.900 | zor 0.765
Ortalama güven (kalibre): test 0.848 | zor 0.692

En kalabalık 10 zor örnek (model = kalibre olasılık):
```
                  n_kopya  insan_Neg  model_Neg  insan_Neu  model_Neu  insan_Pos  model_Pos
text                                                                                       
idare eder            532       0.15       0.22       0.85       0.76       0.00       0.01
Fena değil            470       0.09       0.14       0.91       0.85       0.00       0.01
beğenmedim            394       0.94       0.89       0.05       0.10       0.00       0.01
Güzel ürün            284       0.08       0.14       0.81       0.78       0.11       0.08
güzel bir ürün        280       0.07       0.11       0.87       0.75       0.06       0.14
Uygun fiyat           151       0.05       0.04       0.22       0.22       0.73       0.74
Hiç beğenmedim        151       0.98       0.97       0.02       0.03       0.00       0.01
gayet güzel           126       0.11       0.26       0.70       0.56       0.19       0.18
Güzel beğendim        119       0.04       0.13       0.86       0.73       0.10       0.14
Kokusu çok güzel       94       0.11       0.04       0.59       0.35       0.31       0.61
```

Eşitlik olan 5 örnek:
```
                          n_kopya  insan_Neg  model_Neg  insan_Neu  model_Neu  insan_Pos  model_Pos
text                                                                                               
Süper bir ürün.                16       0.12       0.07       0.44       0.22       0.44       0.71
çok güzel tavsiye ederim       14       0.14       0.21       0.43       0.21       0.43       0.57
Çok güzel  ürün                14       0.00       0.20       0.50       0.37       0.50       0.44
Kokusu çok hoş                 12       0.00       0.04       0.50       0.43       0.50       0.53
Çok beğenemedim                10       0.50       0.34       0.50       0.65       0.00       0.01
```

## 5. Baseline (T-03) ile karşılaştırma

```
                                          baseline  BERTurk    fark
test macro-F1                               0.8032   0.8416  0.0384
test accuracy                               0.8050   0.8436  0.0386
Negative recall                             0.8263   0.8610  0.0347
Negative precision                          0.8586   0.8891  0.0305
ECE (kalibrasyonsuz)                        0.0325   0.0565  0.0240
LLM alt kümesi macro-F1                     0.7960   0.8311  0.0351
zor örnek cross-entropy (kalibrasyonsuz)    1.1039   0.9901 -0.1138
```

Not: Test setinde ±0,6 puanın, LLM alt kümesinde ±1,9 puanın altındaki farklar gürültü sınırında.
