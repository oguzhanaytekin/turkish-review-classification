# T-05 Ara Rapor (otomatik üretildi) — model gemini-3.5-flash-lite, k=5, batch=20

## 1. İlerleme

test/zero: 375/375 istek (100%)  TAMAMLANDI
test/few: 375/375 istek (100%)  TAMAMLANDI
test/kural: 375/375 istek (100%)  TAMAMLANDI
zor/zero: 265/265 istek (100%)  TAMAMLANDI
zor/few: 265/265 istek (100%)  TAMAMLANDI
zor/kural: 136/265 istek (51%)

## 2. Tamamlanan işlerin sonuçları


### test / zero-shot (k=5)
Cevaplanan: 1500/1500 | macro-F1 0.7726 | accuracy 0.7853 | Negative recall 0.8960 / precision 0.7724
ECE: örnekleme 0.0378 | sözel 0.0847 | ortalama güven: örnekleme 0.823, sözel 0.792
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
LLM alt kümesinde macro-F1 (±1,9 puan gürültü sınırı):
```
baseline    0.7960
BERTurk     0.8311
LLM zero    0.7726
```

### test / few-shot (k=5)
Cevaplanan: 1500/1500 | macro-F1 0.7801 | accuracy 0.7907 | Negative recall 0.9020 / precision 0.7816
ECE: örnekleme 0.0353 | sözel 0.0748 | ortalama güven: örnekleme 0.824, sözel 0.781
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
LLM alt kümesinde macro-F1 (±1,9 puan gürültü sınırı):
```
baseline    0.7960
BERTurk     0.8311
LLM few     0.7801
```

### test / kural-shot (k=5)
Cevaplanan: 1500/1500 | macro-F1 0.7723 | accuracy 0.7700 | Negative recall 0.8620 / precision 0.8071
ECE: örnekleme 0.0458 | sözel 0.0572 | ortalama güven: örnekleme 0.816, sözel 0.761
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
LLM alt kümesinde macro-F1 (±1,9 puan gürültü sınırı):
```
baseline     0.7960
BERTurk      0.8311
LLM kural    0.7723
```

### zor / zero-shot (k=5)
Zor CE: örnekleme 1.3337 | sözel 1.3434 | alt sınır 0.6402 | eşit 1.0986

### zor / few-shot (k=5)
Zor CE: örnekleme 1.3261 | sözel 1.2608 | alt sınır 0.6402 | eşit 1.0986
