# T-03 Baseline Raporu: TF-IDF + Lojistik Regresyon (otomatik üretildi)

## 1. Türkçe karaktersiz yazım ne kadar yaygın? (train)

5+ kelimelik yorumlardan hiç Türkçe karakter içermeyen: 1,532 / 95,587 (%1.60)
Sık kelimelerin ASCII biçimini içeren yorum (guzel, cok, urun...): 1,104 (%1.01)
En sık ASCII biçimler: gercekten: 435, yumusak: 167, basarili: 165, kullanisli: 154, boyle: 94, tesekkur: 77, tatli: 37, begendik: 9, begendim: 2, begenmedim: 1
Örnek Türkçe karaktersiz uzun yorumlar:
  'Harika bir renk dokusu efsane.'
  'gerek yok ama surekli kullanimda bilemem'
  'ust kismi biraz acitiyo alicaksanz daha yuwarlak bise alin bu yuvarlak tel gibi degl tercihinizi on…'
  'maalesef benim seboreik dermatit ime iyi gelmedi'
  'Fondoteni biraz emiyor fakat bu fiyata iyi'

## 2. Ablation: ön işleme × temsil (VAL setinde)

```
  on_isleme    temsil  val_macro_f1  val_accuracy  ozellik_sayisi  sure_sn
0       tam    kelime        0.8106        0.8121          133362       15
1     hafif    kelime        0.8104        0.8120          138225       14
2       ham    kelime        0.8067        0.8082          145291       16
3       ham  karakter        0.8049        0.8066           97447       50
4     hafif  karakter        0.8044        0.8063           84428       54
5       tam  karakter        0.8023        0.8038           66879       51
```

Seçilen: tam + kelime. İkinciyle fark: 0.02 puan.
DİKKAT: Fark 0,5 puandan küçük; val setinin belirsizliği (yaklaşık ±0,6 puan) içinde kalıyor. İkisi pratikte eşit sayılmalı.

## 3. Test seti (bir kez): tam + kelime

ANA METRİK  macro-F1 : 0.8032
accuracy             : 0.8050
İŞ METRİĞİ  Negative recall    : 0.8263  (olumsuz yorumların yakalanan oranı)
            Negative precision : 0.8586  (olumsuz denenlerin gerçekten olumsuz oranı)
KALİBRASYON ECE      : 0.0325  (0 = model 'yüzde X eminim' dediğinde gerçekten yüzde X haklı)
log loss             : 0.4915

Sınıf bazında:
```
          precision  recall      f1  destek
Negative     0.8586  0.8263  0.8421    4518
Neutral      0.7290  0.7130  0.7209    4407
Positive     0.8240  0.8704  0.8466    4739
```

Karışıklık matrisi (satır = gerçek, sütun = tahmin):
```
tahmin    Negative  Neutral  Positive
label                                
Negative      3733      636       149
Neutral        533     3142       732
Positive        82      532      4125
```

LLM alt kümesinde (1.500, dengeli) macro-F1: 0.7960  (LLM ile karşılaştırmada bu sayı kullanılacak)

## 4. Zor örnekler: model olasılıkları ile insan dağılımı

Cross-entropy (küçük = insanlara daha yakın):
```
alt sınır (insanların kendi belirsizliği)    0.6402
baseline model                               1.1039
her şeye eşit olasılık (1/3)                 1.0986
train sınıf oranlarını tahmin et             1.1062
```

Çoğunluk etiketi olan 640 zor örnekte, modelin en olası sınıfı çoğunlukla aynı: 0.7422
Modelin ortalama güveni: test 0.773 | zor örnekler 0.753 (zor örneklerde daha düşükse model belirsizliği seziyor demektir)

En kalabalık 10 zor örnek:
```
                  n_kopya  insan_Neg  model_Neg  insan_Neu  model_Neu  insan_Pos  model_Pos
text                                                                                       
idare eder            532       0.15       0.04       0.85       0.95       0.00       0.00
Fena değil            470       0.09       0.02       0.91       0.98       0.00       0.00
beğenmedim            394       0.94       1.00       0.05       0.00       0.00       0.00
Güzel ürün            284       0.08       0.02       0.81       0.59       0.11       0.39
güzel bir ürün        280       0.07       0.01       0.87       0.41       0.06       0.58
Uygun fiyat           151       0.05       0.03       0.22       0.09       0.73       0.88
Hiç beğenmedim        151       0.98       1.00       0.02       0.00       0.00       0.00
gayet güzel           126       0.11       0.00       0.70       0.25       0.19       0.74
Güzel beğendim        119       0.04       0.00       0.86       0.63       0.10       0.37
Kokusu çok güzel       94       0.11       0.00       0.59       0.16       0.31       0.84
```

Eşitlik olan 5 örnek:
```
                          n_kopya  insan_Neg  model_Neg  insan_Neu  model_Neu  insan_Pos  model_Pos
text                                                                                               
Süper bir ürün.                16       0.12       0.02       0.44       0.05       0.44       0.92
çok güzel tavsiye ederim       14       0.14       0.01       0.43       0.04       0.43       0.95
Çok güzel  ürün                14       0.00       0.00       0.50       0.05       0.50       0.95
Kokusu çok hoş                 12       0.00       0.02       0.50       0.39       0.50       0.59
rengi çok koyu                 10       0.50       0.25       0.50       0.72       0.00       0.03
```

## 5. Model neye bakıyor? (her sınıfta ağırlığı en yüksek 15 özellik)

```
               Negative           Neutral        Positive
0          'beğenmedim'             'ama'     'yumuşacık'
1                'iade'           'fakat'        'harika'
2              'kumaşı'           'biraz'      'mükemmel'
3                'kötü'           'güzel'     'çok güzel'
4               'beden'           'ancak'   'teşekkürler'
5              'berbat'             'pek'       'çok iyi'
6             'almayın'            'orta'      'muhteşem'
7           'kalitesiz'         'diyemem'            'en'
8                 'hiç'  'çok beğenmedim'      'memnunum'
9   'tavsiye etmiyorum'        'ortalama'   'küçük değil'
10              'değil'            'fena'  'kullanıyorum'
11          'etmiyorum'         'yine de'     'gerçekten'
12                'dar'        'renkleri'     'yıllardır'
13              'kumaş'  'kullanılabilir'         'gayet'
14           'çok kötü'      'fena değil'            've'
```
