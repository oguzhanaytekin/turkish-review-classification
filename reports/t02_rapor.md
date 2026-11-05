# T-02 Temizlik ve Bölme Raporu (otomatik üretildi)

## 1. Satırlar nereye gitti?

```
                                      satir
ham veri                             150000
- normalleştirince boşalan (atıldı)      60
= kalan                              149940
- tekrar eden kopyalar (atıldı)        3715
- zor örneklere giden satırlar         9593
= ana veri (tekil metin)             136632
```

Zor örnekler seti: 1,049 tekil metin (409 tanesinde eşitlik var, çoğunluk etiketi yok)

## 2. Bölme: sınıf dağılımı (stratified 80/10/10, seed=42)

```
                      Negative Neutral Positive  toplam %Negative %Neutral %Positive
ana veri                 45181   44070    47381  136632      33.1     32.3      34.7
train                    36145   35256    37904  109305      33.1     32.3      34.7
val                       4518    4407     4738   13663      33.1     32.3      34.7
test                      4518    4407     4739   13664      33.1     32.3      34.7
test / LLM alt kümesi      500     500      500    1500      33.3     33.3      33.3
```

Oranlar: train %80.00, val %10.00, test %10.00

## 3. Sızıntı kontrolü

train / val / test / zor örnekler arasında ortak normalleştirilmiş metin: YOK (kontrol geçti)
Ana veride tekrarları birleştirilmiş metin: 2,063 (en çok: 'hiç güzel değil', 54 kopya)

## 4. Zor örnekler seti

Hangi sınıflar arasında çelişki var (metin sayısı):
```
Negative + Neutral               497
Neutral + Positive               390
Negative + Neutral + Positive    123
Negative + Positive               39
```

Çoğunluk etiketi dağılımı (eşitlikler hariç):
```
cogunluk_etiketi
Negative    153
Neutral     312
Positive    175
```

En kalabalık 10 zor örnek ve etiketleyici dağılımı (yumuşak etiket):
```
                  n_kopya  p_Negative  p_Neutral  p_Positive cogunluk_etiketi
norm                                                                         
idare eder            532      0.1485     0.8477      0.0038          Neutral
fena değil            470      0.0894     0.9085      0.0021          Neutral
beğenmedim            394      0.9442     0.0533      0.0025         Negative
güzel ürün            284      0.0845     0.8099      0.1056          Neutral
güzel bir ürün        280      0.0714     0.8679      0.0607          Neutral
uygun fiyat           151      0.0530     0.2185      0.7285         Positive
hiç beğenmedim        151      0.9801     0.0199      0.0000         Negative
gayet güzel           126      0.1111     0.6984      0.1905          Neutral
güzel beğendim        119      0.0420     0.8571      0.1008          Neutral
kokusu çok güzel       94      0.1064     0.5851      0.3085          Neutral
```

Örnek eşitlikler:
  'Süper bir ürün.'  ->  Negative:2, Neutral:7, Positive:7
  'çok güzel tavsiye ederim'  ->  Negative:2, Neutral:6, Positive:6
  'Çok güzel  ürün'  ->  Neutral:7, Positive:7
  'Kokusu çok hoş'  ->  Neutral:6, Positive:6
  'rengi çok koyu'  ->  Negative:5, Neutral:5

## 5. Üretilen dosyalar

  data/processed/train.csv
  data/processed/val.csv
  data/processed/test.csv
  data/processed/zor_ornekler.csv
