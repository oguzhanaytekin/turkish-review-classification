# T-01 Veri Keşfi Raporu (otomatik üretildi)

## 0. Dosya

Dosya: TRSAv1.csv | kodlama: utf-8-sig | ayırıcı: ','
Kolonlar ve tipleri: {'id': 'int64', 'score': 'str', 'review': 'str'}
Seçilen kolonlar -> metin: 'review', etiket: 'score'
İlk 3 satır:
```
  id     score                                       review
0  0  Positive            Kokusu çok güzel etkisi de efsane
1  1  Negative      hiç bir fark olmadı siyah nokta silmedi
2  2  Positive  şeffaf güzel bir krem elimi yumuşacık yaptı
```

## 1. Temel tablo

Toplam satır: 150,000
Boş metin (NaN): 0 | boş etiket (NaN): 0 | sadece boşluktan oluşan metin: 0

Ham etiket değerleri (tırnak içinde, boşluklar görünsün diye):
```
label
'Positive'    50000
'Negative'    50000
'Neutral'     50000
```

Sınıf dağılımı:
```
          satir   oran
label                 
Negative  50000  %33.3
Neutral   50000  %33.3
Positive  50000  %33.3
```

Kelime sayısı dağılımı (sınıfa göre):
```
           %5  %25  medyan   %75   %95  ortalama  max
label                                                
Negative  2.0  5.0     8.0  15.0  31.0      11.5  209
Neutral   2.0  5.0     8.0  14.0  29.0      11.0  283
Positive  3.0  7.0    10.0  17.0  33.0      13.4  207
```

Karakter sayısı (sınıfa göre):
```
          medyan  ortalama   max
label                           
Negative    56.0      77.1  1458
Neutral     54.0      74.5  1993
Positive    71.0      92.9  1433
```

Çok kısa yorumlar:
```
          <=1 kelime  <=2 kelime
label                           
Negative         614        3228
Neutral          300        3947
Positive          42         747
```

## 2. Tekrar analizi

Normalleştirme: Türkçe küçük harf + noktalama/emoji silme + boşluk sadeleştirme (src/text_utils.py)
```
                         birebir  normalleştirilmiş
satir                     150000             149940
tekil_metin               140980             137681
fazlalik_satir              9020              12259
tekrar_eden_metin           2736               3112
tekrar_grubundaki_satir    11756              15371
```

Birebir tekrarların payı: %6.01
Normalleştirilmiş tekrarların payı: %8.18
Normalleştirme ek olarak 3,239 tekrar yakaladı.
Normalleştirince tamamen boşalan metin (sadece noktalama/emoji): 60

En çok tekrar eden 15 metin (normalleştirilmiş):
```
                       toplam                               siniflar
norm_metin                                                          
idare eder                532   Negative:79, Neutral:451, Positive:2
fena değil                470   Negative:42, Neutral:427, Positive:1
beğenmedim                394   Negative:372, Neutral:21, Positive:1
güzel ürün                284  Negative:24, Neutral:230, Positive:30
güzel bir ürün            280  Negative:20, Neutral:243, Positive:17
hiç beğenmedim            151                Negative:148, Neutral:3
uygun fiyat               151   Negative:8, Neutral:33, Positive:110
gayet güzel               126   Negative:14, Neutral:88, Positive:24
güzel beğendim            119   Negative:5, Neutral:102, Positive:12
kokusu çok güzel           94   Negative:10, Neutral:55, Positive:29
hiç bir işe yaramıyor      84                 Negative:81, Neutral:3
çok kalitesiz              78                 Negative:77, Neutral:1
hiçbir işe yaramıyor       75                 Negative:74, Neutral:1
çok güzel bir ürün         71    Negative:6, Neutral:30, Positive:35
fiyatı çok uygun           70    Negative:3, Neutral:10, Positive:57
```

Normalleştirmenin birleştirdiği yazımlara örnek ('güzel', 27 farklı yazım):
     27 x  'güzel'
      4 x  'Güzel......'
      2 x  'Güzel'
      2 x  'Güzel.........'
      2 x  'Güzel.'
      1 x  'güzel ☺️'

## 3. Çelişkili etiketler

Birebir aynı metin, farklı etiket: 889 metin
Normalleştirilmiş metin, farklı etiket: 1,049 metin
Bu metinlere ait satır sayısı: 9,593 (tüm verideki oranı: %6.40)

Hangi sınıflar birbiriyle çelişiyor (metin sayısı):
```
Negative + Neutral               497
Neutral + Positive               390
Negative + Neutral + Positive    123
Negative + Positive               39
```

Çelişkili metinlerin medyan kelime sayısı: 3 (tüm tekil metinlerde: 10)

En kalabalık 15 çelişkili metin:
```
                       Negative  Neutral  Positive
norm_metin                                        
idare eder                   79      451         2
fena değil                   42      427         1
beğenmedim                  372       21         1
güzel ürün                   24      230        30
güzel bir ürün               20      243        17
uygun fiyat                   8       33       110
hiç beğenmedim              148        3         0
gayet güzel                  14       88        24
güzel beğendim                5      102        12
kokusu çok güzel             10       55        29
hiç bir işe yaramıyor        81        3         0
çok kalitesiz                77        1         0
hiçbir işe yaramıyor         74        1         0
çok güzel bir ürün            6       30        35
fiyatı çok uygun              3       10        57
```

## 4. Temizlik senaryoları ve sınıf dengesi

(Normalleştirince boşalan 60 satır tüm senaryolarda hariç.)
```
                                 Negative  Neutral  Positive  toplam  min/max
ham                                 49975    49967     49998  149940    0.999
A: tekilleştir, ilkini tut          45461    44566     47654  137681    0.935
B: çelişkileri at + tekilleştir     45181    44070     47381  136632    0.930
C: çoğunluk etiketi                 45334    44382     47556  137272    0.933
```

C senaryosunda eşitlik yüzünden atılan metin: 409
min/max = en küçük sınıf / en büyük sınıf. 1'e yakınsa denge korunuyor.

## 5. Her sınıftan 5 rastgele örnek


[Negative]
  (  1 kelime) Beğenmedim
  (  9 kelime) Çok kötü inanamadım çok kötü böyle bir şey olmaz
  ( 14 kelime) Bugün geldi elime ne kumaş aynı ne de desen sadece siyah işte o beğenmedim
  (  9 kelime) Çok kötü fırçanın ucu tüy tüy oldu üç kullanmada
  (  3 kelime) kokusu kalıcı değil

[Neutral]
  (  9 kelime) beğendim basarili yumusak ama duzluk konusu icin yeni daha
  ( 14 kelime) Resümdekine göre rengi  açık geldi çok esmerim çingen gibi gezmemek için kaldı öyle 😏😏
  ( 22 kelime) Henüz 1 haftadır kullanıyorum ilk bitiş olarak gayet başarılı.Ama bürün kısmındaki yağlı görüntü henüz tamamen yok olmadı.Umarım düzenli kullanım sonrası fayda…
  ( 14 kelime) kargo hızlıydı ve paketleme güzeldi. ürünü hemen denedim kokusu güzel ama kalitesini zamanla göreceğiz.
  (  6 kelime) Ürün çok minik olduğundan iade ettim

[Positive]
  (  4 kelime) fiyat uygun, ürün güzel.
  ( 71 kelime) ürünü genel anlamda beğendim pembe rengi çok tatli 1 hafta dir kullaniyorum. iyi bir yüz temizle plingi ile kullanılırsa mükemmel sonuç verir. Elinize küçük ge…
  (  9 kelime) güzel çok beğendim hem masaj yapiyo hem temizliyo ☺️👍
  (  9 kelime) ürün gayet güzel ve basarili beğendim AYRICA KARGODA hızlıYDI
  (  7 kelime) Kullandığım en güzel nemlendirici olur kendileri 👍🏻
