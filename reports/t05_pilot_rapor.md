# T-05 Pilot Raporu (otomatik üretildi) — model gemini-3.5-flash-lite

## 1. Kullanılabilir flash-lite modelleri

models/gemini-2.5-flash-lite, models/gemini-3.1-flash-lite, models/gemini-3.1-flash-lite-image, models/gemini-3.5-flash-lite

## 2. Tek tek ve toplu gönderim karşılaştırması (50 yorum, zero-shot, 1 örnekleme)

Tek tek: 50/50 cevap, 442 sn | toplu: 50/50 cevap, 26 sn
İki yöntemin aynı etiketi verme oranı: 0.94  (sıcaklık 1 olduğu için 1,00 beklenmez)
Doğruluk: tek tek 0.84 | toplu 0.86 (50 yorumda belirsizlik ±0,13; kaba fikir verir)

## 3. Logprobs desteği

Desteklenmiyor: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Logprobs is not enabled for this model', 'status': 'INVALID_ARGUMENT'}}

## 4. Tam koşu tahmini

İstek: 53 | ortalama cevap süresi 0.9 sn | dakikalık kota beklemesi 0 | hata 7 | girdi token 12,338 | çıktı token 6,699
k=3, batch=20: 768 istek
k=5, batch=20: 1,280 istek
k=10, batch=20: 2,560 istek
Günlük istek sınırını AI Studio'daki kota sayfasından gör; istek sayısı onu aşıyorsa koşu birkaç güne bölünür.
