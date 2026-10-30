# Öğrenme Günlüğü

Her oturumdan sonra 3–5 satır: ne yaptım, nerede zorlandım, hangi kararı neden aldım.
Mülakatta projeyi anlatırken bu dosya hafıza görevi görecek.
##
Ama bir ayrıntıyı düzeltmem lazım: "İ".lower() sonucunda “i” ve “ı” çıkmıyor. Çıkan şey “i” ve görünmez bir “birleşik nokta” işareti (U+0307). Yani:

python
"İ".lower()   # "i̇"  ->  "i" + U+0307 (iki karakter)
"I".lower()   # "i"  ->  Türkçede "ı" olmalıydı
“Python’un lower() fonksiyonu Türkçeyi bilmiyor. ‘I’yı ‘i’ye, ‘İ’yi de ‘i’ artı görünmez bir nokta işaretine çeviriyor. Bu, aynı yorumların farklı metin sayılmasına yol açıyordu. Küçültmeden önce İ ve I’yı elle eşleyen bir fonksiyon yazdım ve bunu testlerle doğruladım.”
###
---

## 2026-10-30 — Proje başlangıcı
- Veri seti: TRSAv1. Neden: üç sınıf dengeli, yayınlanmış bir referans skor var
  (mBERT ile ~%82 doğruluk), nötr sınıf LLM'ler için ilginç bir zorluk.
- Kapsama alınanlar: train/test bölmesinden önce tekrar temizliği (sızıntıyı önlemek için)
  ve proje sonunda 100–200 hatanın elle incelenmesi.
- Çalışma yöntemi: kodu Claude yazıyor; kritik kararları ben veriyorum, çıktıları ben yorumluyorum.
  Haftada 2 × 30 dk yapay zekâsız SQL/pandas pratiği ayrıca devam ediyor.

## Şablon
### YYYY-AA-GG — T-0X
- Ne yaptım:
- Nerede zorlandım / ne öğrendim:
- Aldığım karar ve nedeni:
