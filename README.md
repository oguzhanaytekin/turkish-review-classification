# Türkçe Ürün Yorumu Sınıflandırma: Klasik ML, BERTurk ve LLM

> Durum: geliştirme aşamasında. Tamamlanan: T-01 veri keşfi, T-02 temizlik ve bölme, T-03 baseline. Devam eden: T-04 BERTurk.
> Kod değişikliklerinin geçmişi: `docs/KOD_GUNLUGU.md` · Terimler sözlüğü: `docs/SOZLUK.md`

## Problem
Bir e-ticaret şirketi her gün binlerce Türkçe ürün yorumu alıyor. Olumsuz yorumları
otomatik ayıklamak için hangi yaklaşım en mantıklı?

Bu projede üç yaklaşımı **aynı test kümesi** üzerinde doğruluk, maliyet ve hız
açısından karşılaştırıyorum:

| Yaklaşım | Açıklama |
|---|---|
| Baseline | TF-IDF + lojistik regresyon |
| Fine-tune | BERTurk |
| LLM | Zero-shot / few-shot sınıflandırma |

Ayrıca modellerin yanıldığı 100–200 örneği elle inceleyip hataların ne kadarının
modelden, ne kadarının etiket gürültüsünden kaynaklandığını raporluyorum.

## Veri
TRSAv1: Türk e-ticaret sitelerinden 150.000 ürün yorumu, olumlu/nötr/olumsuz
sınıflarında dengeli (50.000'er).

- Kaynak: [Hugging Face – maydogan/Turkish_SentimentAnalysis_TRSAv1](https://huggingface.co/datasets/maydogan/Turkish_SentimentAnalysis_TRSAv1)
- Atıf: Aydoğan M, Kocaman V. *TRSAv1: A new benchmark dataset for classifying user
  reviews on Turkish e-commerce websites.* Journal of Information Science, 2022.
  doi:10.1177/01655515221074328

Ham veri repoya eklenmez; `scripts/01_veri_kesfi.py` ilk çalıştırmada `data/raw/` altına indirir.

## Kurulum (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Çalıştırma
```powershell
python scripts\01_veri_kesfi.py      # veri keşfi, rapor: reports\t01_rapor.md
python scripts\02_temizlik_bolme.py  # temizlik + bölme, çıktı: data\processed\, rapor: reports\t02_rapor.md
python scripts\03_baseline.py        # baseline ablation + test, rapor: reports\t03_rapor.md
# BERTurk (GPU gerekir): notebooks/04_berturk_colab.ipynb dosyasını Colab'da aç
pytest                               # testler
```

## Proje yapısı
```
src/          tekrar kullanılan kod (ör. Türkçe metin normalleştirme)
scripts/      adım adım çalıştırılan betikler
docs/         her adımın bulguları ve kararları
reports/      betiklerin ürettiği çıktılar
tests/        testler
notebooks/    GPU gerektiren adımlar için Colab notebook'ları
```
