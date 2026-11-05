# Kod Günlüğü

Kodda ne eklendi, ne değişti, ne çıkarıldı ve neden, yapılış sırasına göre.
Kararların gerekçeleri `docs/0X_*.md` dosyalarında. Bu dosya "hangi kod nerede,
ne zaman ve neden değişti?" sorusunun cevabı.

---

## 2026-10-30 — T-01: Repo iskeleti ve veri keşfi

### Eklenen dosyalar
| Dosya | Ne işe yarıyor |
|---|---|
| `.gitignore` | `.venv`, ham veri ve `.env` repoya girmesin |
| `requirements.txt` | `pandas`, `pytest` |
| `pytest.ini` | Testlerin `src` paketini bulabilmesi için (`pythonpath = .`) |
| `src/text_utils.py` | `turkish_lower()`: Türkçe küçük harf. `normalize_text()`: tekrar tespiti için sadeleştirme |
| `tests/test_text_utils.py` | İ/I ve normalleştirme testleri |
| `scripts/01_veri_kesfi.py` | Veriyi indirir, okur, keşif raporunu üretir (`reports/t01_*`) |
| `docs/01_veri_kesfi.md`, `README.md`, `LEARNING_LOG.md` | Bulgular, proje tanıtımı, öğrenme günlüğü |

### Geliştirme sırasında değişenler
- **`normalize_text` hatası:** İlk sürümde noktalama silinirken birleşik nokta (U+0307)
  boşlukla değiştiriliyordu, bu yüzden "i̇yi" metni "i yi" oluyor ve kelime bölünüyordu.
  Sahte veriyle test ederken fark edildi. Çözüm: Birleşik işaretler önce boşluk
  bırakmadan siliniyor (`_COMBINING`). Yeni test: `test_normalize_hatali_lower_ciktisi_kelimeyi_bolmez`.
- **pandas sürüm uyumu:** `DataFrame.map` yerine `apply(col.map)`, `groupby.apply`
  yerine `groupby.sample` kullanıldı (pandas 3'te `groupby.apply` davranışı değişti).

### Neden böyle kuruldu
- `detect_columns` ve `read_csv_robust`: Kod, veri görülmeden yazıldı. Kolon adları,
  kodlama ve ayırıcı bilinmediği için betik bunları kendisi buluyor.

---

## 2026-11-05 — T-02: Temizlik, zor örnekler ve bölme

### Eklenen dosyalar
| Dosya | Ne işe yarıyor |
|---|---|
| `src/data_io.py` | `ensure_data`, `read_csv_robust` (01'den taşındı) + **yeni** `load_trsa`: veriyi standart kolonlarla (`id, text, label`) döndürür |
| `src/report.py` | `Report`, `short`, `pct` (01'den taşındı) + `use_utf8_stdout` |
| `src/cleaning.py` | `group_by_text`, `separate_clean_and_hard`, `stratified_split`, `llm_subset_mask`, `check_no_leakage` |
| `scripts/02_temizlik_bolme.py` | Temizlik ve bölmeyi çalıştırır; `data/processed/*.csv` ve `reports/t02_*` üretir |
| `tests/test_cleaning.py` | 7 test: gruplama, zor/temiz ayrımı, yumuşak etiket, eşitlik, bölme oranı, tekrarlanabilirlik, sızıntı kontrolü |
| `docs/02_temizlik_bolme.md`, `docs/KOD_GUNLUGU.md` | T-02 kararları, bu günlük |

### Değişen dosyalar
| Dosya | Değişiklik |
|---|---|
| `scripts/01_veri_kesfi.py` | `ensure_data`, `read_csv_robust`, `Report`, `short`, `pct` ve stdout ayarı **çıkarıldı**; artık `src/data_io.py` ve `src/report.py`'den import ediliyor. Davranış aynı kaldı (sahte veriyle doğrulandı) |
| `requirements.txt` | **+ scikit-learn** (stratified bölme için; T-03 baseline'da da gerekecek) |
| `.gitignore` | **+ `data/processed/`**: Betik bu dosyaları her seferinde aynı şekilde yeniden üretebildiği için repoya girmiyor |
| `README.md` | Durum ve çalıştırma komutları güncellendi |
| `docs/01_veri_kesfi.md` | "Kararlar ve alternatifler" tablosu ve C kararının T-02'de nasıl değiştiğine dair not eklendi |

### Neden taşındı? (refactor)
- **Ortak kod `src/` içine:** İki betik aynı okuma kodunu kullanınca kodu kopyalamak
  yerine ortak bir modül yazıldı. Böylece bir hata düzeltilince her yerde birden düzelir.
  Kural: Bir kod iki yerde gerekiyorsa `src/` içine gider.
- **Mantık ile çalıştırma ayrıldı:** Kuralların kendisi `src/cleaning.py` içinde, test
  edilebilir fonksiyonlar olarak duruyor. Dosya okuma/yazma ve rapor ise
  `scripts/02_temizlik_bolme.py` içinde.

### Önemli uygulama ayrıntıları
- Gruplama ham metne değil, **normalleştirilmiş** metne göre yapılıyor.
- Ana veride her grubu temsil eden satır, grubun **ilk geçen ham metni**. Modele ham
  metin gidecek; normalleştirme yalnızca eşleştirme için kullanılıyor.
- `n_kopya` kolonu saklanıyor (metin kaç kez geçmiş). İleride sık geçen metinleri
  analiz etmek ya da ağırlıklandırma yapmak için.
- Bölme iki adımlı: önce 80/20, sonra 20'lik kısım 10/10.
- LLM alt kümesi testten sınıf başına 500 satır (seed 42). `test.csv` içindeki
  `llm_alt_kume` kolonu bunu işaretliyor.
- `check_no_leakage` betik içinde **her çalıştırmada** kontrol ediliyor. Sızıntı varsa
  betik dosya yazmadan durur.

---

## 2026-11-05 — T-03: Baseline (TF-IDF + lojistik regresyon)

### Eklenen dosyalar
| Dosya | Ne işe yarıyor |
|---|---|
| `src/preprocess.py` | `preprocess(texts, mod)`: ham / hafif / tam ön işleme. `light_normalize`: noktalamayı ve emojiyi koruyan küçük harf. `ascii_writing_stats`: "basarili" gibi Türkçe karaktersiz yazımın yaygınlığını ölçer |
| `src/baseline.py` | `build_pipeline(temsil)`: TF-IDF (kelime veya karakter n-gram) + lojistik regresyon. `top_features`: her sınıfta en etkili özellikler |
| `src/metrics.py` | `expected_calibration_error` (ECE), `soft_cross_entropy` ve `entropy` (zor örnekler için), `classification_summary` (macro-F1, accuracy, sınıf bazında P/R/F1) |
| `scripts/03_baseline.py` | ASCII ölçümü → 6'lı ablation (val) → seçilen modelin test değerlendirmesi → zor örnekler → özellik yorumu |
| `tests/test_preprocess_metrics.py` | 6 test: ön işleme modları, ASCII ölçümü, ECE, cross-entropy alt sınırı, `lowercase=False`, emoji/noktalama token'ları |
| `docs/03_baseline.md` | T-03 kararları ve alternatifler |

### Değişen dosyalar
| Dosya | Değişiklik |
|---|---|
| `.gitignore` | **+ `models/`**: Eğitilmiş model dosyası (`baseline.joblib`) repoya girmiyor, betik yeniden üretiyor |
| `README.md` | 03 çalıştırma komutu eklendi |

### Önemli uygulama ayrıntıları
- **`lowercase=False`:** sklearn'ün `TfidfVectorizer` sınıfı varsayılan olarak Python'un `lower()`
  fonksiyonunu kullanır ve T-01'deki İ/I hatasını geri getirir. Bu yüzden kapatıldı, küçük
  harf işlemini `preprocess` yapıyor. Bir testle güvence altına alındı.
- **Token deseni:** sklearn'ün varsayılan deseni (`\b\w\w+\b`) noktalamayı, emojiyi ve tek
  harfli kelimeleri atıyor. "hafif" ön işlemenin korumaya çalıştığı "!!!" ve "👍"
  boşa gitmesin diye desen `kelime | tek sembol` olarak değiştirildi.
- **Karakter n-gram: `char_wb`, 2–5 harf.** `_wb` = n-gram'lar kelime sınırını aşmaz.
- **Val setinde seçim, test setinde tek ölçüm.** Ablation döngüsü test setine hiç dokunmuyor.
- Fark 0,5 puandan küçükse rapor "pratikte eşit" uyarısı veriyor (val belirsizliği ±0,6 puan).
- Test ve zor örnek tahminleri olasılıklarıyla birlikte `reports/` altına kaydediliyor;
  ileride hata analizinde kullanılacak.
- İşlenmiş CSV'ler `keep_default_na=False` ile okunuyor: Metni "NA" ya da "null" olan
  bir yorum boş değer sanılmasın diye.
