from src.text_utils import normalize_text, turkish_lower


def test_python_lower_turkce_bilmez():
    # Kendi fonksiyonumuzu neden yazdığımızın kanıtı
    assert "I".lower() == "i"            # Türkçede "ı" olmalıydı
    assert len("İ".lower()) == 2         # "i" + birleşik nokta


def test_turkish_lower_noktasiz_i():
    assert turkish_lower("IŞIK") == "ışık"


def test_turkish_lower_noktali_i():
    assert turkish_lower("İSTANBUL") == "istanbul"


def test_normalize_noktalama_ve_emoji():
    assert normalize_text("  Çok GÜZEL!!! 👍 ") == "çok güzel"


def test_normalize_ayrik_yazilmis_harf():
    # "I" + U+0307 (birleşik nokta) ekranda "İ" gibi görünür ama iki karakterdir
    assert normalize_text("İyi") == "iyi"


def test_normalize_hatali_lower_ciktisi_kelimeyi_bolmez():
    # Başka bir araç "İYİ" metnini Python lower() ile küçültmüşse "i̇yi̇" olur
    assert "İYİ".lower() == "i̇yi̇"
    assert normalize_text("İYİ".lower()) == "iyi"


def test_normalize_iki_yorum_ayni_sayilir():
    assert normalize_text("Harika ürün, tavsiye ederim.") == normalize_text("harika ürün tavsiye ederim!!")


def test_normalize_bos_ve_eksik_deger():
    assert normalize_text("!!! ...") == ""
    assert normalize_text(None) == ""
