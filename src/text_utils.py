"""Türkçe metin yardımcıları.

Bu dosyadaki fonksiyonlar projenin her aşamasında (keşif, temizlik, model)
aynı şekilde kullanılacak. Normalleştirme kuralını tek bir yerde tutmak,
"keşifte başka, modelde başka temizlik" hatasını önler.
"""

import re
import unicodedata

# Python'un str.lower() fonksiyonu Türkçeyi bilmez:
#   "I".lower()  -> "i"        (Türkçede "ı" olmalı)
#   "İ".lower()  -> "i̇"        ("i" + birleşik nokta U+0307, iki karakter!)
# Bu yüzden önce bu iki büyük harfi elle çeviriyoruz, sonra lower() uyguluyoruz.
_TR_UPPER_MAP = str.maketrans({"I": "ı", "İ": "i"})

# NFC'den sonra hâlâ ayrık kalan birleşik işaretler (ör. "i" + U+0307).
# Bunları boşlukla DEĞİL, hiçbir şeyle değiştiriyoruz; yoksa "i̇yi" -> "i yi" olur, kelime bölünür.
_COMBINING = re.compile(r"[̀-ͯ]")
_NON_WORD = re.compile(r"[^\w\s]")  # harf, rakam ve boşluk dışındaki her şey
_UNDERSCORE = re.compile(r"_+")      # \w alt çizgiyi de harf sayar, onu ayrıca atıyoruz
_SPACES = re.compile(r"\s+")


def turkish_lower(text: str) -> str:
    """Metni Türkçe kurallarıyla küçük harfe çevirir."""
    return text.translate(_TR_UPPER_MAP).lower()


def normalize_text(text) -> str:
    """Tekrar tespiti için metni sadeleştirir.

    Adımlar:
      1. Unicode NFC: "I" + birleşik nokta gibi ayrık yazılmış harfleri
         tek karaktere ("İ") birleştirir. Aynı görünen iki metnin
         bilgisayarda farklı baytlar olmasını engeller.
      2. Türkçe küçük harf.
      3. Ayrık kalan birleşik işaretleri sil (Python'un hatalı lower()
         çıktısından gelen "i̇" gibi).
      4. Noktalama ve emojileri at.
      5. Fazla boşlukları tek boşluğa indir.

    Not: Bu fonksiyon yalnızca "bu iki yorum aynı mı?" sorusu için.
    Modele verilecek metin için ayrı bir karar vereceğiz (ör. "!!!" ve
    emojiler duygu taşıyabilir, modelden silmek istemeyebiliriz).
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = turkish_lower(text)
    text = _COMBINING.sub("", text)
    text = _NON_WORD.sub(" ", text)
    text = _UNDERSCORE.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
