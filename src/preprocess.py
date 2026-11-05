"""T-03: Model girdisi için ön işleme seçenekleri ve ASCII yazım ölçümü.

Üç mod (ablation'da karşılaştırılıyor):
  ham   : metin olduğu gibi (büyük/küçük harf, noktalama, emoji korunur)
  hafif : Türkçe küçük harf + boşluk temizliği; noktalama ve emoji KORUNUR
  tam   : normalize_text (T-01) -> noktalama ve emoji SİLİNİR

Not: normalize_text tekrar tespiti için yazılmıştı. "!!!", "👍", "..." duygu taşıyabildiği
için model girdisinde silinip silinmemesi ayrıca ölçülüyor.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.text_utils import _COMBINING, normalize_text, turkish_lower

MODES = ("ham", "hafif", "tam")
_SPACES = re.compile(r"\s+")


def light_normalize(text: str) -> str:
    """Türkçe küçük harf + Unicode birleştirme + boşluk temizliği. Noktalama/emoji kalır."""
    text = unicodedata.normalize("NFC", str(text))
    text = _COMBINING.sub("", turkish_lower(text))
    return _SPACES.sub(" ", text).strip()


def preprocess(texts: pd.Series, mode: str) -> pd.Series:
    if mode == "ham":
        return texts.astype(str)
    if mode == "hafif":
        return texts.map(light_normalize)
    if mode == "tam":
        return texts.map(normalize_text)
    raise ValueError(f"Bilinmeyen mod: {mode}. Seçenekler: {MODES}")


# --------------------------------------------------------------------------
# ASCII yazım ölçümü ("basarili" -> "başarılı" sorunu ne kadar yaygın?)
# --------------------------------------------------------------------------
_TR_CHARS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")

# Bu veride sık geçen kelimelerin Türkçe karaktersiz yazımları.
# Yalnızca Türkçede başka bir kelimeye denk gelmeyen biçimler alındı
# ("sik" alınmadı: hem "şık" hem "sık" olabilir).
ASCII_VARIANTS = {
    "guzel": "güzel", "cok": "çok", "urun": "ürün", "urunu": "ürünü", "begendim": "beğendim",
    "begenmedim": "beğenmedim", "kotu": "kötü", "hizli": "hızlı", "basarili": "başarılı",
    "gercekten": "gerçekten", "tesekkurler": "teşekkürler", "tesekkur": "teşekkür",
    "guzeldi": "güzeldi", "begendik": "beğendik", "kullanisli": "kullanışlı", "boyle": "böyle",
    "yumusak": "yumuşak", "tatli": "tatlı",
}


def ascii_writing_stats(texts: pd.Series, min_words: int = 5) -> dict:
    """Türkçe karaktersiz yazımın yaygınlığını iki yolla tahmin eder.

    1. min_words veya daha uzun yorumlardan hiç Türkçe karakter içermeyenlerin oranı
       (uzun bir Türkçe yorumun hiç ç/ğ/ı/ö/ş/ü içermemesi nadirdir).
    2. Sık kelimelerin ASCII biçimlerini ("guzel", "cok"...) içeren yorumların oranı.
    """
    t = texts.astype(str)
    words = t.str.split().str.len()
    long_ = t[words >= min_words]
    no_tr = ~long_.str.contains(_TR_CHARS)

    tokens = t.map(lambda s: set(re.findall(r"\w+", turkish_lower(s))))
    variants = set(ASCII_VARIANTS)
    has_variant = tokens.map(lambda toks: bool(toks & variants))
    variant_counts = pd.Series(
        {v: int(tokens.map(lambda toks, v=v: v in toks).sum()) for v in sorted(variants)}
    ).sort_values(ascending=False)

    return {
        "uzun_yorum_sayisi": int(len(long_)),
        "uzun_ve_turkce_karaktersiz": int(no_tr.sum()),
        "uzun_ve_turkce_karaktersiz_oran": float(no_tr.mean()) if len(long_) else 0.0,
        "ascii_kelime_iceren": int(has_variant.sum()),
        "ascii_kelime_iceren_oran": float(has_variant.mean()) if len(t) else 0.0,
        "en_sik_ascii_kelimeler": variant_counts.head(10).to_dict(),
        "ornekler": long_[no_tr].head(5).tolist(),
    }
