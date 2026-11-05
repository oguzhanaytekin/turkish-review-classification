"""T-03: Baseline model, TF-IDF + lojistik regresyon.

İki temsil karşılaştırılıyor:
  kelime   : kelime 1-2 gramları; noktalama ve emoji de ayrı token sayılır
  karakter : kelime sınırları içinde 2-5 harflik parçalar (char_wb)
             -> "basarili" ile "başarılı", "güzel" ile "güzeldi" ortak parçalar paylaşır
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ANALYZERS = ("kelime", "karakter")

# Kelime VEYA tek bir noktalama/emoji karakteri. sklearn'ün varsayılan deseni
# (\b\w\w+\b) noktalamayı, emojiyi ve tek harfli kelimeleri atıyor.
_WORD_OR_SYMBOL = r"(?u)\b\w+\b|[^\w\s]"


def build_pipeline(analyzer: str, C: float = 1.0, seed: int = 42) -> Pipeline:
    if analyzer == "kelime":
        vec = TfidfVectorizer(
            analyzer="word", token_pattern=_WORD_OR_SYMBOL, ngram_range=(1, 2),
            min_df=2, max_features=300_000, sublinear_tf=True,
            lowercase=False,  # ÖNEMLİ: sklearn lower() kullanır ve Türkçe İ/I'yı bozar; küçük harfi biz yapıyoruz
        )
    elif analyzer == "karakter":
        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5),
            min_df=3, max_features=300_000, sublinear_tf=True,
            lowercase=False,
        )
    else:
        raise ValueError(f"Bilinmeyen analyzer: {analyzer}. Seçenekler: {ANALYZERS}")
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    return Pipeline([("tfidf", vec), ("lr", clf)])


def top_features(pipe: Pipeline, n: int = 15) -> pd.DataFrame:
    """Her sınıf için ağırlığı en yüksek n özellik (model neye bakarak karar veriyor?)."""
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    lr = pipe.named_steps["lr"]
    cols = {}
    for i, label in enumerate(lr.classes_):
        idx = np.argsort(lr.coef_[i])[::-1][:n]
        cols[label] = [repr(vocab[j]) for j in idx]
    return pd.DataFrame(cols)
