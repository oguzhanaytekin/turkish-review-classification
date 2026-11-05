import numpy as np
import pandas as pd

from src.baseline import build_pipeline
from src.metrics import entropy, expected_calibration_error, soft_cross_entropy
from src.preprocess import ascii_writing_stats, preprocess


def test_on_isleme_modlari():
    s = pd.Series(["İYİ Ürün!!! 👍"])
    assert preprocess(s, "ham").iloc[0] == "İYİ Ürün!!! 👍"
    assert preprocess(s, "hafif").iloc[0] == "iyi ürün!!! 👍"   # noktalama ve emoji korunur
    assert preprocess(s, "tam").iloc[0] == "iyi ürün"            # silinir


def test_ascii_olcumu():
    s = pd.Series(["urun cok guzel ve kargo hizli geldi", "ürün çok güzel ve kargo hızlı geldi", "iyi"])
    st = ascii_writing_stats(s, min_words=5)
    assert st["uzun_yorum_sayisi"] == 2
    assert st["uzun_ve_turkce_karaktersiz"] == 1
    assert st["ascii_kelime_iceren"] == 1


def test_ece_mukemmel_ve_kotu_kalibrasyon():
    y = np.array([0, 1])
    assert expected_calibration_error(np.array([[1.0, 0.0], [0.0, 1.0]]), y) == 0.0
    # %90 emin ama hep yanılıyor -> ECE 0.9
    assert abs(expected_calibration_error(np.array([[0.1, 0.9], [0.9, 0.1]]), y) - 0.9) < 1e-9


def test_soft_cross_entropy_alt_siniri_entropi():
    human = np.array([[0.1, 0.8, 0.1]])
    assert np.isclose(soft_cross_entropy(human, human)[0], entropy(human)[0])
    assert soft_cross_entropy(human, np.array([[0.8, 0.1, 0.1]]))[0] > entropy(human)[0]


def test_vektorlestirici_turkce_buyuk_harfi_bozmaz():
    # lowercase=False: sklearn'ün lower() hatası devreye girmemeli
    pipe = build_pipeline("kelime")
    assert pipe.named_steps["tfidf"].lowercase is False


def test_kelime_temsili_emoji_ve_noktalamayi_sayar():
    pipe = build_pipeline("kelime").fit(["güzel 👍", "kötü !", "güzel 👍", "kötü !"], ["P", "N", "P", "N"])
    vocab = pipe.named_steps["tfidf"].vocabulary_
    assert "👍" in vocab and "!" in vocab
