import json

import numpy as np
import pandas as pd
import pytest

from src.llm import aggregate, build_prompt, parse_response, pick_few_shot


def test_istem_yorumlari_numaralar_ve_kurali_soylemez():
    p = build_prompt(["Güzel ürün", "Kötü"])
    assert '1. "Güzel ürün"' in p and '2. "Kötü"' in p
    assert "Neutral" in p and "hafif" not in p.lower()  # zero-shot: etiketleme kuralı söylenmiyor


def test_few_shot_yalnizca_verilen_setten_ve_dengeli():
    train = pd.DataFrame({"text": [f"yorum {i}" for i in range(30)], "label": ["Negative", "Neutral", "Positive"] * 10})
    few = pick_few_shot(train, per_class=2)
    assert few["label"].value_counts().tolist() == [2, 2, 2]
    assert set(few["text"]) <= set(train["text"])
    assert "Örnekler" in build_prompt(["x"], few)


def test_cevap_ayristirma_cerceve_ve_normalize():
    raw = "```json\n" + json.dumps([
        {"no": 2, "etiket": "positive", "olasilik": {"Negative": 1, "Neutral": 1, "Positive": 2}},
        {"no": 1, "etiket": "Negative", "olasilik": {"Negative": 0.9, "Neutral": 0.1, "Positive": 0}},
    ]) + "\n```"
    out = parse_response(raw, 2)
    assert out[0]["etiket"] == "Negative" and out[1]["etiket"] == "Positive"  # numaraya göre sıralandı
    assert np.isclose(sum(out[1]["p"]), 1.0) and out[1]["p"][2] == 0.5


@pytest.mark.parametrize("raw", ["metin", "[]", '[{"no": 1, "etiket": "Harika"}]', '[{"no": 3, "etiket": "Neutral"}]'])
def test_bozuk_cevaplar_reddedilir(raw):
    with pytest.raises(ValueError):
        parse_response(raw, 1)


def test_birlestirme_cogunluk_ve_duzeltme():
    s = [{"etiket": "Neutral", "p": [0.1, 0.8, 0.1]}] * 3 + [{"etiket": "Positive", "p": [0.0, 0.3, 0.7]}] * 2
    a = aggregate(s)
    assert a["tahmin"] == "Neutral" and a["k"] == 5
    assert np.isclose(a["p_ornek"].sum(), 1.0) and a["p_ornek"][0] > 0  # hiç çıkmayan sınıf 0 değil
    assert np.isclose(a["p_ornek"][1], 3.5 / 6.5)


def test_birlestirme_esitlikte_sozel_olasilik_karar_verir():
    s = [{"etiket": "Neutral", "p": [0.0, 0.55, 0.45]}, {"etiket": "Positive", "p": [0.0, 0.2, 0.8]}]
    assert aggregate(s)["tahmin"] == "Positive"  # sözel ortalama: Neutral 0,375 < Positive 0,625


def test_kural_istemi_kurali_icerir_digerleri_icermez():
    assert "etiketleme kuralı" in build_prompt(["x"], rule=True)
    assert "etiketleme kuralı" not in build_prompt(["x"])
