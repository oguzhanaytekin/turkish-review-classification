import pandas as pd
import pytest

from src.cleaning import (
    check_no_leakage,
    group_by_text,
    llm_subset_mask,
    separate_clean_and_hard,
    stratified_split,
)


@pytest.fixture
def tiny():
    rows = [
        ("Güzel ürün", "Neutral"), ("güzel ürün", "Neutral"), ("GÜZEL ÜRÜN.", "Neutral"),
        ("güzel ürün!", "Positive"),                        # çelişki, çoğunluk Neutral
        ("idare eder", "Neutral"), ("İdare eder", "Negative"),  # çelişki, eşitlik
        ("harika", "Positive"), ("Harika!!", "Positive"),       # temiz, 2 kopya
        ("berbat", "Negative"),                               # temiz, 1 kopya
        ("!!!", "Positive"),                                  # normalleştirince boş
    ]
    return pd.DataFrame({"id": range(len(rows)), "text": [t for t, _ in rows], "label": [l for _, l in rows]})


def test_gruplama_ve_bos_metin(tiny):
    groups, labels, n_empty = group_by_text(tiny)
    assert n_empty == 1
    assert labels == ["Negative", "Neutral", "Positive"]
    assert set(groups["norm"]) == {"güzel ürün", "idare eder", "harika", "berbat"}


def test_temiz_ve_zor_ayrimi(tiny):
    clean, hard = separate_clean_and_hard(*group_by_text(tiny)[:2])
    assert set(clean["norm"]) == {"harika", "berbat"}
    assert clean.set_index("norm").loc["harika", "n_kopya"] == 2
    assert set(hard["norm"]) == {"güzel ürün", "idare eder"}


def test_yumusak_etiket_ve_cogunluk(tiny):
    _, hard = separate_clean_and_hard(*group_by_text(tiny)[:2])
    h = hard.set_index("norm")
    assert h.loc["güzel ürün", "p_Neutral"] == 0.75
    assert h.loc["güzel ürün", "cogunluk_etiketi"] == "Neutral"
    assert bool(h.loc["idare eder", "esitlik"]) is True
    assert pd.isna(h.loc["idare eder", "cogunluk_etiketi"])
    probs = hard[["p_Negative", "p_Neutral", "p_Positive"]].sum(axis=1)
    assert (probs.round(3) == 1.0).all()


@pytest.fixture
def clean_big():
    labels = ["Negative", "Neutral", "Positive"] * 200
    return pd.DataFrame({"id": range(600), "text": [f"yorum {i}" for i in range(600)],
                         "norm": [f"yorum {i}" for i in range(600)], "label": labels, "n_kopya": 1})


def test_bolme_oranlari_ve_sinif_dengesi(clean_big):
    train, val, test = stratified_split(clean_big)
    assert (len(train), len(val), len(test)) == (480, 60, 60)
    for part in (train, val, test):
        assert part["label"].value_counts().nunique() == 1  # her sınıftan eşit sayıda


def test_bolme_tekrarlanabilir(clean_big):
    a = stratified_split(clean_big)[2]["id"].tolist()
    b = stratified_split(clean_big)[2]["id"].tolist()
    assert a == b  # aynı seed -> aynı test seti


def test_llm_alt_kume(clean_big):
    test = stratified_split(clean_big)[2]
    mask = llm_subset_mask(test, n_per_class=5)
    assert mask.sum() == 15
    assert test[mask]["label"].value_counts().tolist() == [5, 5, 5]


def test_sizinti_kontrolu_yakalar():
    a = pd.DataFrame({"norm": ["x", "y"]})
    b = pd.DataFrame({"norm": ["y", "z"]})
    with pytest.raises(AssertionError, match="Sızıntı"):
        check_no_leakage({"train": a, "test": b})
    check_no_leakage({"train": a, "test": pd.DataFrame({"norm": ["z"]})})  # hata yok
