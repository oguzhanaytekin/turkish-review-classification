"""T-02: Temizlik, zor örnekler seti ve bölme.

Karar (ayrıntı: docs/02_temizlik_bolme.md):
  1. Yorumlar normalleştirilmiş metne göre gruplanır (src/text_utils.normalize_text).
  2. Tek etiketli gruplar  -> ana veri; her gruptan 1 satır kalır (tekilleştirme).
  3. Çok etiketli gruplar  -> "zor örnekler" seti; ana veriye HİÇ girmez.
     Her biri için etiketleyici dağılımı (yumuşak etiket, p_*) ve çoğunluk
     etiketi (eşitlikte boş) saklanır. Bu set yalnızca değerlendirmede kullanılır.
  4. Ana veri stratified 80/10/10 bölünür; testten LLM için sabit bir alt küme çekilir.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.text_utils import normalize_text

SEED = 42


def group_by_text(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    """Her normalleştirilmiş metin için tek satır üretir.

    Girdi: id, text, label kolonları.
    Çıktı kolonları: norm, id, text (grubun ilk geçen ham metni), her sınıf için
    sayım (n_<sınıf>), n_kopya (toplam satır), n_sinif (kaç farklı etiket almış).
    Ayrıca sınıf listesi ve normalleştirince boşalan satır sayısı döner.
    """
    d = df.assign(norm=df["text"].map(normalize_text))
    n_empty = int((d["norm"] == "").sum())
    d = d[d["norm"] != ""]

    labels = sorted(d["label"].unique())
    counts = (
        d.groupby(["norm", "label"]).size().unstack(fill_value=0).reindex(columns=labels, fill_value=0)
    )
    first = d.drop_duplicates("norm").set_index("norm")[["id", "text"]]

    groups = first.join(counts.add_prefix("n_"))
    groups["n_kopya"] = counts.sum(axis=1)
    groups["n_sinif"] = (counts > 0).sum(axis=1)
    return groups.reset_index(), labels, n_empty


def separate_clean_and_hard(groups: pd.DataFrame, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grupları ana veri (tek etiketli) ve zor örnekler (çok etiketli) olarak ayırır."""
    count_cols = [f"n_{lab}" for lab in labels]

    clean = groups[groups["n_sinif"] == 1].copy()
    clean["label"] = clean[count_cols].idxmax(axis=1).str.removeprefix("n_")
    clean = clean[["id", "text", "norm", "label", "n_kopya"]].reset_index(drop=True)

    hard = groups[groups["n_sinif"] > 1].copy()
    counts = hard[count_cols]
    for lab, col in zip(labels, count_cols):
        hard[f"p_{lab}"] = (hard[col] / hard["n_kopya"]).round(4)
    top = counts.max(axis=1)
    tie = counts.eq(top, axis=0).sum(axis=1) > 1
    hard["esitlik"] = tie
    hard["cogunluk_etiketi"] = counts.idxmax(axis=1).str.removeprefix("n_").where(~tie)
    hard = hard[
        ["id", "text", "norm", "n_kopya", *count_cols, *[f"p_{lab}" for lab in labels], "cogunluk_etiketi", "esitlik"]
    ].sort_values("n_kopya", ascending=False).reset_index(drop=True)
    return clean, hard


def stratified_split(
    clean: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sınıf oranlarını koruyarak train/val/test böler (varsayılan 80/10/10).

    İki adımda: önce train ve "geri kalan" (%20), sonra geri kalanı val ve test olarak ikiye.
    """
    holdout = val_frac + test_frac
    train, rest = train_test_split(clean, test_size=holdout, stratify=clean["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=test_frac / holdout, stratify=rest["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def llm_subset_mask(test: pd.DataFrame, n_per_class: int = 500, seed: int = SEED) -> pd.Series:
    """Testten her sınıftan n_per_class örnek seçer (LLM maliyetini sınırlamak için).

    True/False dizisi döner; test satırlarıyla aynı sırada.
    """
    picked = []
    for _, g in test.groupby("label"):
        picked.extend(g.sample(n=min(n_per_class, len(g)), random_state=seed).index)
    return test.index.isin(picked)


def check_no_leakage(parts: dict[str, pd.DataFrame]) -> None:
    """Hiçbir normalleştirilmiş metin iki farklı parçada bulunmamalı. Varsa hata fırlatır."""
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(parts[a]["norm"]) & set(parts[b]["norm"])
            if overlap:
                example = next(iter(overlap))
                raise AssertionError(f"Sızıntı: {len(overlap)} metin hem '{a}' hem '{b}' içinde (ör. {example!r})")
