"""
T-02: Temizlik, zor örnekler seti ve train/val/test bölmesi.

Çalıştırma (proje kök klasöründen):
    python scripts/02_temizlik_bolme.py

Ne yapar?
  1. TRSAv1'i okur, metinleri normalleştirip gruplar.
  2. Tek etiketli metinler -> ana veri (tekilleştirilmiş).
     Çok etiketli (çelişkili) metinler -> zor örnekler seti (ana veriye girmez).
  3. Ana veriyi stratified 80/10/10 böler, testten LLM için 1.500'lük alt küme işaretler.
  4. Sızıntı kontrolü yapar: hiçbir metin iki parçada birden olmamalı.

Çıktılar:
  data/processed/train.csv, val.csv, test.csv, zor_ornekler.csv   (git'e girmez, betik yeniden üretir)
  reports/t02_rapor.md, reports/t02_ozet.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cleaning import (  # noqa: E402
    SEED,
    check_no_leakage,
    group_by_text,
    llm_subset_mask,
    separate_clean_and_hard,
    stratified_split,
)
from src.data_io import DEFAULT_CSV, PROCESSED, REPORTS, load_trsa  # noqa: E402
from src.report import Report, pct, short, use_utf8_stdout  # noqa: E402

pd.set_option("display.width", 200)


def class_table(parts: dict[str, pd.DataFrame], labels: list[str]) -> pd.DataFrame:
    """Her parçada sınıf başına satır sayısı ve oranı."""
    rows = {}
    for name, df in parts.items():
        vc = df["label"].value_counts().reindex(labels, fill_value=0)
        rows[name] = {**{lab: int(vc[lab]) for lab in labels}, "toplam": int(vc.sum())}
        rows[name].update({f"%{lab}": f"{100 * vc[lab] / vc.sum():.1f}" for lab in labels})
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser(description="T-02: temizlik ve bölme")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    use_utf8_stdout()
    REPORTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_trsa(args.csv, allow_download=not args.no_download)
    groups, labels, n_empty = group_by_text(df)
    clean, hard = separate_clean_and_hard(groups, labels)
    train, val, test = stratified_split(clean, seed=SEED)
    test["llm_alt_kume"] = llm_subset_mask(test, n_per_class=500, seed=SEED)
    check_no_leakage({"train": train, "val": val, "test": test, "zor": hard})

    for name, part in {"train": train, "val": val, "test": test, "zor_ornekler": hard}.items():
        part.to_csv(PROCESSED / f"{name}.csv", index=False, encoding="utf-8")

    # ------------------------------------------------------------------ rapor
    r = Report()
    r.p("# T-02 Temizlik ve Bölme Raporu (otomatik üretildi)")

    r.h("1. Satırlar nereye gitti?")
    n_ties = int(hard["esitlik"].sum())
    flow = pd.DataFrame(
        {
            "satir": [
                len(df),
                n_empty,
                len(df) - n_empty,
                int(clean["n_kopya"].sum() - len(clean)),
                int(hard["n_kopya"].sum()),
                len(clean),
            ]
        },
        index=[
            "ham veri",
            "- normalleştirince boşalan (atıldı)",
            "= kalan",
            "- tekrar eden kopyalar (atıldı)",
            "- zor örneklere giden satırlar",
            "= ana veri (tekil metin)",
        ],
    )
    r.table(flow)
    r.p(f"\nZor örnekler seti: {len(hard):,} tekil metin ({n_ties:,} tanesinde eşitlik var, çoğunluk etiketi yok)")

    r.h("2. Bölme: sınıf dağılımı (stratified 80/10/10, seed=42)")
    parts = {"ana veri": clean, "train": train, "val": val, "test": test, "test / LLM alt kümesi": test[test["llm_alt_kume"]]}
    r.table(class_table(parts, labels))
    r.p(f"\nOranlar: train {pct(len(train), len(clean))}, val {pct(len(val), len(clean))}, test {pct(len(test), len(clean))}")

    r.h("3. Sızıntı kontrolü")
    r.p("train / val / test / zor örnekler arasında ortak normalleştirilmiş metin: YOK (kontrol geçti)")
    multi = clean[clean["n_kopya"] > 1]
    r.p(f"Ana veride tekrarları birleştirilmiş metin: {len(multi):,} (en çok: {short(multi.sort_values('n_kopya').iloc[-1]['norm'], 40)!r}, "
        f"{int(multi['n_kopya'].max())} kopya)" if len(multi) else "Ana veride tekrarı olan metin yok.")

    r.h("4. Zor örnekler seti")
    pairs = hard[[f"n_{lab}" for lab in labels]].gt(0).apply(
        lambda row: " + ".join(lab for lab, on in zip(labels, row.values) if on), axis=1
    ).value_counts()
    r.p("Hangi sınıflar arasında çelişki var (metin sayısı):")
    r.table(pairs)
    r.p("\nÇoğunluk etiketi dağılımı (eşitlikler hariç):")
    r.table(hard["cogunluk_etiketi"].value_counts().reindex(labels, fill_value=0))
    r.p("\nEn kalabalık 10 zor örnek ve etiketleyici dağılımı (yumuşak etiket):")
    show = hard.head(10)[["norm", "n_kopya", *[f"p_{lab}" for lab in labels], "cogunluk_etiketi"]].copy()
    show["norm"] = show["norm"].map(lambda t: short(t, 35))
    r.table(show.set_index("norm"))
    r.p("\nÖrnek eşitlikler:" if n_ties else "\nEşitlik olan zor örnek yok.")
    for _, row in hard[hard["esitlik"]].head(5).iterrows():
        dist = ", ".join(f"{lab}:{int(row[f'n_{lab}'])}" for lab in labels if row[f"n_{lab}"] > 0)
        r.p(f"  {short(row['text'], 60)!r}  ->  {dist}")

    r.h("5. Üretilen dosyalar")
    for name in ("train", "val", "test", "zor_ornekler"):
        r.p(f"  data/processed/{name}.csv")

    summary = {
        "ham_satir": len(df),
        "normalize_bos": n_empty,
        "ana_veri": len(clean),
        "zor_ornekler": len(hard),
        "zor_esitlik": n_ties,
        "bolme": {k: int(len(v)) for k, v in {"train": train, "val": val, "test": test}.items()},
        "llm_alt_kume": int(test["llm_alt_kume"].sum()),
        "siniflar": labels,
        "seed": SEED,
    }
    (REPORTS / "t02_ozet.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    r.save(REPORTS / "t02_rapor.md")
    print(f"\nRapor kaydedildi: {REPORTS / 't02_rapor.md'}")


if __name__ == "__main__":
    main()
