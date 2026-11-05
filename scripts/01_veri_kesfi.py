"""
T-01: TRSAv1 veri keşfi ve tekrar analizi.

Çalıştırma (proje kök klasöründen):
    python scripts/01_veri_kesfi.py

Ne yapar?
  1. data/raw/TRSAv1.csv yoksa Hugging Face'ten indirir.
  2. Temel tablo: satır sayısı, boş değerler, sınıf dağılımı, yorum uzunlukları.
  3. Tekrar analizi: birebir aynı metinler ve normalleştirilmiş tekrarlar.
  4. Çelişkili etiketler: aynı (normalleştirilmiş) metin, farklı sınıf.
  5. Tekrarlar atılınca sınıf dengesi (üç temizlik senaryosu).
  6. Her sınıftan 5 rastgele örnek.

Çıktılar (reports/ altında):
  t01_rapor.md           ekranda gördüğün raporun aynısı
  t01_ozet.json          ana sayılar (sonraki adımlar okuyabilsin diye)
  t01_en_cok_tekrar.csv  en çok tekrar eden metinler
  t01_celiskiler.csv     etiketi çelişen metinlerin tamamı
  t01_ornekler.csv       her sınıftan rastgele örnekler

Bu betik veriyi DEĞİŞTİRMEZ, sadece ölçer. Temizlik kuralına
bu çıktılara bakıp birlikte karar vereceğiz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_io import DEFAULT_CSV, REPORTS, ensure_data, read_csv_robust  # noqa: E402
from src.report import Report, pct, short, use_utf8_stdout  # noqa: E402
from src.text_utils import normalize_text  # noqa: E402

SEED = 42  # rastgele örneklerin her çalıştırmada aynı çıkması için

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 90)


# --------------------------------------------------------------------------
# Yardımcılar (Report, short, pct artık src/report.py içinde)
# --------------------------------------------------------------------------
def label_breakdown(row: pd.Series) -> str:
    """Bir satırdaki sınıf sayılarını 'Negative:12, Notr:3' biçiminde yazar."""
    return ", ".join(f"{k}:{int(v)}" for k, v in row.items() if v > 0)


# --------------------------------------------------------------------------
# Kolon tespiti (veriyi indirme/okuma src/data_io.py içinde)
# --------------------------------------------------------------------------
def detect_columns(df: pd.DataFrame, text_col: str | None, label_col: str | None) -> tuple[str, str]:
    """Metin kolonu: en uzun ortalama metne sahip yazı kolonu.
    Etiket kolonu: 2–10 farklı değeri olan kolon."""
    for col in (text_col, label_col):
        if col and col not in df.columns:
            sys.exit(f"'{col}' kolonu yok. Mevcut kolonlar: {list(df.columns)}")

    def is_texty(s: pd.Series) -> bool:
        return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

    label_cands = [c for c in df.columns if 1 < df[c].nunique(dropna=True) <= 10]
    text_cands = [c for c in df.columns if is_texty(df[c]) and c not in label_cands]

    label = label_col or (label_cands[0] if len(label_cands) == 1 else None)
    text = text_col or (
        max(text_cands, key=lambda c: df[c].astype(str).str.len().mean()) if text_cands else None
    )
    if label is None or text is None:
        sys.exit(
            f"Kolonlar otomatik bulunamadı. Mevcut kolonlar: {list(df.columns)}\n"
            "Elle belirt: python scripts/01_veri_kesfi.py --text-col <metin> --label-col <etiket>"
        )
    return text, label


# --------------------------------------------------------------------------
# Analiz adımları
# --------------------------------------------------------------------------
def dup_stats(s: pd.Series) -> dict:
    counts = s.value_counts()
    repeated = counts[counts > 1]
    return {
        "satir": int(len(s)),
        "tekil_metin": int(len(counts)),
        "fazlalik_satir": int(len(s) - len(counts)),  # silinirse kaybolacak satır
        "tekrar_eden_metin": int(len(repeated)),      # en az 2 kez geçen farklı metin
        "tekrar_grubundaki_satir": int(repeated.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="T-01: TRSAv1 veri keşfi")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--text-col")
    parser.add_argument("--label-col")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    use_utf8_stdout()
    REPORTS.mkdir(exist_ok=True)
    ensure_data(args.csv, allow_download=not args.no_download)
    raw_df, encoding, sep = read_csv_robust(args.csv)
    text_col, label_col = detect_columns(raw_df, args.text_col, args.label_col)

    r = Report()
    r.p("# T-01 Veri Keşfi Raporu (otomatik üretildi)")

    # ---------------------------------------------------------------- 0
    r.h("0. Dosya")
    r.p(f"Dosya: {args.csv.name} | kodlama: {encoding} | ayırıcı: {sep!r}")
    r.p(f"Kolonlar ve tipleri: {dict(raw_df.dtypes.astype(str))}")
    r.p(f"Seçilen kolonlar -> metin: '{text_col}', etiket: '{label_col}'")
    r.p("İlk 3 satır:")
    r.table(raw_df.head(3).astype(str).apply(lambda col: col.map(short)))

    # ---------------------------------------------------------------- 1
    r.h("1. Temel tablo")
    n_total = len(raw_df)
    null_text = int(raw_df[text_col].isna().sum())
    null_label = int(raw_df[label_col].isna().sum())
    d = raw_df.loc[raw_df[text_col].notna() & raw_df[label_col].notna(), [text_col, label_col]].copy()
    d.columns = ["text", "label"]
    d["text"] = d["text"].astype(str)
    raw_label_values = d["label"].map(repr).value_counts()  # repr: sondaki boşlukları görünür yapar
    d["label"] = d["label"].astype(str).str.strip()
    blank_text = int(d["text"].str.strip().eq("").sum())

    r.p(f"Toplam satır: {n_total:,}")
    r.p(f"Boş metin (NaN): {null_text:,} | boş etiket (NaN): {null_label:,} | sadece boşluktan oluşan metin: {blank_text:,}")
    r.p("\nHam etiket değerleri (tırnak içinde, boşluklar görünsün diye):")
    r.table(raw_label_values)
    if d["label"].nunique() != len(raw_label_values):
        r.p("DİKKAT: Etiketlerin başında/sonunda boşluk var; temizlenince sınıf sayısı değişiyor.")

    labels = sorted(d["label"].unique())
    class_dist = d["label"].value_counts().reindex(labels)
    r.p("\nSınıf dağılımı:")
    r.table(pd.DataFrame({"satir": class_dist, "oran": (class_dist / class_dist.sum()).map(lambda x: f"%{100 * x:.1f}")}))

    d["kelime"] = d["text"].str.split().str.len()
    d["karakter"] = d["text"].str.len()
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    words = d.groupby("label")["kelime"].quantile(q).unstack()
    words.columns = ["%5", "%25", "medyan", "%75", "%95"]
    words["ortalama"] = d.groupby("label")["kelime"].mean().round(1)
    words["max"] = d.groupby("label")["kelime"].max()
    r.p("\nKelime sayısı dağılımı (sınıfa göre):")
    r.table(words)
    chars = d.groupby("label")["karakter"].agg(["median", "mean", "max"]).round(1)
    chars.columns = ["medyan", "ortalama", "max"]
    r.p("\nKarakter sayısı (sınıfa göre):")
    r.table(chars)
    very_short = d.assign(k1=d["kelime"] <= 1, k2=d["kelime"] <= 2).groupby("label")[["k1", "k2"]].sum()
    very_short.columns = ["<=1 kelime", "<=2 kelime"]
    r.p("\nÇok kısa yorumlar:")
    r.table(very_short)

    # ---------------------------------------------------------------- 2
    r.h("2. Tekrar analizi")
    d["norm"] = d["text"].map(normalize_text)
    empty_norm = int(d["norm"].eq("").sum())
    exact = dup_stats(d["text"])
    norm = dup_stats(d.loc[d["norm"] != "", "norm"])
    comp = pd.DataFrame({"birebir": exact, "normalleştirilmiş": norm})
    r.p("Normalleştirme: Türkçe küçük harf + noktalama/emoji silme + boşluk sadeleştirme (src/text_utils.py)")
    r.table(comp)
    r.p(f"\nBirebir tekrarların payı: {pct(exact['fazlalik_satir'], exact['satir'])}")
    r.p(f"Normalleştirilmiş tekrarların payı: {pct(norm['fazlalik_satir'], norm['satir'])}")
    r.p(f"Normalleştirme ek olarak {norm['fazlalik_satir'] - exact['fazlalik_satir']:,} tekrar yakaladı.")
    r.p(f"Normalleştirince tamamen boşalan metin (sadece noktalama/emoji): {empty_norm:,}")

    # norm x label sayım matrisi: sonraki tüm adımlar bunu kullanıyor
    nd = d[d["norm"] != ""]
    counts = nd.groupby(["norm", "label"]).size().unstack(fill_value=0)
    counts["toplam"] = counts[labels].sum(axis=1)
    counts["sinif_sayisi"] = (counts[labels] > 0).sum(axis=1)
    example_text = nd.groupby("norm")["text"].first()
    variants = nd.groupby("norm")["text"].nunique()

    top = counts[counts["toplam"] > 1].sort_values("toplam", ascending=False)
    top_out = pd.DataFrame({
        "norm_metin": top.index,
        "toplam": top["toplam"].values,
        "siniflar": top[labels].apply(label_breakdown, axis=1).values,
        "farkli_yazim": variants.reindex(top.index).values,
        "ornek_ham_metin": example_text.reindex(top.index).values,
    })
    top_out.head(100).to_csv(REPORTS / "t01_en_cok_tekrar.csv", index=False, encoding="utf-8-sig")
    r.p("\nEn çok tekrar eden 15 metin (normalleştirilmiş):")
    show = top_out.head(15)[["norm_metin", "toplam", "siniflar"]].copy()
    show["norm_metin"] = show["norm_metin"].map(lambda t: short(t, 60))
    r.table(show.set_index("norm_metin"))

    if len(top) and variants.reindex(top.index).max() > 1:
        key = variants.reindex(top.index).idxmax()
        forms = nd.loc[nd["norm"] == key, "text"].value_counts().head(6)
        r.p(f"\nNormalleştirmenin birleştirdiği yazımlara örnek ('{short(key, 50)}', {int(variants[key])} farklı yazım):")
        for form, c in forms.items():
            r.p(f"  {c:>5} x  {short(repr(form), 80)}")

    # ---------------------------------------------------------------- 3
    r.h("3. Çelişkili etiketler")
    conflicts = counts[counts["sinif_sayisi"] > 1].sort_values("toplam", ascending=False)
    exact_conflict_groups = int((d.groupby("text")["label"].nunique() > 1).sum())
    rows_in_conflict = int(conflicts["toplam"].sum())
    r.p(f"Birebir aynı metin, farklı etiket: {exact_conflict_groups:,} metin")
    r.p(f"Normalleştirilmiş metin, farklı etiket: {len(conflicts):,} metin")
    r.p(f"Bu metinlere ait satır sayısı: {rows_in_conflict:,} (tüm verideki oranı: {pct(rows_in_conflict, len(nd))})")

    if len(conflicts):
        pairs = conflicts[labels].gt(0).apply(lambda row: " + ".join(row.index[row.values]), axis=1).value_counts()
        r.p("\nHangi sınıflar birbiriyle çelişiyor (metin sayısı):")
        r.table(pairs)

        conf_words = pd.Series(conflicts.index).str.split().str.len()
        all_words = pd.Series(counts.index).str.split().str.len()
        r.p(f"\nÇelişkili metinlerin medyan kelime sayısı: {conf_words.median():.0f} "
            f"(tüm tekil metinlerde: {all_words.median():.0f})")

        conf_out = pd.DataFrame({
            "norm_metin": conflicts.index,
            "toplam": conflicts["toplam"].values,
            **{lab: conflicts[lab].values for lab in labels},
            "ornek_ham_metin": example_text.reindex(conflicts.index).values,
        })
        conf_out.to_csv(REPORTS / "t01_celiskiler.csv", index=False, encoding="utf-8-sig")
        r.p("\nEn kalabalık 15 çelişkili metin:")
        show = conf_out.head(15)[["norm_metin", *labels]].copy()
        show["norm_metin"] = show["norm_metin"].map(lambda t: short(t, 60))
        r.table(show.set_index("norm_metin"))

    # ---------------------------------------------------------------- 4
    r.h("4. Temizlik senaryoları ve sınıf dengesi")
    r.p(f"(Normalleştirince boşalan {empty_norm:,} satır tüm senaryolarda hariç.)")
    first_label = nd.drop_duplicates("norm").set_index("norm")["label"]
    top_count = counts[labels].max(axis=1)
    n_top = counts[labels].eq(top_count, axis=0).sum(axis=1)
    majority = counts.loc[n_top == 1, labels].idxmax(axis=1)
    ties = int((n_top > 1).sum())

    scenarios = {
        "ham": nd["label"].value_counts(),
        "A: tekilleştir, ilkini tut": first_label.value_counts(),
        "B: çelişkileri at + tekilleştir": first_label[counts["sinif_sayisi"].reindex(first_label.index) == 1].value_counts(),
        "C: çoğunluk etiketi": majority.value_counts(),
    }
    bal = pd.DataFrame(scenarios).reindex(labels).fillna(0).astype(int).T
    bal["toplam"] = bal[labels].sum(axis=1)
    bal["min/max"] = (bal[labels].min(axis=1) / bal[labels].max(axis=1)).round(3)
    r.table(bal)
    r.p(f"\nC senaryosunda eşitlik yüzünden atılan metin: {ties:,}")
    r.p("min/max = en küçük sınıf / en büyük sınıf. 1'e yakınsa denge korunuyor.")

    # ---------------------------------------------------------------- 5
    r.h("5. Her sınıftan 5 rastgele örnek")
    samples = d.groupby("label").sample(n=5, random_state=SEED)[["label", "kelime", "text"]]
    samples.to_csv(REPORTS / "t01_ornekler.csv", index=False, encoding="utf-8-sig")
    for lab in labels:
        r.p(f"\n[{lab}]")
        for _, row in samples[samples["label"] == lab].iterrows():
            r.p(f"  ({row['kelime']:>3} kelime) {short(row['text'], 160)}")

    # ---------------------------------------------------------------- özet
    summary = {
        "dosya": args.csv.name,
        "kodlama": encoding,
        "metin_kolonu": text_col,
        "etiket_kolonu": label_col,
        "toplam_satir": n_total,
        "bos_metin": null_text,
        "bos_etiket": null_label,
        "normalize_sonrasi_bos": empty_norm,
        "sinif_dagilimi": {k: int(v) for k, v in class_dist.items()},
        "birebir_tekrar": exact,
        "normalize_tekrar": norm,
        "celiskili_metin": int(len(conflicts)),
        "celiskili_satir": rows_in_conflict,
        "senaryolar": {k: {lab: int(bal.loc[k, lab]) for lab in labels} for k in bal.index},
    }
    (REPORTS / "t01_ozet.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    r.save(REPORTS / "t01_rapor.md")
    print(f"\nRapor kaydedildi: {REPORTS / 't01_rapor.md'}")


if __name__ == "__main__":
    main()
