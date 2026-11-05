"""
T-03: Baseline, TF-IDF + lojistik regresyon.

Çalıştırma (proje kök klasöründen, önce 02 çalışmış olmalı):
    python scripts/03_baseline.py

Ne yapar?
  1. Türkçe karaktersiz yazımın ("basarili") ne kadar yaygın olduğunu ölçer.
  2. Ablation: {ham, hafif, tam} ön işleme × {kelime, karakter} temsil = 6 model.
     Hepsi train'de eğitilir, VAL'de macro-F1 ile karşılaştırılır. Test'e dokunulmaz.
  3. En iyi kombinasyon test setinde BİR KEZ ölçülür: macro-F1, accuracy,
     Negative precision/recall, ECE, karışıklık matrisi, LLM alt kümesi skoru.
  4. Zor örnekler: model olasılıkları ile insan dağılımı (cross-entropy) karşılaştırılır.
  5. Modelin her sınıf için en çok baktığı özellikleri listeler.

Süre: 6 model eğitildiği için birkaç dakika sürebilir.

Çıktılar:
  reports/t03_rapor.md, t03_ozet.json, t03_test_tahminleri.csv, t03_zor_tahminleri.csv
  models/baseline.joblib   (git'e girmez)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline import ANALYZERS, build_pipeline, top_features  # noqa: E402
from src.data_io import PROCESSED, REPORTS  # noqa: E402
from src.metrics import (  # noqa: E402
    classification_summary,
    entropy,
    expected_calibration_error,
    soft_cross_entropy,
)
from src.preprocess import MODES, ascii_writing_stats, preprocess  # noqa: E402
from src.report import Report, pct, short, use_utf8_stdout  # noqa: E402

MODELS = ROOT / "models"
pd.set_option("display.width", 200)


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED / f"{name}.csv"
    if not path.exists():
        sys.exit(f"{path} yok. Önce çalıştır: python scripts/02_temizlik_bolme.py")
    # keep_default_na=False: "NA", "null", "None" yazan yorumlar boş değer sanılmasın
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def main() -> None:
    use_utf8_stdout()
    MODELS.mkdir(exist_ok=True)
    train, val, test, hard = (load_split(n) for n in ("train", "val", "test", "zor_ornekler"))
    labels = sorted(train["label"].unique())

    r = Report()
    r.p("# T-03 Baseline Raporu: TF-IDF + Lojistik Regresyon (otomatik üretildi)")

    # ------------------------------------------------------------------ 1
    r.h("1. Türkçe karaktersiz yazım ne kadar yaygın? (train)")
    a = ascii_writing_stats(train["text"])
    r.p(f"5+ kelimelik yorumlardan hiç Türkçe karakter içermeyen: {a['uzun_ve_turkce_karaktersiz']:,} / "
        f"{a['uzun_yorum_sayisi']:,} ({pct(a['uzun_ve_turkce_karaktersiz'], a['uzun_yorum_sayisi'])})")
    r.p(f"Sık kelimelerin ASCII biçimini içeren yorum (guzel, cok, urun...): {a['ascii_kelime_iceren']:,} "
        f"({pct(a['ascii_kelime_iceren'], len(train))})")
    r.p("En sık ASCII biçimler: " + ", ".join(f"{k}: {v}" for k, v in a["en_sik_ascii_kelimeler"].items()))
    r.p("Örnek Türkçe karaktersiz uzun yorumlar:")
    for t in a["ornekler"]:
        r.p(f"  {short(t, 100)!r}")

    # ------------------------------------------------------------------ 2
    r.h("2. Ablation: ön işleme × temsil (VAL setinde)")
    rows, best = [], None
    for mode in MODES:
        x_tr, x_va = preprocess(train["text"], mode), preprocess(val["text"], mode)
        for an in ANALYZERS:
            print(f"  eğitiliyor: {mode:>5} + {an:<8} ...", end="", flush=True)
            t0 = time.time()
            pipe = build_pipeline(an).fit(x_tr, train["label"])
            s = classification_summary(val["label"], pipe.predict(x_va), labels)
            secs = time.time() - t0
            print(f" macro-F1 {s['macro_f1']:.4f} ({secs:.0f} sn)")
            rows.append({
                "on_isleme": mode, "temsil": an, "val_macro_f1": round(s["macro_f1"], 4),
                "val_accuracy": round(s["accuracy"], 4),
                "ozellik_sayisi": len(pipe.named_steps["tfidf"].vocabulary_), "sure_sn": round(secs),
            })
            if best is None or s["macro_f1"] > best[0]:
                best = (s["macro_f1"], mode, an, pipe)
    abl = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    r.table(abl)
    _, mode, an, pipe = best
    gap = abl.loc[0, "val_macro_f1"] - abl.loc[1, "val_macro_f1"]
    r.p(f"\nSeçilen: {mode} + {an}. İkinciyle fark: {100 * gap:.2f} puan.")
    if gap < 0.005:
        r.p("DİKKAT: Fark 0,5 puandan küçük; val setinin belirsizliği (yaklaşık ±0,6 puan) içinde kalıyor. "
            "İkisi pratikte eşit sayılmalı.")

    # ------------------------------------------------------------------ 3
    r.h(f"3. Test seti (bir kez): {mode} + {an}")
    x_te = preprocess(test["text"], mode)
    classes = list(pipe.classes_)
    proba = pipe.predict_proba(x_te)
    pred = np.array(classes)[proba.argmax(axis=1)]
    y_idx = test["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    s = classification_summary(test["label"], pred, labels)
    ece = expected_calibration_error(proba, y_idx)
    ll = log_loss(test["label"], proba, labels=classes)
    neg = s["sinif_bazinda"].loc["Negative"] if "Negative" in labels else None

    r.p(f"ANA METRİK  macro-F1 : {s['macro_f1']:.4f}")
    r.p(f"accuracy             : {s['accuracy']:.4f}")
    if neg is not None:
        r.p(f"İŞ METRİĞİ  Negative recall    : {neg['recall']:.4f}  (olumsuz yorumların yakalanan oranı)")
        r.p(f"            Negative precision : {neg['precision']:.4f}  (olumsuz denenlerin gerçekten olumsuz oranı)")
    r.p(f"KALİBRASYON ECE      : {ece:.4f}  (0 = model 'yüzde X eminim' dediğinde gerçekten yüzde X haklı)")
    r.p(f"log loss             : {ll:.4f}")
    r.p("\nSınıf bazında:")
    r.table(s["sinif_bazinda"].round(4))
    r.p("\nKarışıklık matrisi (satır = gerçek, sütun = tahmin):")
    r.table(pd.crosstab(test["label"], pd.Series(pred, name="tahmin", index=test.index)))

    llm_mask = test["llm_alt_kume"].astype(bool).to_numpy()
    s_llm = classification_summary(test["label"][llm_mask], pred[llm_mask], labels)
    r.p(f"\nLLM alt kümesinde (1.500, dengeli) macro-F1: {s_llm['macro_f1']:.4f}  "
        "(LLM ile karşılaştırmada bu sayı kullanılacak)")

    # ------------------------------------------------------------------ 4
    r.h("4. Zor örnekler: model olasılıkları ile insan dağılımı")
    q = pipe.predict_proba(preprocess(hard["text"], mode))
    human = hard[[f"p_{c}" for c in classes]].to_numpy(dtype=float)
    human = human / human.sum(axis=1, keepdims=True)
    prior = train["label"].value_counts(normalize=True).reindex(classes).to_numpy()
    ce_model = soft_cross_entropy(human, q).mean()
    ce_floor = entropy(human).mean()
    ce_uniform = soft_cross_entropy(human, np.full_like(human, 1 / len(classes))).mean()
    ce_prior = soft_cross_entropy(human, np.tile(prior, (len(human), 1))).mean()
    r.p("Cross-entropy (küçük = insanlara daha yakın):")
    r.table(pd.Series({
        "alt sınır (insanların kendi belirsizliği)": round(ce_floor, 4),
        "baseline model": round(ce_model, 4),
        "her şeye eşit olasılık (1/3)": round(ce_uniform, 4),
        "train sınıf oranlarını tahmin et": round(ce_prior, 4),
    }, name="cross_entropy"))

    hpred = np.array(classes)[q.argmax(axis=1)]
    has_major = hard["cogunluk_etiketi"].notna().to_numpy()
    maj_acc = (hpred[has_major] == hard["cogunluk_etiketi"].to_numpy()[has_major]).mean()
    r.p(f"\nÇoğunluk etiketi olan {has_major.sum():,} zor örnekte, modelin en olası sınıfı çoğunlukla aynı: {maj_acc:.4f}")
    r.p(f"Modelin ortalama güveni: test {proba.max(axis=1).mean():.3f} | zor örnekler {q.max(axis=1).mean():.3f} "
        "(zor örneklerde daha düşükse model belirsizliği seziyor demektir)")

    comp = hard[["text", "n_kopya"]].copy()
    for i, c in enumerate(classes):
        comp[f"insan_{c[:3]}"] = human[:, i].round(2)
        comp[f"model_{c[:3]}"] = q[:, i].round(2)
    comp["text"] = comp["text"].map(lambda t: short(t, 28))
    r.p("\nEn kalabalık 10 zor örnek:")
    r.table(comp.head(10).set_index("text"))
    ties = comp[hard["esitlik"].astype(bool).to_numpy()]
    r.p("\nEşitlik olan 5 örnek:")
    r.table(ties.head(5).set_index("text"))

    # ------------------------------------------------------------------ 5
    r.h("5. Model neye bakıyor? (her sınıfta ağırlığı en yüksek 15 özellik)")
    r.table(top_features(pipe, n=15))

    # ------------------------------------------------------------------ kaydet
    out = test[["id", "text", "label"]].copy()
    out["tahmin"] = pred
    for i, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, i].round(4)
    out["llm_alt_kume"] = llm_mask
    out.to_csv(REPORTS / "t03_test_tahminleri.csv", index=False, encoding="utf-8-sig")
    hout = hard[["id", "text", "n_kopya", "cogunluk_etiketi", "esitlik"]].copy()
    hout["tahmin"] = hpred
    for i, c in enumerate(classes):
        hout[f"insan_{c}"] = human[:, i].round(4)
        hout[f"model_{c}"] = q[:, i].round(4)
    hout.to_csv(REPORTS / "t03_zor_tahminleri.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"pipeline": pipe, "on_isleme": mode, "temsil": an}, MODELS / "baseline.joblib")

    summary = {
        "secilen": {"on_isleme": mode, "temsil": an},
        "ablation": rows,
        "test": {
            "macro_f1": round(s["macro_f1"], 4), "accuracy": round(s["accuracy"], 4),
            "negative_recall": round(float(neg["recall"]), 4) if neg is not None else None,
            "negative_precision": round(float(neg["precision"]), 4) if neg is not None else None,
            "ece": round(ece, 4), "log_loss": round(float(ll), 4),
            "llm_alt_kume_macro_f1": round(s_llm["macro_f1"], 4),
        },
        "zor": {
            "ce_model": round(float(ce_model), 4), "ce_alt_sinir": round(float(ce_floor), 4),
            "ce_esit": round(float(ce_uniform), 4), "ce_prior": round(float(ce_prior), 4),
            "cogunluk_uyumu": round(float(maj_acc), 4),
        },
        "ascii_yazim": {k: v for k, v in a.items() if k != "ornekler"},
    }
    (REPORTS / "t03_ozet.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    r.save(REPORTS / "t03_rapor.md")
    print(f"\nRapor kaydedildi: {REPORTS / 't03_rapor.md'}")


if __name__ == "__main__":
    main()
