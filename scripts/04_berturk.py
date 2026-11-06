"""
T-04: BERTurk fine-tune (GPU gerekir; Colab'da çalıştırılmak üzere yazıldı).

Çalıştırma (proje kök klasöründen, önce 02 çalışmış olmalı):
    python scripts/04_berturk.py --out-dir /content/drive/MyDrive/trc_runs/berturk_seed42
    python scripts/04_berturk.py --smoke --out-dir /content/drive/MyDrive/trc_runs/smoke   # 2-3 dk'lık deneme

Ne yapar?
  1. Bölmenin T-02 ile birebir aynı olduğunu kontrol eder (aynı değilse durur).
  2. dbmdz/bert-base-turkish-cased modelini train'de eğitir; her epoch sonunda val
     macro-F1'e bakar, en iyi epoch'u tutar. Checkpoint'ler --out-dir'e (Drive) yazılır;
     Colab oturumu koparsa aynı komut kaldığı yerden devam eder.
  3. Val setinde sıcaklık (temperature scaling) öğrenir.
  4. Testte BİR KEZ ölçer: T-03 ile aynı metrikler + kalibrasyon öncesi/sonrası + hız.
  5. Zor örneklerde model olasılıklarını insan dağılımıyla karşılaştırır.
  6. Sonuçları baseline (reports/t03_ozet.json) ile yan yana koyar.

Çıktılar (reports/ altına ve --out-dir/reports altına kopya):
  t04_rapor.md, t04_ozet.json, t04_test_tahminleri.csv, t04_zor_tahminleri.csv
  --out-dir/best/  (en iyi model; eğitim bitince ara checkpoint'ler silinir)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration import fit_temperature, nll, softmax  # noqa: E402
from src.data_io import REPORTS, load_processed  # noqa: E402
from src.metrics import (  # noqa: E402
    classification_summary,
    entropy,
    expected_calibration_error,
    soft_cross_entropy,
)
from src.report import Report, short, use_utf8_stdout  # noqa: E402

pd.set_option("display.width", 200)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T-04: BERTurk fine-tune")
    p.add_argument("--model", default="dbmdz/bert-base-turkish-cased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "berturk")
    p.add_argument("--smoke", action="store_true", help="küçük alt kümeyle hızlı uçtan uca deneme")
    return p.parse_args()


def check_split(train, val, test, hard) -> None:
    """Colab'da yeniden üretilen bölme, yerelde T-02/T-03'te kullanılanla aynı mı?"""
    path = REPORTS / "t02_ozet.json"
    if not path.exists():
        print("UYARI: reports/t02_ozet.json yok, bölme kontrolü atlandı.")
        return
    ref = json.loads(path.read_text(encoding="utf-8"))
    now = {"train": len(train), "val": len(val), "test": len(test)}
    if now != ref["bolme"] or len(hard) != ref["zor_ornekler"]:
        sys.exit(f"Bölme T-02'den farklı! Şimdi: {now}, zor {len(hard)} | T-02: {ref['bolme']}, "
                 f"zor {ref['zor_ornekler']}. Baseline ile karşılaştırma geçersiz olur, durduruldu.")


def ids_hash(ids) -> str:
    return hashlib.sha1(",".join(map(str, sorted(ids))).encode()).hexdigest()[:12]


def main() -> None:
    args = parse_args()
    use_utf8_stdout()

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint
    from sklearn.metrics import f1_score

    train, val, test, hard = (load_processed(n) for n in ("train", "val", "test", "zor_ornekler"))
    check_split(train, val, test, hard)
    test_hash = ids_hash(test["id"])
    suffix = ""
    if args.smoke:
        suffix = "_smoke"
        train = train.groupby("label").sample(n=700, random_state=args.seed)
        val = val.groupby("label").sample(n=150, random_state=args.seed)
        test = test.sample(n=1000, random_state=args.seed)
        args.epochs = 1
    if not torch.cuda.is_available():
        print("UYARI: GPU bulunamadı. Colab'da Çalışma zamanı > Çalışma zamanı türünü değiştir > T4 GPU.")

    labels = sorted(train["label"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    ckpt_dir = args.out_dir / "checkpoints"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ veri
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    class ReviewDataset(torch.utils.data.Dataset):
        """Metinleri bir kez token'a çevirir; dolgu (padding) batch oluşturulurken yapılır."""

        def __init__(self, texts: pd.Series, y: pd.Series | None = None):
            self.enc = tokenizer(list(texts.astype(str)), truncation=True, max_length=args.max_len)
            self.y = None if y is None else y.map(label2id).to_numpy()

        def __len__(self):
            return len(self.enc["input_ids"])

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            if self.y is not None:
                item["labels"] = int(self.y[i])
            return item

    print("Token'lara çevriliyor ...", flush=True)
    ds_train, ds_val = ReviewDataset(train["text"], train["label"]), ReviewDataset(val["text"], val["label"])
    full_lens = pd.Series([len(x) for x in tokenizer(list(test["text"].astype(str)))["input_ids"]])

    # ------------------------------------------------------------------ eğitim
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id
    )

    def compute_metrics(p):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        pred = logits.argmax(-1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro"),
                "accuracy": float((pred == p.label_ids).mean())}

    steps = math.ceil(len(ds_train) / args.batch) * args.epochs
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        weight_decay=0.01,
        warmup_steps=int(0.06 * steps),          # ilk %6 adımda öğrenme oranı yavaşça artar
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,             # en iyi val macro-F1'li epoch geri yüklenir
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),          # yarı hassasiyet: T4'te ~2 kat hızlı
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
    )
    last = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    if last:
        print(f"Yarım kalan eğitim bulundu, devam ediliyor: {last}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last)
    train_secs = time.time() - t0
    best_dir = args.out_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # ------------------------------------------------------------------ tahmin
    def get_logits(df: pd.DataFrame) -> np.ndarray:
        out = trainer.predict(ReviewDataset(df["text"]))
        return out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions

    val_logits = get_logits(val)
    t0 = time.time()
    test_logits = get_logits(test)
    ms_per_review = 1000 * (time.time() - t0) / len(test)
    hard_logits = get_logits(hard)

    y_val = val["label"].map(label2id).to_numpy()
    y_test = test["label"].map(label2id).to_numpy()
    T = fit_temperature(val_logits, y_val)
    p_raw, p_cal = softmax(test_logits), softmax(test_logits / T)
    pred = np.array(labels)[p_raw.argmax(axis=1)]          # sıcaklık argmax'ı değiştirmez
    s = classification_summary(test["label"], pred, labels)
    neg = s["sinif_bazinda"].loc["Negative"]
    ece_raw, ece_cal = expected_calibration_error(p_raw, y_test), expected_calibration_error(p_cal, y_test)
    ll_raw, ll_cal = nll(test_logits, y_test, 1.0), nll(test_logits, y_test, T)
    llm_mask = test["llm_alt_kume"].astype(bool).to_numpy()
    llm_f1 = classification_summary(test["label"][llm_mask], pred[llm_mask], labels)["macro_f1"] if llm_mask.any() else float("nan")

    human = hard[[f"p_{c}" for c in labels]].to_numpy(dtype=float)
    human = human / human.sum(axis=1, keepdims=True)
    q_raw, q_cal = softmax(hard_logits), softmax(hard_logits / T)
    ce_raw, ce_cal = soft_cross_entropy(human, q_raw).mean(), soft_cross_entropy(human, q_cal).mean()
    ce_floor = entropy(human).mean()
    ce_uniform = soft_cross_entropy(human, np.full_like(human, 1 / len(labels))).mean()
    hpred = np.array(labels)[q_raw.argmax(axis=1)]
    has_major = hard["cogunluk_etiketi"].notna().to_numpy()
    maj_acc = float((hpred[has_major] == hard["cogunluk_etiketi"].to_numpy()[has_major]).mean())

    # ------------------------------------------------------------------ rapor
    r = Report()
    r.p(f"# T-04 BERTurk Raporu{' (SMOKE: küçük deneme, sonuçlar anlamsız)' if args.smoke else ''} (otomatik üretildi)")

    r.h("1. Kurulum")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "yok (CPU)"
    r.p(f"Model: {args.model} | GPU: {gpu} | seed {args.seed} | epoch {args.epochs} | lr {args.lr} | "
        f"batch {args.batch} | max_len {args.max_len}")
    r.p(f"Veri: train {len(train):,} / val {len(val):,} / test {len(test):,} / zor {len(hard):,} | test kimlik özeti {test_hash}")
    r.p(f"{args.max_len} token'dan uzun test yorumu (kesildi): {(full_lens > args.max_len).sum():,} "
        f"(%{100 * (full_lens > args.max_len).mean():.2f}) | medyan uzunluk {full_lens.median():.0f} token")
    r.p(f"Eğitim süresi: {train_secs / 60:.1f} dk | çıkarım hızı: {ms_per_review:.2f} ms/yorum")

    r.h("2. Epoch'lar (val)")
    hist = pd.DataFrame([h for h in trainer.state.log_history if "eval_macro_f1" in h])
    if len(hist):
        r.table(hist[["epoch", "eval_loss", "eval_macro_f1", "eval_accuracy"]].round(4).set_index("epoch"))
    r.p(f"En iyi checkpoint: {trainer.state.best_model_checkpoint}")

    r.h("3. Test seti (bir kez)")
    r.p(f"ANA METRİK  macro-F1 : {s['macro_f1']:.4f}")
    r.p(f"accuracy             : {s['accuracy']:.4f}")
    r.p(f"İŞ METRİĞİ  Negative recall    : {neg['recall']:.4f}")
    r.p(f"            Negative precision : {neg['precision']:.4f}")
    r.p(f"Öğrenilen sıcaklık T = {T:.3f}  ({'>1: model aşırı emin, olasılıklar yumuşatıldı' if T > 1 else '<1: model çekingen, olasılıklar keskinleştirildi'})")
    if T > 9.9 or T < 0.06:
        r.p("DİKKAT: Sıcaklık arama sınırına dayandı; model val setinde neredeyse rastgele tahmin ediyor olabilir.")
    r.p(f"ECE      : ham {ece_raw:.4f} -> kalibre {ece_cal:.4f}")
    r.p(f"log loss : ham {ll_raw:.4f} -> kalibre {ll_cal:.4f}")
    r.p("\nSınıf bazında:")
    r.table(s["sinif_bazinda"].round(4))
    r.p("\nKarışıklık matrisi (satır = gerçek, sütun = tahmin):")
    r.table(pd.crosstab(test["label"], pd.Series(pred, name="tahmin", index=test.index)))
    r.p(f"\nLLM alt kümesinde macro-F1: {llm_f1:.4f}")

    r.h("4. Zor örnekler")
    r.table(pd.Series({
        "alt sınır (insanların kendi belirsizliği)": round(ce_floor, 4),
        "BERTurk (ham olasılık)": round(ce_raw, 4),
        "BERTurk (kalibre)": round(ce_cal, 4),
        "her şeye eşit olasılık (1/3)": round(ce_uniform, 4),
    }, name="cross_entropy"))
    r.p(f"\nÇoğunluk etiketiyle uyum: {maj_acc:.4f}")
    r.p(f"Ortalama güven (ham): test {p_raw.max(1).mean():.3f} | zor {q_raw.max(1).mean():.3f}")
    r.p(f"Ortalama güven (kalibre): test {p_cal.max(1).mean():.3f} | zor {q_cal.max(1).mean():.3f}")
    comp = hard[["text", "n_kopya"]].copy()
    for i, c in enumerate(labels):
        comp[f"insan_{c[:3]}"] = human[:, i].round(2)
        comp[f"model_{c[:3]}"] = q_cal[:, i].round(2)
    comp["text"] = comp["text"].map(lambda t: short(t, 28))
    r.p("\nEn kalabalık 10 zor örnek (model = kalibre olasılık):")
    r.table(comp.head(10).set_index("text"))
    r.p("\nEşitlik olan 5 örnek:")
    r.table(comp[hard["esitlik"].astype(bool).to_numpy()].head(5).set_index("text"))

    r.h("5. Baseline (T-03) ile karşılaştırma")
    base_path = REPORTS / "t03_ozet.json"
    berturk = {
        "test macro-F1": s["macro_f1"], "test accuracy": s["accuracy"],
        "Negative recall": float(neg["recall"]), "Negative precision": float(neg["precision"]),
        "ECE (kalibrasyonsuz)": ece_raw, "LLM alt kümesi macro-F1": llm_f1,
        "zor örnek cross-entropy (kalibrasyonsuz)": ce_raw,
    }
    if base_path.exists() and not args.smoke:
        b = json.loads(base_path.read_text(encoding="utf-8"))
        base = {
            "test macro-F1": b["test"]["macro_f1"], "test accuracy": b["test"]["accuracy"],
            "Negative recall": b["test"]["negative_recall"], "Negative precision": b["test"]["negative_precision"],
            "ECE (kalibrasyonsuz)": b["test"]["ece"], "LLM alt kümesi macro-F1": b["test"]["llm_alt_kume_macro_f1"],
            "zor örnek cross-entropy (kalibrasyonsuz)": b["zor"]["ce_model"],
        }
        cmp_df = pd.DataFrame({"baseline": base, "BERTurk": berturk}).round(4)
        cmp_df["fark"] = (cmp_df["BERTurk"] - cmp_df["baseline"]).round(4)
        r.table(cmp_df)
        r.p("\nNot: Test setinde ±0,6 puanın, LLM alt kümesinde ±1,9 puanın altındaki farklar gürültü sınırında.")
    else:
        r.p("Smoke modunda ya da t03_ozet.json olmadan karşılaştırma yapılmaz.")

    # ------------------------------------------------------------------ kaydet
    out = test[["id", "text", "label"]].copy()
    out["tahmin"] = pred
    for i, c in enumerate(labels):
        out[f"p_{c}"] = p_cal[:, i].round(4)
    out["llm_alt_kume"] = llm_mask
    out.to_csv(REPORTS / f"t04{suffix}_test_tahminleri.csv", index=False, encoding="utf-8-sig")
    hout = hard[["id", "text", "n_kopya", "cogunluk_etiketi", "esitlik"]].copy()
    hout["tahmin"] = hpred
    for i, c in enumerate(labels):
        hout[f"insan_{c}"] = human[:, i].round(4)
        hout[f"model_{c}"] = q_cal[:, i].round(4)
    hout.to_csv(REPORTS / f"t04{suffix}_zor_tahminleri.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": args.model, "seed": args.seed, "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "max_len": args.max_len, "smoke": args.smoke, "gpu": gpu, "test_kimlik_ozeti": test_hash,
        "egitim_dk": round(train_secs / 60, 1), "ms_per_yorum": round(ms_per_review, 3), "sicaklik": round(T, 4),
        "test": {
            "macro_f1": round(s["macro_f1"], 4), "accuracy": round(s["accuracy"], 4),
            "negative_recall": round(float(neg["recall"]), 4), "negative_precision": round(float(neg["precision"]), 4),
            "ece_ham": round(ece_raw, 4), "ece_kalibre": round(ece_cal, 4),
            "log_loss_ham": round(ll_raw, 4), "log_loss_kalibre": round(ll_cal, 4),
            "llm_alt_kume_macro_f1": round(float(llm_f1), 4),
        },
        "zor": {"ce_ham": round(float(ce_raw), 4), "ce_kalibre": round(float(ce_cal), 4),
                "ce_alt_sinir": round(float(ce_floor), 4), "ce_esit": round(float(ce_uniform), 4),
                "cogunluk_uyumu": round(maj_acc, 4)},
    }
    (REPORTS / f"t04{suffix}_ozet.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    r.save(REPORTS / f"t04{suffix}_rapor.md")

    copy_dir = args.out_dir / "reports"
    copy_dir.mkdir(exist_ok=True)
    for f in REPORTS.glob(f"t04{suffix}_*"):
        if args.smoke or "_smoke" not in f.name:  # tam koşuda smoke dosyaları kopyalanmasın
            shutil.copy2(f, copy_dir / f.name)
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # en iyi model best/ altında; ara checkpoint'ler Drive'da yer kaplamasın
    print(f"\nRapor: {REPORTS / f't04{suffix}_rapor.md'} (kopyası: {copy_dir})")


if __name__ == "__main__":
    main()
