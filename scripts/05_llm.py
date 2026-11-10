"""
T-05: LLM ile sınıflandırma (Gemini, ücretsiz katman).

Önce: reponun kök klasöründe .env dosyasında GEMINI_API_KEY=... olmalı; 02 çalışmış olmalı.

Çalıştırma:
    pip install -r requirements-llm.txt
    python scripts/05_llm.py --pilot          # ~5 dk: kota, toplu gönderim, logprobs desteği ölçülür
    python scripts/05_llm.py --k 5 --batch 20 # tam koşu; kota dolarsa aynı komut kaldığı yerden devam eder
    python scripts/05_llm.py --ara-rapor --k 5 --batch 20  # API'ye gitmeden, biten işlerin sonuçları
    python scripts/05_llm.py --dry-run ...    # API'ye gitmeden, sahte cevaplarla kodu dener

Ne yapar? (tam koşu)
  1. LLM alt kümesi (1.500 test yorumu) ve zor örnekler (1.049) için, zero-shot ve few-shot istemle,
     her yorumu k kez sınıflandırır. Her istekte --batch kadar yorum gider; sıra her örneklemede karışır.
  2. Her cevap data/llm_cache/ altına yazılır (git'e girmez). Yarıda kalırsa tekrar istek atılmaz.
  3. Etiket: k örneklemenin çoğunluğu. Olasılık iki türlü: örnekleme sıklığı ve sözel (LLM'in söylediği).
  4. Baseline ve BERTurk ile aynı metriklerle karşılaştırır; maliyet ve hızı raporlar.

Çıktılar: reports/t05_rapor.md, t05_ozet.json, t05_tahminler.csv (pilotta t05_pilot_*)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_io import REPORTS, load_processed  # noqa: E402
from src.llm import LABELS, MODES, aggregate, build_prompt, parse_response, pick_few_shot  # noqa: E402
from src.metrics import (  # noqa: E402
    classification_summary,
    entropy,
    expected_calibration_error,
    soft_cross_entropy,
)
from src.report import Report, short, use_utf8_stdout  # noqa: E402

CACHE = ROOT / "data" / "llm_cache"
pd.set_option("display.width", 200)


class QuotaExhausted(Exception):
    """Günlük kota doldu: beklemenin anlamı yok, yarın devam edilmeli."""


# --------------------------------------------------------------------------
# API çağırıcıları
# --------------------------------------------------------------------------
def load_api_key() -> str | None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():  # -sig: başta BOM varsa yok sayar
            if line.strip().startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return os.environ.get("GEMINI_API_KEY")


class Stats:
    def __init__(self) -> None:
        self.istek = self.hata = self.kota_bekleme = self.girdi_token = self.cikti_token = 0
        self.sure_sn = 0.0

    def as_dict(self) -> dict:
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in vars(self).items()}


class GeminiCaller:
    """generate_content etrafında: hız sınırı (rpm), 429'da bekleme, günlük kotada durma."""

    def __init__(self, model: str, rpm: float, temperature: float) -> None:
        from google import genai
        from google.genai import types

        key = load_api_key()
        if not key:
            sys.exit("GEMINI_API_KEY bulunamadı. Reponun kök klasöründeki .env dosyasını kontrol et.")
        self.client, self.types = genai.Client(api_key=key), types
        self.model, self.gap, self.temperature = model, 60.0 / rpm, temperature
        self.last, self.stats = 0.0, Stats()

    def list_models(self, contains: str = "flash-lite") -> list[str]:
        return sorted(m.name for m in self.client.models.list() if contains in m.name)

    def __call__(self, prompt: str, extra_config: dict | None = None):
        cfg = {"temperature": self.temperature, "response_mime_type": "application/json", **(extra_config or {})}
        for attempt in range(6):
            wait = self.gap - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            t0 = time.time()
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=self.types.GenerateContentConfig(**cfg)
                )
            except Exception as e:  # SDK hata sınıfları sürüme göre değişebiliyor; mesaja bakıyoruz
                self.last = time.time()
                msg, code = str(e), getattr(e, "code", None)
                if code == 429 or "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    if "PerDay" in msg or "per day" in msg.lower():
                        raise QuotaExhausted(msg[:300])
                    m = re.search(r"retry(?:Delay)?\D{0,10}(\d+(?:\.\d+)?)s", msg)
                    delay = float(m.group(1)) + 1 if m else 20.0 * (attempt + 1)
                    self.stats.kota_bekleme += 1
                    print(f"    dakikalık kota: {delay:.0f} sn bekleniyor", flush=True)
                    time.sleep(delay)
                    continue
                if code in (500, 503) or "UNAVAILABLE" in msg or "INTERNAL" in msg:
                    self.stats.hata += 1
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
            self.last = time.time()
            self.stats.istek += 1
            self.stats.sure_sn += self.last - t0
            u = getattr(resp, "usage_metadata", None)
            self.stats.girdi_token += (getattr(u, "prompt_token_count", 0) or 0) if u else 0
            self.stats.cikti_token += (getattr(u, "candidates_token_count", 0) or 0) if u else 0
            return resp
        raise RuntimeError("6 denemede de cevap alınamadı")


class FakeCaller:
    """--dry-run: API'ye gitmeden kodu denemek için rastgele ama geçerli cevaplar üretir."""

    def __init__(self, *_, **__) -> None:
        self.stats, self.rng = Stats(), random.Random(0)

    def list_models(self, contains: str = "flash-lite") -> list[str]:
        return ["models/sahte-flash-lite"]

    def __call__(self, prompt: str, extra_config: dict | None = None):
        n = len(re.findall(r"^\d+\. ", prompt.split("\nYorumlar:")[-1], flags=re.M))
        out = []
        for i in range(1, n + 1):
            p = np.array([self.rng.random() for _ in LABELS]) ** 3
            p = p / p.sum()
            out.append({"no": i, "etiket": LABELS[int(p.argmax())], "olasilik": dict(zip(LABELS, p.round(3).tolist()))})
        if self.rng.random() < 0.03:  # ara sıra bozuk cevap: hata yolu da denensin
            out = out[:-1]
        self.stats.istek += 1
        self.stats.girdi_token += len(prompt) // 4
        self.stats.cikti_token += 40 * n
        return type("Resp", (), {"text": json.dumps(out, ensure_ascii=False)})()


# --------------------------------------------------------------------------
# Bir işi (split × istem türü) k örneklemeyle çalıştırma
# --------------------------------------------------------------------------
def run_job(caller, df: pd.DataFrame, few, k: int, batch: int, cache_path: Path, label: str, rule: bool = False) -> dict:
    """df'deki her yorum için k örneklemenin cevaplarını döndürür: {id: [ {etiket, p}, ... ]}.

    Cevaplar cache_path'e satır satır yazılır; yeniden çalıştırınca yapılmış istekler atlanır.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["key"]] = rec

    n_batches = math.ceil(len(df) / batch)
    todo = k * n_batches - sum(1 for key in done if int(key.split(":")[0]) < k)
    print(f"  {label}: {len(df):,} yorum × {k} örnekleme, {n_batches} istek/örnekleme, kalan {max(todo, 0)} istek", flush=True)
    with cache_path.open("a", encoding="utf-8") as fh:
        for s in range(k):
            order = df.sample(frac=1, random_state=1000 + s)  # her örneklemede farklı sıra ve komşular
            for b in range(n_batches):
                key = f"{s}:{b}"
                if key in done:
                    continue
                chunk = order.iloc[b * batch:(b + 1) * batch]
                parsed = None
                for _ in range(2):
                    try:
                        parsed = parse_response(caller(build_prompt(chunk["text"].tolist(), few, rule)).text, len(chunk))
                        break
                    except ValueError:
                        caller.stats.hata += 1
                if parsed is None:  # toplu cevap iki kez bozuk geldi: yorumları tek tek sor
                    parsed = []
                    for t in chunk["text"]:
                        try:
                            parsed.append(parse_response(caller(build_prompt([t], few, rule)).text, 1)[0])
                        except ValueError:
                            caller.stats.hata += 1
                            parsed.append(None)
                rec = {"key": key, "ids": chunk["id"].tolist(), "cevap": parsed}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[key] = rec
                if (b + 1) % 10 == 0:
                    print(f"    örnekleme {s + 1}/{k}, istek {b + 1}/{n_batches}", flush=True)

    return read_cache(cache_path, k)[0]


def read_cache(cache_path: Path, k: int) -> tuple[dict, int]:
    """Önbellekteki cevapları {id: [cevaplar]} olarak döndürür; ayrıca tamamlanmış istek sayısını.

    API'ye hiç gitmez; ara rapor da bunu kullanıyor.
    """
    out: dict = {}
    n_done = 0
    if not cache_path.exists():
        return out, 0
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if int(rec["key"].split(":")[0]) >= k:
            continue
        n_done += 1
        for i, ans in zip(rec["ids"], rec["cevap"]):
            if ans is not None:
                out.setdefault(i, []).append(ans)
    return out, n_done


def to_frame(df: pd.DataFrame, answers: dict) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        samples = answers.get(r["id"], [])
        if not samples:
            rows.append({"id": r["id"], "tahmin": None, "k": 0})
            continue
        a = aggregate(samples)
        rows.append({
            "id": r["id"], "tahmin": a["tahmin"], "k": a["k"],
            **{f"po_{c}": a["p_ornek"][i] for i, c in enumerate(LABELS)},
            **{f"ps_{c}": a["p_sozel"][i] for i, c in enumerate(LABELS)},
        })
    return df.merge(pd.DataFrame(rows), on="id", how="left")


# --------------------------------------------------------------------------
# Pilot
# --------------------------------------------------------------------------
def pilot(caller, test: pd.DataFrame, few, model: str, batch: int, r: Report) -> None:
    sub = test[test["llm_alt_kume"].astype(bool)].sample(n=50, random_state=7)
    r.h("1. Kullanılabilir flash-lite modelleri")
    try:
        names = caller.list_models()
        r.p(", ".join(names) or "(liste boş)")
        if not any(n.endswith("/" + model) or n == model for n in names) and not model.endswith("__dryrun"):
            r.p(f"DURDURULDU: Seçili model ({model}) listede yok. --model ile listeden birini seç.")
            return
    except Exception as e:
        r.p(f"Model listesi alınamadı: {str(e)[:200]}")

    r.h("2. Tek tek ve toplu gönderim karşılaştırması (50 yorum, zero-shot, 1 örnekleme)")
    t0 = time.time()
    one = run_job(caller, sub, None, 1, 1, CACHE / f"pilot__{model}__b1.jsonl", "tek tek")
    t1 = time.time()
    many = run_job(caller, sub, None, 1, batch, CACHE / f"pilot__{model}__b{batch}.jsonl", f"toplu ({batch})")
    t2 = time.time()
    f1, fb = to_frame(sub, one), to_frame(sub, many)
    both = f1["tahmin"].notna() & fb["tahmin"].notna()
    agree = (f1.loc[both, "tahmin"].to_numpy() == fb.loc[both, "tahmin"].to_numpy()).mean()
    r.p(f"Tek tek: {f1['tahmin'].notna().sum()}/50 cevap, {t1 - t0:.0f} sn | "
        f"toplu: {fb['tahmin'].notna().sum()}/50 cevap, {t2 - t1:.0f} sn")
    r.p(f"İki yöntemin aynı etiketi verme oranı: {agree:.2f}  (sıcaklık 1 olduğu için 1,00 beklenmez)")
    r.p(f"Doğruluk: tek tek {(f1['tahmin'] == f1['label']).mean():.2f} | toplu {(fb['tahmin'] == fb['label']).mean():.2f} "
        "(50 yorumda belirsizlik ±0,13; kaba fikir verir)")

    r.h("3. Logprobs desteği")
    try:
        resp = caller(build_prompt(sub["text"].head(1).tolist()), {"response_logprobs": True, "logprobs": 3})
        has = getattr(resp, "candidates", None) and getattr(resp.candidates[0], "logprobs_result", None) is not None
        r.p("Destekleniyor." if has else "Hata vermedi ama logprobs dönmedi.")
    except Exception as e:
        r.p(f"Desteklenmiyor: {str(e)[:200]}")

    r.h("4. Tam koşu tahmini")
    st = caller.stats
    per_call = st.sure_sn / max(st.istek, 1)
    r.p(f"İstek: {st.istek} | ortalama cevap süresi {per_call:.1f} sn | dakikalık kota beklemesi {st.kota_bekleme} | "
        f"hata {st.hata} | girdi token {st.girdi_token:,} | çıktı token {st.cikti_token:,}")
    for k in (3, 5, 10):
        calls = (math.ceil(1500 / batch) + math.ceil(1049 / batch)) * 2 * k
        r.p(f"k={k}, batch={batch}: {calls:,} istek")
    r.p("Günlük istek sınırını AI Studio'daki kota sayfasından gör; istek sayısı onu aşıyorsa koşu birkaç güne bölünür.")


# --------------------------------------------------------------------------
# Ara rapor (API'ye gitmez, sadece önbellek)
# --------------------------------------------------------------------------
def interim_report(test: pd.DataFrame, hard: pd.DataFrame, model_tag: str, args, r: Report) -> None:
    """Tamamlanan işlerin sonuçlarını, yarım kalanların ilerlemesini gösterir."""
    sub = test[test["llm_alt_kume"].astype(bool)].reset_index(drop=True)
    human = hard[[f"p_{c}" for c in LABELS]].to_numpy(dtype=float)
    human = human / human.sum(axis=1, keepdims=True)

    r.h("1. İlerleme")
    status = {}
    for split, df in (("test", sub), ("zor", hard)):
        for mode in MODES:
            path = CACHE / f"{model_tag}__{split}__{mode}__b{args.batch}.jsonl"
            ans, n_done = read_cache(path, args.k)
            need = args.k * math.ceil(len(df) / args.batch)
            status[(split, mode)] = (ans, n_done >= need)
            r.p(f"{split}/{mode}: {n_done}/{need} istek ({100 * n_done / need:.0f}%)"
                + ("  TAMAMLANDI" if n_done >= need else ""))

    r.h("2. Tamamlanan işlerin sonuçları")
    base = {}
    for name, f in (("baseline", "t03_ozet.json"), ("BERTurk", "t04_ozet.json")):
        pth = REPORTS / f
        if pth.exists():
            base[name] = json.loads(pth.read_text(encoding="utf-8"))
    any_done = False
    for (split, mode), (ans, complete) in status.items():
        if not complete:
            continue
        any_done = True
        df = sub if split == "test" else hard
        fr = to_frame(df, ans)
        ok = fr["tahmin"].notna().to_numpy()
        r.p(f"\n### {split} / {mode}-shot (k={args.k})")
        if split == "test":
            s = classification_summary(fr.loc[ok, "label"], fr.loc[ok, "tahmin"], list(LABELS))
            neg = s["sinif_bazinda"].loc["Negative"]
            y = fr.loc[ok, "label"].map({c: i for i, c in enumerate(LABELS)}).to_numpy()
            po = fr.loc[ok, [f"po_{c}" for c in LABELS]].to_numpy()
            ps = fr.loc[ok, [f"ps_{c}" for c in LABELS]].to_numpy()
            r.p(f"Cevaplanan: {ok.sum()}/{len(fr)} | macro-F1 {s['macro_f1']:.4f} | accuracy {s['accuracy']:.4f} | "
                f"Negative recall {neg['recall']:.4f} / precision {neg['precision']:.4f}")
            r.p(f"ECE: örnekleme {expected_calibration_error(po, y):.4f} | sözel {expected_calibration_error(ps, y):.4f} | "
                f"ortalama güven: örnekleme {po.max(1).mean():.3f}, sözel {ps.max(1).mean():.3f}")
            r.table(s["sinif_bazinda"].round(4))
            r.table(pd.crosstab(fr.loc[ok, "label"], fr.loc[ok, "tahmin"].rename("tahmin")))
            if base:
                cmp = {n: j["test"]["llm_alt_kume_macro_f1"] for n, j in base.items()}
                cmp[f"LLM {mode}"] = s["macro_f1"]
                r.p("LLM alt kümesinde macro-F1 (±1,9 puan gürültü sınırı):")
                r.table(pd.Series(cmp, name="macro_f1").round(4))
        else:
            hpo = fr.loc[ok, [f"po_{c}" for c in LABELS]].to_numpy()
            hps = fr.loc[ok, [f"ps_{c}" for c in LABELS]].to_numpy()
            r.p(f"Zor CE: örnekleme {soft_cross_entropy(human[ok], hpo).mean():.4f} | sözel "
                f"{soft_cross_entropy(human[ok], hps).mean():.4f} | alt sınır {entropy(human).mean():.4f} | eşit {np.log(3):.4f}")
    if not any_done:
        r.p("Henüz tamamlanan iş yok.")


# --------------------------------------------------------------------------
# Tam koşu raporu
# --------------------------------------------------------------------------
def evaluate(res: dict, hard: pd.DataFrame, args, caller, few, r: Report) -> dict:
    summary = {"model": args.model, "k": args.k, "batch": args.batch, "sicaklik": args.temperature}
    r.h("1. Kurulum")
    r.p(f"Model: {args.model} | k={args.k} | batch={args.batch} | sıcaklık {args.temperature} | few-shot: {len(few)} örnek (train'den)")
    st = caller.stats.as_dict()
    r.p(f"Bu çalıştırmadaki istekler: {st}")

    human = hard[[f"p_{c}" for c in LABELS]].to_numpy(dtype=float)
    human = human / human.sum(axis=1, keepdims=True)
    rows = {}
    for mi, mode in enumerate(MODES, start=1):
        t, h = res[("test", mode)], res[("zor", mode)]
        ok = t["tahmin"].notna().to_numpy()
        s = classification_summary(t.loc[ok, "label"], t.loc[ok, "tahmin"], list(LABELS))
        neg = s["sinif_bazinda"].loc["Negative"]
        y = t.loc[ok, "label"].map({c: i for i, c in enumerate(LABELS)}).to_numpy()
        po = t.loc[ok, [f"po_{c}" for c in LABELS]].to_numpy()
        ps = t.loc[ok, [f"ps_{c}" for c in LABELS]].to_numpy()
        hok = h["tahmin"].notna().to_numpy()
        hpo = h.loc[hok, [f"po_{c}" for c in LABELS]].to_numpy()
        hps = h.loc[hok, [f"ps_{c}" for c in LABELS]].to_numpy()
        hm = human[hok]
        maj = h.loc[hok, "cogunluk_etiketi"]
        rows[mode] = {
            "cevaplanan (test)": f"{ok.sum()}/{len(t)}",
            "macro-F1": s["macro_f1"], "accuracy": s["accuracy"],
            "Negative recall": float(neg["recall"]), "Negative precision": float(neg["precision"]),
            "ECE (örnekleme)": expected_calibration_error(po, y), "ECE (sözel)": expected_calibration_error(ps, y),
            "zor CE (örnekleme)": float(soft_cross_entropy(hm, hpo).mean()),
            "zor CE (sözel)": float(soft_cross_entropy(hm, hps).mean()),
            "çoğunlukla uyum (zor)": float((h.loc[hok, "tahmin"][maj.notna()] == maj[maj.notna()]).mean()),
            "ort. güven test (örnekleme)": float(po.max(1).mean()), "ort. güven zor (örnekleme)": float(hpo.max(1).mean()),
            "ort. güven test (sözel)": float(ps.max(1).mean()), "ort. güven zor (sözel)": float(hps.max(1).mean()),
        }
        r.h(f"2.{mi} {mode}: sınıf bazında ve karışıklık matrisi (LLM alt kümesi)")
        r.table(s["sinif_bazinda"].round(4))
        r.table(pd.crosstab(t.loc[ok, "label"], t.loc[ok, "tahmin"].rename("tahmin")))

    r.h("3. Üç istem türünün özeti (zero = örneksiz, few = rastgele örnekli, kural = kural açıkça yazılı)")
    tbl = pd.DataFrame(rows)
    r.table(tbl.apply(lambda col: col.map(lambda v: round(v, 4) if isinstance(v, float) else v)))
    r.p(f"\nZor örneklerde alt sınır (insanların kendi belirsizliği): {entropy(human).mean():.4f} | "
        f"eşit olasılık: {np.log(3):.4f}")

    r.h("4. Üç yaklaşımın karşılaştırması (LLM alt kümesi, 1.500 yorum)")
    cmp = {}
    for name, f in (("baseline", "t03_ozet.json"), ("BERTurk", "t04_ozet.json")):
        p = REPORTS / f
        if p.exists():
            j = json.loads(p.read_text(encoding="utf-8"))
            zor = j["zor"]
            cmp[name] = {"macro-F1 (LLM alt kümesi)": j["test"]["llm_alt_kume_macro_f1"],
                         "zor CE (en iyi olasılık)": zor.get("ce_kalibre", zor.get("ce_model"))}
    for mode in MODES:
        cmp[f"LLM {mode}"] = {"macro-F1 (LLM alt kümesi)": rows[mode]["macro-F1"],
                                   "zor CE (en iyi olasılık)": min(rows[mode]["zor CE (örnekleme)"], rows[mode]["zor CE (sözel)"])}
    r.table(pd.DataFrame(cmp).round(4))
    r.p("\nNot: 1.500 yorumda ±1,9 puanın altındaki macro-F1 farkları gürültü sınırında.")

    r.h("5. Zor örnekler: insanlar ve her istem türü (örnekleme olasılığı)")
    show = hard[["text", "n_kopya"]].copy()
    for i, c in enumerate(LABELS):
        show[f"insan_{c[:3]}"] = human[:, i].round(2)
    for mode in MODES:
        for c in LABELS:
            show[f"{mode}_{c[:3]}"] = res[("zor", mode)][f"po_{c}"].round(2).to_numpy()
    show["text"] = show["text"].map(lambda t: short(t, 22))
    r.table(show.head(12).set_index("text"))

    r.h("6. Maliyet ve hız")
    n_reviews = sum(len(v) for v in res.values())
    cost = caller.stats.girdi_token / 1e6 * args.price_in + caller.stats.cikti_token / 1e6 * args.price_out
    r.p(f"Bu çalıştırmada: {caller.stats.istek} istek, girdi {caller.stats.girdi_token:,} / çıktı {caller.stats.cikti_token:,} token.")
    r.p(f"Ücretli katman fiyatıyla (${args.price_in}/${args.price_out} / 1M token) yaklaşık maliyet: ${cost:.3f} (ücretsiz katmanda 0).")
    r.p(f"Not: Önbellekten gelen cevaplar bu sayılara dahil değil; koşu birden çok güne bölündüyse toplam daha yüksek. "
        f"Değerlendirilen yorum×istem sayısı: {n_reviews:,}.")

    summary.update({**{m: rows[m] for m in MODES}, "istekler": caller.stats.as_dict(),
                    "few_shot_ornekleri": few.to_dict(orient="records")})
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="T-05: LLM ile sınıflandırma")
    p.add_argument("--model", default="gemini-3.5-flash-lite")
    p.add_argument("--k", type=int, default=5, help="her yorum kaç kez sınıflandırılsın")
    p.add_argument("--batch", type=int, default=20, help="bir istekte kaç yorum")
    p.add_argument("--rpm", type=float, default=10, help="dakikada en fazla istek")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--price-in", type=float, default=0.30, help="ücretli katman, 1M girdi token $ (maliyet tahmini için)")
    p.add_argument("--price-out", type=float, default=2.50)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="API yerine sahte cevaplar")
    p.add_argument("--ara-rapor", action="store_true", help="API'ye gitmeden, önbellekteki cevaplarla ara rapor")
    args = p.parse_args()
    use_utf8_stdout()

    train, test, hard = (load_processed(n) for n in ("train", "test", "zor_ornekler"))
    few = pick_few_shot(train)
    model_tag = re.sub(r"[^\w.-]", "_", args.model) + ("__dryrun" if args.dry_run else "")
    r = Report()
    if args.ara_rapor:
        r.p(f"# T-05 Ara Rapor (otomatik üretildi) — model {args.model}, k={args.k}, batch={args.batch}")
        interim_report(test, hard, model_tag, args, r)
        r.save(REPORTS / "t05_ara_rapor.md")
        print(f"\nRapor: {REPORTS / 't05_ara_rapor.md'}")
        return
    caller = (FakeCaller if args.dry_run else GeminiCaller)(args.model, args.rpm, args.temperature)

    try:
        if args.pilot:
            r.p(f"# T-05 Pilot Raporu (otomatik üretildi) — model {args.model}")
            pilot(caller, test, None, model_tag, args.batch, r)
            r.save(REPORTS / "t05_pilot_rapor.md")
            print(f"\nRapor: {REPORTS / 't05_pilot_rapor.md'}")
            return

        sub = test[test["llm_alt_kume"].astype(bool)].reset_index(drop=True)
        res = {}
        # Sıra: önce test işleri (yeni eklenen "kural" sonucu erken görünsün), sonra zor örnekler
        for split, df in (("test", sub), ("zor", hard)):
            for mode in MODES:
                ans = run_job(caller, df, few if mode == "few" else None, args.k, args.batch,
                              CACHE / f"{model_tag}__{split}__{mode}__b{args.batch}.jsonl", f"{split}/{mode}",
                              rule=(mode == "kural"))
                res[(split, mode)] = to_frame(df, ans)
    except QuotaExhausted as e:
        print(f"\nGünlük kota doldu. Yapılan istekler kaydedildi; yarın AYNI komutu çalıştır, kaldığı yerden devam eder.\n{e}")
        return

    r.p(f"# T-05 LLM Raporu (otomatik üretildi){' — DRY RUN, sonuçlar anlamsız' if args.dry_run else ''}")
    summary = evaluate(res, hard, args, caller, few, r)
    out = pd.concat([d.assign(bolum=s, istem=m) for (s, m), d in res.items()], ignore_index=True)
    out.drop(columns=[c for c in out.columns if c.startswith("n_")], errors="ignore").to_csv(
        REPORTS / "t05_tahminler.csv", index=False, encoding="utf-8-sig")
    (REPORTS / "t05_ozet.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    r.save(REPORTS / "t05_rapor.md")
    print(f"\nRapor: {REPORTS / 't05_rapor.md'}")


if __name__ == "__main__":
    main()
