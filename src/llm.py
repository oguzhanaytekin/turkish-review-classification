"""T-05: LLM ile sınıflandırma, istem (prompt) kurma, cevap ayrıştırma ve birleştirme.

Ağ bağlantısı gerektirmez; API çağrısı scripts/05_llm.py içinde. Böylece buradaki mantık
yerelde test edilebilir.

Karar (docs/05_llm.md):
  - zero-shot: Sadece görev tanımı. Etiketleme kuralından ("hafif övgü = Neutral") bahsedilmez.
  - few-shot : Aynı tanım + train setinden rastgele seçilmiş, sınıf başına eşit sayıda örnek.
  - kural    : Aynı tanım + etiketleme kuralının açık yazılı tarifi (örnek yok). T-05 ara sonuçlarından
               sonra eklendi: rastgele örnekler kuralı öğretemedi (örnekler kendi içinde çelişiyordu).
  - Her istekte birden çok yorum gönderilir (kota için); sıra her örneklemede karıştırılır.
  - Her cevaptan hem etiket hem sözel olasılık alınır. k kez örnekleme ile etiket dağılımı da çıkar.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

LABELS = ("Negative", "Neutral", "Positive")

_TASK = (
    "Sen bir e-ticaret sitesinde müşteri yorumlarını sınıflandıran bir asistansın.\n"
    "Her yorumu duygusuna göre şu üç sınıftan birine ata: Negative (olumsuz), Neutral (nötr), "
    "Positive (olumlu).\n"
    "Ayrıca her yorum için üç sınıfın olasılığını 0 ile 1 arasında ver; toplamları 1 olmalı.\n"
    "Yalnızca JSON döndür, başka hiçbir şey yazma. Biçim:\n"
    '[{"no": 1, "etiket": "Positive", "olasilik": {"Negative": 0.05, "Neutral": 0.15, "Positive": 0.8}}, ...]\n'
    "Her yorum için tam olarak bir nesne olmalı ve numaralar yorumlarınkiyle aynı olmalı."
)


MODES = ("zero", "few", "kural")

# T-01'de veri incelenerek bulunan etiketleme kuralı. Örnek ifadeler genel kalıplar; test setinden
# tek tek seçilmiş yorumlar değil. Not: Kural T-01'de, bölme yapılmadan önce tüm veriye bakılarak
# çıkarılmıştı; raporda bu açıkça belirtiliyor.
RULE = (
    "\nBu veri setinin etiketleme kuralı (dikkatle uygula):\n"
    "- Positive: Güçlü ve net övgü ya da açık tavsiye. Ör. çok beğenme, bayılma, mükemmel/harika bulma, "
    "'kesinlikle tavsiye ederim'.\n"
    "- Neutral: Hafif, genel ya da yüzeysel övgüler de Neutral sayılır (ör. 'güzel', 'güzel ürün', 'fena değil', "
    "'idare eder', 'fiyatına göre iyi'). Hem olumlu hem olumsuz yön içeren karışık yorumlar ('güzel ama ...'), "
    "henüz kullanılmadığını söyleyen ya da sadece bilgi veren yorumlar da Neutral'dır.\n"
    "- Negative: Açık şikâyet, memnuniyetsizlik, iade, tavsiye etmeme.\n"
    "Emin olmadığın hafif övgülerde Positive yerine Neutral'ı tercih et."
)


def pick_few_shot(train: pd.DataFrame, per_class: int = 5, max_words: int = 25, seed: int = 42) -> pd.DataFrame:
    """Few-shot örnekleri: YALNIZCA train setinden, sınıf başına eşit, rastgele (elle seçilmemiş).

    Çok uzun yorumlar istemi şişirmesin diye max_words sınırı var.
    """
    short = train[train["text"].str.split().str.len() <= max_words]
    return short.groupby("label").sample(n=per_class, random_state=seed)[["text", "label"]]


def build_prompt(texts: list[str], few_shot: pd.DataFrame | None = None, rule: bool = False) -> str:
    parts = [_TASK]
    if rule:
        parts.append(RULE)
    if few_shot is not None and len(few_shot):
        parts.append("\nÖrnekler (bu etiketleri insanlar vermiştir):")
        for _, row in few_shot.sample(frac=1, random_state=0).iterrows():  # sınıflar karışık sırada
            parts.append(f"- {json.dumps(_clean(row['text']), ensure_ascii=False)} → {row['label']}")
    parts.append("\nYorumlar:")
    for i, t in enumerate(texts, start=1):
        parts.append(f"{i}. {json.dumps(_clean(t), ensure_ascii=False)}")
    return "\n".join(parts)


def _clean(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_response(raw: str, n: int) -> list[dict]:
    """LLM cevabını [{etiket, olasilik}] listesine çevirir. Bozuksa ValueError fırlatır.

    Kabul edilen sapmalar: ```json ... ``` çerçevesi, olasılıkların toplamının 1 olmaması
    (normalize edilir), eksik olasılık (etikete 1 verilir).
    """
    text = raw.strip()
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        raise ValueError("JSON dizisi bulunamadı")
    data = json.loads(m.group(0))
    if not isinstance(data, list) or len(data) != n:
        raise ValueError(f"{n} nesne bekleniyordu, {len(data) if isinstance(data, list) else '?'} geldi")
    by_no = {}
    for item in data:
        no, label = int(item.get("no")), str(item.get("etiket", "")).strip().capitalize()
        if label not in LABELS:
            raise ValueError(f"Geçersiz etiket: {label!r}")
        probs = item.get("olasilik") or {}
        p = np.array([float(probs.get(c, 0.0)) for c in LABELS])
        p = p / p.sum() if p.sum() > 0 else np.eye(3)[LABELS.index(label)]
        by_no[no] = {"etiket": label, "p": p.round(4).tolist()}
    if sorted(by_no) != list(range(1, n + 1)):
        raise ValueError("Numaralar eksik ya da tekrarlı")
    return [by_no[i] for i in range(1, n + 1)]


def aggregate(samples: list[list[dict]], alpha: float = 0.5) -> dict:
    """Bir yorumun k örneklemesini birleştirir.

    samples: her örneklemede o yoruma dönen {etiket, p}.
    Döndürür:
      tahmin   : örneklemelerde en çok çıkan etiket (eşitlikte sözel olasılığı yüksek olan)
      p_ornek  : etiket sıklığı, toplamsal düzeltmeyle ((sayı + alpha) / (k + 3·alpha)).
                 Düzeltme olmazsa hiç çıkmayan sınıfa 0 olasılık düşer ve log(0) cross-entropy'yi patlatır.
      p_sozel  : sözel olasılıkların ortalaması
    """
    flat = [s for s in samples]
    counts = np.array([sum(s["etiket"] == c for s in flat) for c in LABELS], dtype=float)
    verbal = np.mean([s["p"] for s in flat], axis=0)
    verbal = verbal / verbal.sum()
    top = np.flatnonzero(counts == counts.max())
    pred_idx = top[np.argmax(verbal[top])]
    return {
        "tahmin": LABELS[pred_idx],
        "p_ornek": (counts + alpha) / (counts.sum() + 3 * alpha),
        "p_sozel": verbal,
        "k": int(counts.sum()),
    }
