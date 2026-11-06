"""T-04: Olasılık kalibrasyonu (temperature scaling).

Fikir: Model logit'lerini tek bir sayıya (T) bölüp softmax alırız.
  T > 1  -> olasılıklar yumuşar (aşırı emin modeli sakinleştirir)
  T < 1  -> olasılıklar keskinleşir
T, val setinde log loss'u (NLL) en aza indirecek şekilde seçilir; test setine dokunulmaz.
Tahmin edilen sınıf DEĞİŞMEZ (argmax aynı kalır), sadece güven skorları düzelir.
Bu yüzden macro-F1 aynı kalır, ECE ve log loss değişir.

Torch gerektirmez; logit'ler numpy dizisi olarak gelir, böylece yerelde test edilebilir.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)  # sayısal taşmayı önlemek için
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    """Ortalama negatif log-olabilirlik (log loss). y: sınıf indeksleri."""
    p = softmax(logits / temperature)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Val setinde NLL'yi en aza indiren sıcaklığı bulur (0,05–10 aralığında)."""
    res = minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.05, 10.0), method="bounded")
    return float(res.x)
