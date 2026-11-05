"""T-03: Değerlendirme metrikleri.

Karar (docs/03_baseline.md):
  Ana metrik        : macro-F1 (model seçimi)
  İş metriği        : Negative sınıfının precision ve recall değerleri
  Kalibrasyon       : ECE (Expected Calibration Error)
  Zor örnekler seti : yumuşak etiketle cross-entropy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE: Model "%80 eminim" dediğinde gerçekten %80 oranında mı haklı?

    Tahminler en yüksek olasılığa (güven) göre kutulara ayrılır. Her kutuda
    |ortalama güven - doğruluk| farkı, kutunun büyüklüğüyle ağırlıklandırılıp toplanır.
    0 = mükemmel kalibrasyon. y_true, proba kolonlarıyla aynı sırada sınıf indeksleri olmalı.
    """
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(conf[in_bin].mean() - correct[in_bin].mean())
    return float(ece)


def soft_cross_entropy(human: np.ndarray, model: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Her örnek için H(insan, model) = -Σ p_insan · log(p_model).

    Model olasılıkları insan dağılımına ne kadar yakınsa o kadar küçük.
    Alt sınır insan dağılımının kendi entropisi (model insanlarla birebir aynı derse).
    """
    return -(human * np.log(np.clip(model, eps, 1.0))).sum(axis=1)


def entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -(p * np.log(np.clip(p, eps, 1.0))).sum(axis=1)


def classification_summary(y_true, y_pred, labels: list[str]) -> dict:
    """Ana metrik, accuracy ve sınıf bazında precision/recall/F1."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "destek": s}, index=labels)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sinif_bazinda": per_class,
    }
