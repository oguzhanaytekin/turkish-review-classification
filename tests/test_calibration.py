import numpy as np

from src.calibration import fit_temperature, nll, softmax


def test_softmax_satirlari_bire_toplanir_ve_tasmaz():
    p = softmax(np.array([[1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.isfinite(p).all()


def test_sicaklik_asiri_guveni_bulur():
    # Kalibre bir modelin logit'lerini 3 ile çarpıp "aşırı emin" bir model üretiyoruz.
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(5000, 3))
    y = np.array([rng.choice(3, p=p) for p in softmax(true_logits)])
    overconfident = true_logits * 3
    t = fit_temperature(overconfident, y)
    assert 2.6 < t < 3.4                       # gerçek sıcaklık 3
    assert nll(overconfident, y, t) < nll(overconfident, y, 1.0)


def test_sicaklik_tahmini_degistirmez():
    logits = np.array([[2.0, 1.0, 0.1], [0.2, 0.1, 3.0]])
    assert (softmax(logits / 2.5).argmax(1) == softmax(logits).argmax(1)).all()
