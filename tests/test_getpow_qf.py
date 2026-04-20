import numpy as np

from spur import getpow_qf


def test_getpow_qf_returns_probability() -> None:
    om0 = np.array([[1.0, 0.2], [0.2, 1.5]])
    om1 = np.array([[1.2, 0.1], [0.1, 0.9]])
    rng = np.random.default_rng(42)
    e = rng.standard_normal((2, 500))

    power = getpow_qf(om0, om1, e)

    assert np.isfinite(power)
    assert 0.0 <= power <= 1.0
