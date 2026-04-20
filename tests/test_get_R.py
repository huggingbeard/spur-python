import numpy as np
import numpy.testing as npt

from spur import get_R
from tests.config import ATOL, RTOL


def test_get_R_returns_top_eigenvectors_in_descending_order() -> None:
    sigma = np.diag([1.0, 4.0, 9.0])

    R = get_R(sigma, qmax=2)

    expected = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    npt.assert_allclose(np.abs(R), expected, atol=ATOL, rtol=RTOL)
