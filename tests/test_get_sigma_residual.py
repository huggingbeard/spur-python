import numpy as np
import numpy.testing as npt

from spur.utils.matrix import get_sigma_residual
from tests.config import ATOL, RTOL


def test_get_sigma_residual_matches_projection_formula() -> None:
    distmat = np.array(
        [
            [0.0, 0.2, 0.8],
            [0.2, 0.0, 0.6],
            [0.8, 0.6, 0.0],
        ]
    )
    M = np.array(
        [
            [1.0, -0.5, -0.5],
            [0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    sigma = get_sigma_residual(distmat, c=1.1, M=M)
    expected = M @ np.exp(-1.1 * distmat) @ M.T

    npt.assert_allclose(sigma, expected, atol=ATOL, rtol=RTOL)
