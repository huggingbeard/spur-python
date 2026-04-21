import numpy as np
import numpy.testing as npt

from spur.utils.matrix import get_sigma_dm
from tests.config import ATOL, RTOL


def test_get_sigma_dm_returns_symmetric_demeaned_covariance() -> None:
    distmat = np.array(
        [
            [0.0, 0.3, 0.8],
            [0.3, 0.0, 0.4],
            [0.8, 0.4, 0.0],
        ]
    )

    sigma = get_sigma_dm(distmat, c=1.2)

    npt.assert_allclose(sigma, sigma.T, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(sigma.mean(axis=0), 0.0, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(sigma.mean(axis=1), 0.0, atol=ATOL, rtol=RTOL)
