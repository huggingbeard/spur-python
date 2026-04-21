import numpy as np
import numpy.testing as npt

from spur.utils.matrix import get_sigma_lbm
from tests.config import ATOL, RTOL


def test_get_sigma_lbm_matches_closed_form_definition() -> None:
    distmat = np.array(
        [
            [0.0, 0.2, 0.8],
            [0.2, 0.0, 0.6],
            [0.8, 0.6, 0.0],
        ]
    )
    expected = 0.5 * (distmat[:, [0]] + distmat[[0], :] - distmat)

    sigma = get_sigma_lbm(distmat)

    npt.assert_allclose(sigma, expected, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(sigma, sigma.T, atol=ATOL, rtol=RTOL)
