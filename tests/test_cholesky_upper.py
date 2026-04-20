import numpy as np
import numpy.testing as npt

from spur.utils.matrix import cholesky_upper
from tests.config import ATOL, RTOL


def test_cholesky_upper_reconstructs_spd_matrix() -> None:
    mat = np.array(
        [
            [4.0, 1.0, 2.0],
            [1.0, 3.0, 0.5],
            [2.0, 0.5, 5.0],
        ]
    )

    upper = cholesky_upper(mat)

    assert np.allclose(upper, np.triu(upper))
    npt.assert_allclose(upper.T @ upper, mat, atol=ATOL, rtol=RTOL)
