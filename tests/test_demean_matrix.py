import numpy as np
import numpy.testing as npt

from spur.utils.matrix import demean_matrix
from tests.config import ATOL, RTOL


def test_demean_matrix_zeroes_row_and_column_means() -> None:
    mat = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 6.0, 8.0],
            [3.0, 7.0, 9.0],
        ]
    )

    demeaned = demean_matrix(mat)

    npt.assert_allclose(demeaned.mean(axis=1), 0.0, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(demeaned.mean(axis=0), 0.0, atol=ATOL, rtol=RTOL)
