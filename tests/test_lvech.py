import numpy as np
import numpy.testing as npt

from spur.utils.dist import lvech


def test_lvech_extracts_strict_lower_triangle_in_row_order() -> None:
    mat = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    values = lvech(mat)

    npt.assert_array_equal(values, np.array([4.0, 7.0, 8.0]))
