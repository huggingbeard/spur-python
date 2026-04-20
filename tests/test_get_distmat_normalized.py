import numpy as np
import numpy.testing as npt

from spur.spurtest import get_distmat_normalized
from tests.config import ATOL, RTOL


def test_get_distmat_normalized_scales_max_distance_to_one() -> None:
    coords = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])

    distmat = get_distmat_normalized(coords, latlon=False)

    npt.assert_allclose(distmat, distmat.T, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(np.diag(distmat), 0.0, atol=ATOL, rtol=RTOL)
    assert distmat.max() == 1.0
