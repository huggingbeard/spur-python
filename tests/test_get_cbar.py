import numpy as np
import pytest

from spur.utils.dist import get_cbar, lvech


def test_get_cbar_matches_target_average_correlation() -> None:
    distmat = np.array(
        [
            [0.0, 0.2, 0.8],
            [0.2, 0.0, 0.6],
            [0.8, 0.6, 0.0],
        ]
    )
    rhobar = 0.5

    c = get_cbar(rhobar, distmat)
    avg_corr = np.mean(np.exp(-c * lvech(distmat)))

    assert c > 0
    assert avg_corr == pytest.approx(rhobar, abs=1e-3)
