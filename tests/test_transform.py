import textwrap
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from spur.utils.matrix import transform
from tests.config import ATOL, PARITY_ATOL, PARITY_RTOL, RTOL
from tests.utils import (
    STATA,
    ensure_spur_stata_installed,
    execute_stata_command,
    stata_path,
)


@pytest.fixture
def grid_coords() -> np.ndarray:
    rng = np.random.default_rng(42)
    lat = rng.uniform(45, 55, 10)
    lon = rng.uniform(5, 15, 10)
    return np.column_stack([lat, lon])


def run_stata_transform(
    tmp_path: Path,
    coords: np.ndarray,
    values: np.ndarray,
    *,
    method: str,
    radius: float | None = None,
) -> np.ndarray:
    stata_root = ensure_spur_stata_installed()
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    input_csv = tmp_path / "transform_input.csv"
    output_csv = tmp_path / "transform_output.csv"

    df = pd.DataFrame({"lat": coords[:, 0], "lon": coords[:, 1], "y": values})
    df.to_csv(input_csv, index=False)

    radius_opt = f" radius({radius})" if radius is not None else ""
    script = textwrap.dedent(
        f"""
        clear all
        set more off

        sysdir set PLUS "{stata_path(plus)}"
        sysdir set PERSONAL "{stata_path(personal)}"

        import delimited using "{stata_path(input_csv)}", clear asdouble
        rename lat s_1
        rename lon s_2

        spurtransform y, prefix(d_) transformation({method}){radius_opt} latlong

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    execute_stata_command(script, tmp_path)

    return pd.read_csv(output_csv)["d_y"].to_numpy()


@pytest.mark.parametrize(
    "method,radius", [("nn", None), ("iso", 1_000_000.0), ("lbmgls", None)]
)
def test_transform_maps_constant_to_zero(
    grid_coords: np.ndarray,
    method: str,
    radius: float | None,
) -> None:
    constant = np.ones(len(grid_coords)) * 3.14

    result = transform(
        constant,
        grid_coords,
        method=method,
        radius=radius,
        latlon=True,
    )

    npt.assert_allclose(result, 0.0, atol=ATOL, rtol=RTOL)


def test_transform_maps_cluster_constant_to_zero() -> None:
    coords = np.zeros((6, 2))
    cluster = np.array([1, 1, 2, 2, 3, 3])
    constant = np.ones(6) * 7.0

    result = transform(constant, coords, method="cluster", cluster=cluster)

    npt.assert_allclose(result, 0.0, atol=ATOL, rtol=RTOL)


def test_transform_is_linear_for_nn(grid_coords: np.ndarray) -> None:
    rng = np.random.default_rng(42)
    y = rng.standard_normal(len(grid_coords))
    z = rng.standard_normal(len(grid_coords))
    a, b = 2.5, -1.3

    lhs = transform(a * y + b * z, grid_coords, method="nn", latlon=True)
    rhs = a * transform(y, grid_coords, method="nn", latlon=True)
    rhs += b * transform(z, grid_coords, method="nn", latlon=True)

    npt.assert_allclose(lhs, rhs, atol=1e-10, rtol=0.0)


def test_transform_requires_radius_for_iso(grid_coords: np.ndarray) -> None:
    with pytest.raises(ValueError, match="radius"):
        transform(np.ones(len(grid_coords)), grid_coords, method="iso")


def test_transform_rejects_unknown_method(grid_coords: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        transform(np.ones(len(grid_coords)), grid_coords, method="bogus")


def test_transform_requires_cluster_argument(grid_coords: np.ndarray) -> None:
    with pytest.raises(ValueError, match="cluster"):
        transform(np.ones(len(grid_coords)), grid_coords, method="cluster")


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_transform_matches_stata_for_lbmgls(
    tmp_path: Path,
    grid_coords: np.ndarray,
) -> None:
    rng = np.random.default_rng(42)
    values = rng.standard_normal(len(grid_coords))

    py_value = transform(values, grid_coords, method="lbmgls", latlon=True)
    st_value = run_stata_transform(
        tmp_path,
        grid_coords,
        values,
        method="lbmgls",
    )

    npt.assert_allclose(py_value, st_value, atol=PARITY_ATOL, rtol=PARITY_RTOL)
