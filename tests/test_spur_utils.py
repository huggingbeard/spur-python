import numpy as np
import pandas as pd
import pytest

from spur.utils.dist import resolve_spur_coords
from spur.utils.formula import (
    parse_residual_formula,
    parse_single_var_formula,
    parse_transform_formula,
    rewrite_formula_with_prefix,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lon": [10.0, 11.0, 12.0, 13.0],
            "lat": [50.0, 51.0, 52.0, 53.0],
            "xcoord": [0.0, 1.0, 2.0, 3.0],
            "ycoord": [0.0, 1.0, 0.0, 1.0],
            "am": [1.0, 2.0, 3.0, 4.0],
            "x": [2.0, 1.0, 0.0, -1.0],
            "z": [4.0, 5.0, 6.0, 7.0],
        }
    )


def test_parse_single_var_formula_accepts_bare_string(sample_df: pd.DataFrame) -> None:
    parsed = parse_single_var_formula("am", sample_df, fn_name="spurtest()")

    assert parsed["var"] == "am"
    assert parsed["use_rows"].tolist() == [True, True, True, True]


def test_parse_single_var_formula_accepts_y_tilde_1(sample_df: pd.DataFrame) -> None:
    parsed = parse_single_var_formula("am ~ 1", sample_df, fn_name="spurtest()")

    assert parsed["var"] == "am"
    assert parsed["use_rows"].tolist() == [True, True, True, True]


def test_parse_single_var_formula_rejects_rhs_variables(
    sample_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        parse_single_var_formula("am ~ x", sample_df, fn_name="spurtest()")


def test_parse_single_var_formula_rejects_multiple_variables(
    sample_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        parse_single_var_formula("am + x", sample_df, fn_name="spurtest()")


def test_parse_residual_formula_builds_y_and_x(sample_df: pd.DataFrame) -> None:
    parsed = parse_residual_formula(
        "am ~ x + z", sample_df, fn_name="spurtest_i1resid()"
    )

    assert parsed["Y"].shape == (4, 1)
    assert parsed["X_in"].shape[0] == 4
    assert parsed["X_in"].shape[1] == 3
    assert parsed["use_rows"].tolist() == [True, True, True, True]


def test_parse_residual_formula_handles_intercept_only(sample_df: pd.DataFrame) -> None:
    parsed = parse_residual_formula("am ~ 1", sample_df, fn_name="spurtest_i1resid()")

    assert parsed["Y"].shape == (4, 1)
    assert parsed["X_in"].shape == (4, 1)
    assert np.allclose(parsed["X_in"], 1.0)


def test_parse_residual_formula_requires_two_sided_formula(
    sample_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        parse_residual_formula("am", sample_df, fn_name="spurtest_i1resid()")


def test_parse_residual_formula_drops_incomplete_rows() -> None:
    df = pd.DataFrame(
        {
            "am": [1.0, 2.0, 3.0],
            "x": [10.0, np.nan, 30.0],
            "z": [7.0, 8.0, 9.0],
        }
    )

    parsed = parse_residual_formula("am ~ x + z", df, fn_name="spurtest_i1resid()")

    assert parsed["use_rows"].tolist() == [True, False, True]
    assert parsed["Y"].shape == (2, 1)
    assert parsed["Y"].ravel().tolist() == [1.0, 3.0]
    assert parsed["X_in"].shape == (2, 3)
    assert np.allclose(parsed["X_in"][:, 0], 1.0)
    assert parsed["X_in"][:, 1].tolist() == [10.0, 30.0]
    assert parsed["X_in"][:, 2].tolist() == [7.0, 9.0]


def test_parse_transform_formula_collects_all_variables(
    sample_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    vars_ = parse_transform_formula("am ~ x + z", sample_df)
    out = capsys.readouterr().out

    assert vars_ == ["am", "x", "z"]
    assert "[formula-parser] y = am, covars (selection) = ['x', 'z']" in out


def test_parse_transform_formula_handles_single_variable_case(
    sample_df: pd.DataFrame,
) -> None:
    vars_ = parse_transform_formula("am ~ 1", sample_df)

    assert vars_ == ["am"]


def test_parse_transform_formula_deduplicates_rhs_variables(
    sample_df: pd.DataFrame,
) -> None:
    vars_ = parse_transform_formula("am ~ x + x", sample_df)

    assert vars_ == ["am", "x"]


def test_rewrite_formula_with_prefix_rewrites_both_sides() -> None:
    rewritten = rewrite_formula_with_prefix("am ~ x + z", "h_")

    assert rewritten == "h_am ~ h_x + h_z"


def test_resolve_spur_coords_returns_internal_lat_lon_order(
    sample_df: pd.DataFrame,
) -> None:
    coord_info = resolve_spur_coords(
        data=sample_df,
        use_rows=np.array([True, True, True, True]),
        lon="lon",
        lat="lat",
        coords_euclidean=None,
    )

    assert coord_info["latlong"] is True
    assert coord_info["coords"].shape == (4, 2)
    assert np.allclose(coord_info["coords"][:, 0], sample_df["lat"].to_numpy())
    assert np.allclose(coord_info["coords"][:, 1], sample_df["lon"].to_numpy())


def test_resolve_spur_coords_accepts_euclidean_columns(sample_df: pd.DataFrame) -> None:
    coord_info = resolve_spur_coords(
        data=sample_df,
        use_rows=np.array([True, True, True, True]),
        lon=None,
        lat=None,
        coords_euclidean=["xcoord", "ycoord"],
    )

    assert coord_info["latlong"] is False
    assert coord_info["coords"].shape == (4, 2)
    assert np.allclose(coord_info["coords"][:, 0], sample_df["xcoord"].to_numpy())
    assert np.allclose(coord_info["coords"][:, 1], sample_df["ycoord"].to_numpy())


@pytest.mark.parametrize(
    "formula",
    ["", "am ~ x", "am + x", "missing"],
)
def test_parse_single_var_formula_rejects_malformed_input(
    sample_df: pd.DataFrame, formula: str
) -> None:
    with pytest.raises(Exception):
        parse_single_var_formula(formula, sample_df, fn_name="spurtest()")


@pytest.mark.parametrize(
    "formula",
    ["am", "~ x", "am ~ x +", "am ~ missing"],
)
def test_parse_residual_formula_rejects_malformed_input(
    sample_df: pd.DataFrame, formula: str
) -> None:
    with pytest.raises(Exception):
        parse_residual_formula(formula, sample_df, fn_name="spurtest_i1resid()")


def test_parse_residual_formula_rejects_all_incomplete_rows() -> None:
    df = pd.DataFrame(
        {
            "am": [1.0, 2.0],
            "x": [np.nan, np.nan],
        }
    )

    with pytest.raises(Exception):
        parse_residual_formula("am ~ x", df, fn_name="spurtest_i1resid()")


@pytest.mark.parametrize(
    "formula",
    ["am", "am ~ x +", "1 ~ 1", "am ~ missing"],
)
def test_parse_transform_formula_rejects_malformed_input(
    sample_df: pd.DataFrame, formula: str
) -> None:
    with pytest.raises(Exception):
        parse_transform_formula(formula, sample_df)
