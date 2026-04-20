import pandas as pd
import pytest

from spur import standardize


def test_standardize_overwrites_columns_in_copy() -> None:
    df = pd.DataFrame({"am": [1.0, 2.0, 3.0], "tlfpr": [4.0, 5.0, 6.0]})

    standardized = standardize(df, ["am", "tlfpr"])

    assert standardized is not df
    assert df["am"].tolist() == [1.0, 2.0, 3.0]
    assert df["tlfpr"].tolist() == [4.0, 5.0, 6.0]
    assert abs(standardized["am"].mean()) < 1e-10
    assert abs(standardized["tlfpr"].mean()) < 1e-10


def test_standardize_appends_suffix_columns() -> None:
    df = pd.DataFrame({"am": [1.0, 2.0, 3.0], "tlfpr": [4.0, 5.0, 6.0]})

    standardized = standardize(df, ["am", "tlfpr"], appendix="_std")

    assert "am_std" in standardized.columns
    assert "tlfpr_std" in standardized.columns
    assert abs(standardized["am_std"].mean()) < 1e-10
    assert abs(standardized["tlfpr_std"].mean()) < 1e-10


def test_standardize_raises_for_missing_column() -> None:
    df = pd.DataFrame({"am": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="not found"):
        standardize(df, ["missing"])
