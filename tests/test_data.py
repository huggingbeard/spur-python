from importlib import resources
import pandas as pd
from spur import load_chetty_data


def test_chetty_packaged():
    asset = resources.files("spur").joinpath("assets", "chetty.csv")
    assert asset.is_file()


def test_load_chetty_data():
    df = load_chetty_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    expected_columns = {"CZ", "State", "Lat", "Lon", "AM", "TLFPR"}
    assert expected_columns.issubset(df.columns)
