from importlib import resources

from spur import load_chetty_data


def test_load_chetty_data_returns_packaged_dataset() -> None:
    asset = resources.files("spur").joinpath("assets", "chetty.csv")

    df = load_chetty_data()

    assert asset.is_file()
    assert not df.empty
    assert {"am", "lat", "lon", "state"}.issubset(df.columns)
