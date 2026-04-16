from importlib import resources

import pandas as pd


def load_chetty_data() -> pd.DataFrame:
    """Load the packaged Chetty dataset."""
    asset = resources.files("spur").joinpath("assets", "chetty.csv")
    with asset.open("r", encoding="utf-8") as handle:
        return pd.read_csv(handle)
