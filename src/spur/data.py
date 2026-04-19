from importlib import resources
import pandas as pd


def load_chetty_data() -> pd.DataFrame:
    """Load the packaged Chetty dataset."""
    asset = resources.files("spur").joinpath("assets", "chetty.csv")
    with asset.open("r", encoding="utf-8") as handle:
        return pd.read_csv(handle)


def standardize(
    df: pd.DataFrame,
    vars: list[str],
    appendix: str | None = None,
) -> pd.DataFrame:
    """Standardize selected columns and return a new DataFrame."""
    out = df.copy()

    for var in vars:
        if var not in out.columns:
            raise ValueError(f"Column '{var}' not found in DataFrame")

        standardized = (out[var] - out[var].mean()) / out[var].std()
        target = var if appendix is None else f"{var}{appendix}"
        out[target] = standardized

    return out
