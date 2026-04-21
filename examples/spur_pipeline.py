import spur
from spur import load_chetty_data, standardize

if __name__ == "__main__":
    df = load_chetty_data()

    df = df[~df.state.isin(["AK", "HI"])][["am", "fracblack", "lat", "lon"]]
    df = df.dropna(subset=["am", "fracblack", "lat", "lon"])
    df = df.reset_index(drop=True)
    df.index += 1
    df = standardize(df, ["am", "fracblack"])

    result = spur.spur(
        "am ~ fracblack",
        df,
        lon="lon",
        lat="lat",
        q=10,
        nrep=500,
        seed=42,
    )
    print(result.summary())
