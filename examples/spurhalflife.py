from spur import spurhalflife, load_chetty_data, standardize


if __name__ == "__main__":
    df = load_chetty_data()

    # preprocess
    df = df[~df.state.isin(["AK", "HI"])][["am", "lat", "lon"]]
    df = df.dropna(subset=["am", "lat", "lon"])
    df = standardize(df, ["am"])

    # halflife estimate/test
    hl = spurhalflife("am", df, lon="lon", lat="lat", nrep=1000)
    print(hl.summary())
