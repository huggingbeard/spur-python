from spur import spurtest, load_chetty_data, standardize


if __name__ == "__main__":
    df = load_chetty_data()

    # preprocess
    df = df[~df.state.isin(["AK", "HI"])][["am", "lat", "lon"]]
    df = df.dropna(subset=["am", "lat", "lon"])
    df = standardize(df, ["am"])

    # i1 test
    i1res = spurtest("am", df, test="i1", lon="lon", lat="lat")
    print(i1res.summary())

    # i0 test
    i0res = spurtest("am", df, test="i0", lon="lon", lat="lat")
    print(i0res.summary())
