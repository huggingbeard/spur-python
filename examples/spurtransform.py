from spur import spurtransform, load_chetty_data, standardize


if __name__ == "__main__":
    df = load_chetty_data()

    # drop non-contiguous states
    df = df[~df.state.isin(["AK", "HI"])][["am", "fracblack", "lat", "lon", "state"]]
    df = df.dropna(subset=["am", "fracblack", "lat", "lon"])

    df = standardize(df, ["am", "fracblack"])

    # LBM-GLS transformation (recommended default)
    transformed_df = spurtransform(df, ["am"], ["lat", "lon"], method="lbmgls")

    # Nearest-neighbor differencing
    transformed_df = spurtransform(df, ["am"], ["lat", "lon"], method="nn")

    # Isotropic (200km radius)
    transformed_df = spurtransform(
        df, ["am"], ["lat", "lon"], method="iso", radius=200000
    )

    # Within-cluster demeaning
    transformed_df = spurtransform(
        df, ["am"], ["lat", "lon"], method="cluster", cluster_col="state"
    )
