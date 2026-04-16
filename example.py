"""
SPUR Python Example

Demonstrates spatial differencing transformations for removing spatial unit roots.
Uses synthetic data to show the workflow.

For validation against the original Stata/MATLAB implementations, see the test
script that loads data from the Muller-Watson replication package.
"""

import numpy as np
import pandas as pd
from spur import spurtransform, get_transformation_stats, get_distance_matrix

# Set seed for reproducibility
np.random.seed(42)


def create_synthetic_data(n: int = 100) -> pd.DataFrame:
    """
    Create synthetic georeferenced panel data.

    Generates n observations with:
    - Random coordinates in a Europe-like region
    - Outcome variable with spatial correlation
    - Treatment variable
    """
    # Coordinates: latitude and longitude (roughly Central Europe)
    lat = np.random.uniform(45, 55, n)
    lon = np.random.uniform(5, 20, n)

    # Generate spatially correlated outcome
    # y = f(lat, lon) + noise
    spatial_trend = 0.5 * (lat - 50) + 0.3 * (lon - 12.5)
    y = spatial_trend + np.random.randn(n) * 0.5

    # Treatment variable (also spatially correlated)
    treatment = 0.2 * (lat - 50) + np.random.randn(n) * 0.3

    # Create DataFrame
    df = pd.DataFrame({
        'id': np.arange(1, n + 1),
        'latitude': lat,
        'longitude': lon,
        'outcome': y,
        'treatment': treatment
    })

    return df


def main():
    print("=" * 60)
    print("SPUR Python - Spatial Unit Root Transformations")
    print("=" * 60)

    # Create synthetic data
    print("\n1. Creating synthetic data...")
    df = create_synthetic_data(n=100)
    print(f"   Generated {len(df)} observations")
    print(f"   Latitude range: {df['latitude'].min():.2f} - {df['latitude'].max():.2f}")
    print(f"   Longitude range: {df['longitude'].min():.2f} - {df['longitude'].max():.2f}")

    # Get distance statistics
    print("\n2. Distance matrix statistics...")
    coords = df[['latitude', 'longitude']].values
    stats = get_transformation_stats(coords, method='nn', latlon=True)
    print(f"   Number of observations: {stats['n_obs']}")
    print(f"   Min pairwise distance: {stats['dist_min']/1000:.1f} km")
    print(f"   Max pairwise distance: {stats['dist_max']/1000:.1f} km")
    print(f"   Mean pairwise distance: {stats['dist_mean']/1000:.1f} km")
    print(f"   Mean nearest-neighbor distance: {stats['nn_dist_mean']/1000:.1f} km")

    # Apply nearest-neighbor transformation
    print("\n3. Applying nearest-neighbor transformation...")
    df_nn = spurtransform(
        df,
        varlist=['outcome', 'treatment'],
        coord_cols=['latitude', 'longitude'],
        method='nn',
        latlon=True,
        prefix='nn_'
    )

    print("   Original vs transformed variable statistics:")
    print(f"   outcome:    mean={df['outcome'].mean():.4f}, std={df['outcome'].std():.4f}")
    print(f"   nn_outcome: mean={df_nn['nn_outcome'].mean():.4f}, std={df_nn['nn_outcome'].std():.4f}")
    print(f"   treatment:    mean={df['treatment'].mean():.4f}, std={df['treatment'].std():.4f}")
    print(f"   nn_treatment: mean={df_nn['nn_treatment'].mean():.4f}, std={df_nn['nn_treatment'].std():.4f}")

    # Apply isotropic transformation with different radii
    print("\n4. Applying isotropic transformation (200 km radius)...")
    stats_iso = get_transformation_stats(coords, method='iso', radius=200000, latlon=True)
    print(f"   Average neighbors within radius: {stats_iso['neighbors_mean']:.1f}")
    print(f"   Isolated observations (no neighbors): {stats_iso['n_isolated']}")

    df_iso = spurtransform(
        df,
        varlist=['outcome', 'treatment'],
        coord_cols=['latitude', 'longitude'],
        method='iso',
        radius=200000,  # 200 km in meters
        latlon=True,
        prefix='iso_'
    )

    print("   Original vs transformed variable statistics:")
    print(f"   outcome:     mean={df['outcome'].mean():.4f}, std={df['outcome'].std():.4f}")
    print(f"   iso_outcome: mean={df_iso['iso_outcome'].mean():.4f}, std={df_iso['iso_outcome'].std():.4f}")

    # Compare NN vs ISO
    print("\n5. Comparing transformation methods...")
    # Combine both transformations
    df_both = spurtransform(
        df_nn,
        varlist=['outcome'],
        coord_cols=['latitude', 'longitude'],
        method='iso',
        radius=200000,
        prefix='iso_'
    )

    print("   Correlation between original and transformed:")
    print(f"   outcome & nn_outcome:  {np.corrcoef(df_both['outcome'], df_both['nn_outcome'])[0,1]:.4f}")
    print(f"   outcome & iso_outcome: {np.corrcoef(df_both['outcome'], df_both['iso_outcome'])[0,1]:.4f}")
    print(f"   nn_outcome & iso_outcome: {np.corrcoef(df_both['nn_outcome'], df_both['iso_outcome'])[0,1]:.4f}")

    # Demonstrate with Euclidean coordinates
    print("\n6. Using Euclidean coordinates (latlon=False)...")
    # Create simple x,y coordinates
    df_xy = df.copy()
    df_xy['x'] = df_xy['longitude'] * 111  # rough km conversion
    df_xy['y'] = df_xy['latitude'] * 111

    df_xy = spurtransform(
        df_xy,
        varlist=['outcome'],
        coord_cols=['x', 'y'],
        method='nn',
        latlon=False,
        prefix='euc_'
    )

    print(f"   Euclidean NN transform applied")
    print(f"   euc_outcome: mean={df_xy['euc_outcome'].mean():.4f}, std={df_xy['euc_outcome'].std():.4f}")

    # Save example output
    print("\n7. Saving example output...")
    output_file = 'example_output.csv'
    df_both.to_csv(output_file, index=False)
    print(f"   Saved to {output_file}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
