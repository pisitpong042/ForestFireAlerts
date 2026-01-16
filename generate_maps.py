#!/usr/bin/env python3
"""
Command-line utility to pre-generate all fire danger maps.

Usage:
    python generate_maps.py --data-dir . --gpkg gadm41_THA.gpkg --output-dir ./maps
"""

import os
import argparse
import urllib.request

from fire_utils import (
    read_nc,
    sample_to_admin,
    find_two_latest_nc,
    generate_all_maps,
)

# ---------------------------------------------------------------------
# Geometry simplification tolerances (degrees)
# Conservative values suitable for Thailand
# ---------------------------------------------------------------------
SIMPLIFY_TOLERANCE = {
    1: 0.01,    # Province
    2: 0.005,   # District
    3: 0.001,   # Sub-district
}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate all fire danger maps as Plotly JSON files"
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing *_d03.nc files (default: .)",
    )
    parser.add_argument(
        "--gpkg",
        default="gadm41_THA.gpkg",
        help="Path to GADM 4.1 GeoPackage (default: gadm41_THA.gpkg)",
    )
    parser.add_argument(
        "--output-dir",
        default="./maps",
        help="Directory to save generated map JSON files (default: ./maps)",
    )
    parser.add_argument(
        "--noon-index",
        type=int,
        default=6,
        help="Time index for noon extraction (default: 6 = 12:00 UTC)",
    )
    args = parser.parse_args()

    # Check if gadm41_THA.gpkg exists, and download it if not
    if not os.path.exists(args.gpkg):
        print(f"{args.gpkg} not found. Downloading...")
        url = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_THA.gpkg"
        try:
            urllib.request.urlretrieve(url, args.gpkg)
            print(f"Downloaded {args.gpkg} successfully.\n")
        except Exception as e:
            print(f"Failed to download {args.gpkg}: {e}")
            return

    # Find and read the two latest NetCDF files
    print("Finding NetCDF files...")
    y_nc, t_nc = find_two_latest_nc(args.data_dir)
    today_filename = os.path.basename(t_nc)

    # Extract date from filename
    date_str = today_filename.split("_")[0] if "_" in today_filename else today_filename

    print(f"Using NetCDF files:")
    print(f"  Yesterday: {os.path.basename(y_nc)}")
    print(f"  Today:     {today_filename}")
    print(f"  Date:      {date_str}\n")

    # Read NetCDF data
    print("Reading NetCDF data...")
    read_nc(y_nc, t_nc, noon_index=args.noon_index)

    # Calculate FWI indices in dependency order
    print("Calculating FWI indices (FFMC, ISI, DMC, DC, BUI, FWI)...")
    from fire_utils import cal_ffmc, cal_isi, cal_dmc, cal_dc, cal_bui, cal_fwi

    # Stage 1
    cal_ffmc()
    cal_dmc()
    cal_dc()

    # Stage 2
    cal_isi()
    cal_bui()

    # Stage 3
    cal_fwi()
    print()

    # -----------------------------------------------------------------
    # Sample data to administrative levels + geometry simplification
    # -----------------------------------------------------------------
    print("Sampling data to administrative levels...")
    gdfs = {}

    for lvl in (1, 2, 3):
        gdf = sample_to_admin(args.gpkg, lvl)

        tol = SIMPLIFY_TOLERANCE.get(lvl)
        if tol is not None:
            print(f"  Simplifying admin{lvl} geometry (tolerance={tol})...")
            gdf["geometry"] = gdf["geometry"].simplify(
                tolerance=tol,
                preserve_topology=True,
            )

        gdfs[lvl] = gdf

    # Debug: Check sampled values
    gdf_lvl1 = gdfs[1]
    print("DEBUG: Sample admin1 GeoDataFrame values (first 3 rows):")
    print(gdf_lvl1[['NAME', 'FFMC', 'ISI', 'DMC', 'DC', 'BUI', 'FWI']].head(3))
    print()

    # Generate all maps
    print("Generating map files...")
    generate_all_maps(gdfs, output_dir=args.output_dir)

    # Save metadata
    import json
    metadata = {
        "date": date_str,
        "filename": today_filename,
        "generated_at": __import__('datetime').datetime.now().isoformat(),
    }
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Map generation complete!")
    print(f"  Maps saved to: {args.output_dir}/")
    print(f"  Date: {date_str}")


if __name__ == "__main__":
    main()
