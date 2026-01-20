#!/usr/bin/env python3
"""
Command-line utility to calculate FWI values for Thai provinces from NetCDF files.

Usage:
    python fwi_provinces.py <nc_file>

Where <nc_file> is the path to the latest NetCDF file (e.g., 2025-12-09_00UTC_d03.nc).
The script will automatically find the predecessor file and calculate FWI for each province.
Output is CSV format: Date,Province,FWI
"""

import os
import argparse
import sys

# Import utilities from fire_utils
from fire_utils import (
    read_nc,
    cal_ffmc,
    cal_isi,
    cal_dmc,
    cal_dc,
    cal_bui,
    cal_fwi,
    sample_to_admin,
)


def find_predecessor_nc(data_dir, latest_basename):
    """Find the predecessor NC file for the given latest file."""
    import glob
    from datetime import datetime
    
    # Get all NC files
    nc_files = glob.glob(os.path.join(data_dir, "*_00UTC_d03.nc"))
    
    # Parse dates
    file_dates = []
    for f in nc_files:
        basename = os.path.basename(f)
        try:
            date_str = basename.split('_')[0]
            date = datetime.strptime(date_str, "%Y-%m-%d")
            file_dates.append((date, f))
        except ValueError:
            continue
    
    # Sort by date
    file_dates.sort(key=lambda x: x[0])
    
    # Find the latest
    latest_path = os.path.join(data_dir, latest_basename)
    latest_date = None
    for date, f in file_dates:
        if f == latest_path:
            latest_date = date
            break
    
    if latest_date is None:
        raise ValueError(f"Latest file {latest_basename} not found in {data_dir}")
    
    # Find the predecessor
    predecessor = None
    for date, f in reversed(file_dates):
        if date < latest_date:
            predecessor = f
            break
    
    if predecessor is None:
        raise ValueError(f"No predecessor found for {latest_basename}")
    
    return latest_path, predecessor


def main():
    parser = argparse.ArgumentParser(
        description="Calculate FWI values for Thai provinces from NetCDF files"
    )
    parser.add_argument(
        "nc_file",
        help="Path to the latest NetCDF file (e.g., 2025-12-09_00UTC_d03.nc)",
    )
    parser.add_argument(
        "--gpkg",
        default="gadm41_THA.gpkg",
        help="Path to GADM GeoPackage (default: gadm41_THA.gpkg)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.nc_file):
        print(f"Error: NC file '{args.nc_file}' not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.gpkg):
        print(f"Error: GPKG file '{args.gpkg}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        # Find the predecessor NC file
        data_dir = os.path.dirname(args.nc_file)
        latest_basename = os.path.basename(args.nc_file)
        latest, previous = find_predecessor_nc(data_dir, latest_basename)

        print(f"Using latest: {latest}", file=sys.stderr)
        print(f"Using previous: {previous}", file=sys.stderr)

        # Read NetCDF data and populate global grids
        read_nc(previous, latest)

        # Calculate FWI components
        cal_ffmc()
        cal_isi()
        cal_dmc()
        cal_dc()
        cal_bui()
        cal_fwi()

        # Sample to province level (admin level 1)
        gdf = sample_to_admin(args.gpkg, 1)

        # Extract date from latest file name (assuming format: YYYY-MM-DD_00UTC_d03.nc)
        date = os.path.basename(latest).split('_')[0]

        # Output CSV header
        print("Date,Province,FWI")

        # Output FWI for each province
        for _, row in gdf.iterrows():
            gid = row['GID']
            fwi_val = row['FWI']

            # Extract province number from GID (format: THA.X_1)
            # THA.1_1 -> TH-1, THA.10_1 -> TH-10, etc.
            try:
                province_num = gid.split('.')[1].split('_')[0]
                province_code = f"TH-{province_num}"
            except (IndexError, ValueError):
                # Fallback to full GID if parsing fails
                province_code = gid

            print(f"{date},{province_code},{fwi_val:.2f}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()