"""
test_fire_utils_3days.py

Clean test / demo script to:
1) Compute FWI for all days in a multi-day WRF NetCDF file
2) Sample ALL days to Thai provinces (admin level 1)
3) Print in CSV format:

    Date,Province,FWI

Where Province is formatted as TH-xxx

Assumes:
- fire_utils_3day.py is in the same directory (or in PYTHONPATH)
- A GADM GeoPackage for Thailand exists (e.g. gadm41_THA.gpkg)

"""

import numpy as np
import geopandas as gpd
import netCDF4

from fire_utils_3days import compute_fwi_all_days


# -----------------------------------------------------------------------------
# GADM reader (admin level 1 = provinces)
# -----------------------------------------------------------------------------

def read_admin_layer(geo_pkg_path: str, level: int = 1) -> gpd.GeoDataFrame:
    """Read GADM admin layer and normalize to columns: GID, NAME, geometry."""
    candidates = {
        1: ["ADM_ADM_1", "ADM_1", "gadm41_THA_1", "tha_adm1"],
    }[level]

    gdf = None
    for lyr in candidates:
        try:
            gdf = gpd.read_file(geo_pkg_path, layer=lyr)
            break
        except Exception:
            pass

    if gdf is None:
        gdf = gpd.read_file(geo_pkg_path)

    name_c = next((c for c in gdf.columns if c.startswith("NAME")), None)
    gid_c  = next((c for c in gdf.columns if c.startswith("GID")), None)

    if not name_c or not gid_c:
        raise ValueError("Could not find NAME_x / GID_x columns in GADM file")

    gdf = gdf.rename(columns={name_c: "NAME", gid_c: "GID"})
    gdf = gdf.set_geometry("geometry").to_crs(4326)

    return gdf[["GID", "NAME", "geometry"]]


# -----------------------------------------------------------------------------
# Read dates from NetCDF DateTime variable
# -----------------------------------------------------------------------------

def read_day_dates(nc_file, n_days, noon_hour=12):
    """
    Extract one date string per day from NetCDF `DateTime` variable using the noon index.

    Supports formats:
    - integer like 2026012100 (YYYYMMDDHH)
    - string like '2026-01-21 00:00:00'

    Returns list of ISO date strings: ["YYYY-MM-DD", ...]
    """
    nc = netCDF4.Dataset(nc_file)

    if "DateTime" not in nc.variables:
        raise ValueError("NetCDF file does not contain 'DateTime' variable")

    dt = nc.variables["DateTime"][:]

    dates = []
    for day in range(n_days):
        idx = day * 24 + noon_hour
        val = dt[idx]

        # case 1: numeric YYYYMMDDHH
        if np.issubdtype(type(val), np.number) or isinstance(val, (int, np.integer)):
            s = str(int(val))
            date = f"{s[0:4]}-{s[4:6]}-{s[6:8]}"

        # case 2: byte / string
        else:
            if isinstance(val, bytes):
                s = val.decode("ascii")
            else:
                s = str(val)

            # try common patterns
            if "_" in s:
                date = s.split("_")[0]
            elif " " in s:
                date = s.split(" ")[0]
            else:
                date = s[0:10]

        dates.append(date)

    nc.close()
    return dates


def read_day_dates(nc_file, n_days, noon_hour=12):
    """
    Extract one date string per day from NetCDF `DateTime` variable using the noon index.

    Assumes DateTime values like: 2026012100 (YYYYMMDDHH)

    Returns list: ["YYYY-MM-DD", ...]
    """
    import netCDF4
    import numpy as np

    nc = netCDF4.Dataset(nc_file)

    Times = nc.variables["DateTime"][:]   # shape: (time,)

    dates = []
    for day in range(n_days):
        idx = day * 24 + noon_hour
        val = Times[idx]                  # e.g. 2026012112

        s = str(int(val))                 # '2026012112'
        date = f"{s[0:4]}-{s[4:6]}-{s[6:8]}"

        dates.append(date)

    nc.close()
    return dates


# -----------------------------------------------------------------------------
# Sampling: nearest grid point to province centroid
# -----------------------------------------------------------------------------

def sample_state_to_provinces(state, geo_pkg_path):
    """
    For each province, sample FWI at the nearest model grid point to its centroid.

    Returns GeoDataFrame with columns: GID, NAME, FWI
    """

    gdf = read_admin_layer(geo_pkg_path, level=1)

    # centroids
    cent = gdf.geometry.centroid
    gdf["cent_lat"] = cent.y
    gdf["cent_lon"] = cent.x

    lat = state["lat"]
    lon = state["lon"]
    fwi = state["fwi"]

    # KD-tree for fast nearest neighbour search
    try:
        from scipy.spatial import cKDTree

        pts = np.column_stack([lat.ravel(), lon.ravel()])
        tree = cKDTree(pts)

        q = np.column_stack([gdf["cent_lat"].to_numpy(), gdf["cent_lon"].to_numpy()])
        _, idx = tree.query(q, k=1)
        rr, cc = np.unravel_index(idx, lat.shape)

    except Exception:
        # fallback O(N) search
        rr, cc = [], []
        for la, lo in zip(gdf["cent_lat"], gdf["cent_lon"]):
            d2 = (lat - la) ** 2 + (lon - lo) ** 2
            r, c = np.unravel_index(np.argmin(d2), lat.shape)
            rr.append(r)
            cc.append(c)
        rr = np.asarray(rr)
        cc = np.asarray(cc)

    gdf["FWI"] = fwi[rr, cc]

    return gdf.drop(columns=["cent_lat", "cent_lon"])


# -----------------------------------------------------------------------------
# Province code formatter: GID -> TH-xxx
# -----------------------------------------------------------------------------

def format_province_code(gid):
    """
    Convert GADM GID (e.g. 'THA.1_1') into TH-xxx style code.

    This is an approximation: TH-<first numeric part>
    Example:
        THA.1_1 -> TH-1
    """
    try:
        base = gid.split(".")[1]
        num = base.split("_")[0]
        return f"TH-{num}"
    except Exception:
        return "TH-NA"


# -----------------------------------------------------------------------------
# Main test
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Compute FWI for all days and print province-level CSV (Date,Province,FWI)"
    )
    parser.add_argument(
        "nc_file",
        help="Path to multi-day NetCDF file containing DateTime variable",
    )
    parser.add_argument(
        "--gadm",
        default="gadm41_THA.gpkg",
        help="Path to Thailand GADM GeoPackage (default: gadm41_THA.gpkg)",
    )

    args = parser.parse_args()

    nc_file = args.nc_file
    gadm_file = args.gadm

    # --- compute all days ---
    print("Computing FWI for all days in file...\n")
    days = compute_fwi_all_days(nc_file)

    n_days = len(days)
    print(f"Computed {n_days} daily FWI fields")

    # --- read dates from DateTime ---
    dates = read_day_dates(nc_file, n_days)

    # --- CSV header ---
    print("Date,Province,FWI")

    # --- loop over all days and all provinces ---
    for day in range(n_days):
        date = dates[day]

        gdf_prov = sample_state_to_provinces(days[day], gadm_file)

        for _, row in gdf_prov.iterrows():
            code = format_province_code(row["GID"])
            fwi  = row["FWI"]

            try:
                fwi_val = float(fwi)
                print(f"{date},{code},{fwi_val:.2f}")
            except Exception:
                print(f"{date},{code},NA")

    print("\nDone.")
