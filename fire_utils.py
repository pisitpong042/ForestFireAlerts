"""
Fire Danger Utility Library
Provides data processing, FWI calculations, classification, and GADM utilities
for the Thailand Fire Danger Dash application.
"""

import os
import json
import time
import numpy as np
import netCDF4
import geopandas as gpd

# Optional: downloader (works if download_d03.py exposes fetch_latest_two)
try:
    from download_d03 import fetch_latest_two
    HAVE_DOWNLOADER = True
except Exception:
    HAVE_DOWNLOADER = False

# --------------------------  FWI library  ------------------------------------
try:
    from fwi import (
        bui as fwi_bui,
        dc as fwi_dc,
        dmc as fwi_dmc,
        isi as fwi_isi,
        ffmc as fwi_ffmc,
        fwi as fwi_fwi,
    )
    HAVE_FWI_LIB = True
except Exception:
    HAVE_FWI_LIB = False
    print("FWI library not found.")
# -----------------------  Classification  ------------------------------------
CLASS_LABELS = np.array(["Low", "Moderate", "High", "Very High", "Extreme"])
CLASS_COLORS = ["blue", "green", "yellow", "red", "brown"]  # keep order
CLASS_SEQ = ["Low", "Moderate", "High", "Very High", "Extreme"]

FFMC_BOUNDS = [-1e9, 87, 90, 93, 98, 1e9]
ISI_BOUNDS  = [-1e9,  7, 14, 20, 32, 1e9]
FWI_BOUNDS  = [-1e9, 17, 31, 40, 54, 1e9]
DC_BOUNDS   = [-1e9, 256.1, 334.1, 450.6, 600, 1e9]
DMC_BOUNDS  = [-1e9, 15.7, 27.9, 53.1, 83.6, 1e9]
BUI_BOUNDS  = [-1e9, 24.2, 40.7, 73.3, 133.1, 1e9]

BOUNDS_BY_METRIC = {
    "FWI":  FWI_BOUNDS,
    "DC":   DC_BOUNDS,
    "DMC":  DMC_BOUNDS,
    "BUI":  BUI_BOUNDS,
    "FFMC": FFMC_BOUNDS,
    "ISI":  ISI_BOUNDS,
}


def classify(arr, bounds):
    """Return (class_index, class_label) for a 1-D array using right-inclusive bins."""
    arr = np.asarray(arr)
    bins = bounds[1:-1]
    idx = np.full(arr.shape, -1, dtype=int)
    mask = ~np.isnan(arr)
    idx[mask] = np.digitize(arr[mask], bins, right=True)
    labels = np.full(arr.shape, "NA", dtype=object)
    labels[mask] = CLASS_LABELS[idx[mask]]
    return idx, labels


def _fmt(v: float) -> str:
    """Pretty number for legend ranges."""
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    s = f"{v:.1f}".rstrip("0").rstrip(".")
    return s


def bounds_to_rows(bounds):
    """
    Convert [-inf, a, b, c, d, +inf] to human ranges (right-inclusive):
    Low: ≤ a ; Moderate: > a – b ; High: > b – c ; Very High: > c – d ; Extreme: > d
    """
    a, b, c, d = bounds[1:-1]
    return [
        ("Low",        f"≤ {_fmt(a)}"),
        ("Moderate",   f"> {_fmt(a)} – {_fmt(b)}"),
        ("High",       f"> {_fmt(b)} – {_fmt(c)}"),
        ("Very High",  f"> {_fmt(c)} – {_fmt(d)}"),
        ("Extreme",    f"> {_fmt(d)}"),
    ]


# --------------------------  Grid globals  -----------------------------------
GB_Lat = GB_Lon = None
GB_Temp_noon = GB_Rh_noon = None
GB_u_10m_gr_noon = GB_v_10m_gr_noon = GB_Wsp_noon = None
GB_precip_24hr = None
GB_ffmc_yda = GB_dmc_yda = GB_dc_yda = None
GB_ffmc = GB_dmc = GB_dc = GB_isi = GB_bui = GB_fwi = None

_time_start = time.time()


def _time(msg):
    global _time_start
    print(f"{msg}: {round(time.time() - _time_start, 2)} s")
    _time_start = time.time()


# --------------------------  NetCDF reader  ----------------------------------
def read_nc(yesterday_nc_file, today_nc_file, noon_index=6):
    """Populate global grid fields from the two NetCDF files."""
    global GB_Lat, GB_Lon, GB_Temp_noon, GB_Rh_noon, GB_Wsp_noon
    global GB_u_10m_gr_noon, GB_v_10m_gr_noon, GB_precip_24hr
    global GB_ffmc, GB_dmc, GB_dc, GB_isi, GB_bui, GB_fwi
    global GB_ffmc_yda, GB_dmc_yda, GB_dc_yda

    # Yesterday: accumulate precip from 14:00 UTC onwards
    y_nc = netCDF4.Dataset(yesterday_nc_file)
    precip_hr_y = y_nc.variables["precip_hr"]   # (time, south_north, west_east)
    GB_precip_24hr = np.array(precip_hr_y[7, :, :])
    for t in range(8, 24):
        GB_precip_24hr += precip_hr_y[t, :, :]
    y_nc.close()

    # Today: noon met + add morning precip 07:00..noon
    t_nc = netCDF4.Dataset(today_nc_file)
    lat = t_nc.variables["lat"]
    lon = t_nc.variables["lon"]
    t_2m = t_nc.variables["T_2m"]
    rh_2m = t_nc.variables["rh_2m"]
    precip_hr_t = t_nc.variables["precip_hr"]
    u_10m_gr = t_nc.variables["u_10m_gr"]
    v_10m_gr = t_nc.variables["v_10m_gr"]

    GB_Lat = np.array(lat[:])
    GB_Lon = np.array(lon[:])
    sn, we = GB_Lat.shape

    GB_Temp_noon = np.array(t_2m[noon_index, :, :])
    GB_Rh_noon   = np.array(rh_2m[noon_index, :, :])
    GB_u_10m_gr_noon = np.array(u_10m_gr[noon_index, :, :])
    GB_v_10m_gr_noon = np.array(v_10m_gr[noon_index, :, :])

    for t in range(0, 6):  # 07:00..12:00
        GB_precip_24hr += precip_hr_t[t, :, :]
    t_nc.close()

    wsp_mps = np.sqrt(GB_u_10m_gr_noon**2 + GB_v_10m_gr_noon**2)
    GB_Wsp_noon = wsp_mps * 3.6  # km/h

    # allocate outputs
    GB_ffmc = np.empty((sn, we)); GB_dmc = np.empty((sn, we)); GB_dc = np.empty((sn, we))
    GB_isi  = np.empty((sn, we)); GB_bui = np.empty((sn, we)); GB_fwi = np.empty((sn, we))
    GB_ffmc_yda = np.zeros((sn, we)); GB_dmc_yda = np.zeros((sn, we)); GB_dc_yda = np.zeros((sn, we))


# ----------------------  Index computations  ---------------------------------
def cal_ffmc():
    global GB_ffmc
    sn, we = GB_ffmc.shape
    for r in range(sn):
        for c in range(we):
            GB_ffmc[r, c] = fwi_ffmc(
                GB_ffmc_yda[r, c],
                float(GB_Temp_noon[r, c]),
                float(GB_Rh_noon[r, c]),
                float(GB_Wsp_noon[r, c]),
                float(GB_precip_24hr[r, c]),
            )
    _time("FFMC")


def cal_isi():
    global GB_isi
    sn, we = GB_isi.shape
    for r in range(sn):
        for c in range(we):
            GB_isi[r, c] = fwi_isi(float(GB_ffmc[r, c]), float(GB_Wsp_noon[r, c]), True)
            GB_isi[r, c] = 100 * GB_isi[r, c]
    _time("ISI")


def cal_dmc():
    global GB_dmc
    sn, we = GB_dmc.shape
    for r in range(sn):
        for c in range(we):
            GB_dmc[r, c] = 100 * fwi_dmc( #mult
                GB_dmc_yda[r, c],
                float(GB_Temp_noon[r, c]),
                float(GB_Rh_noon[r, c]),
                float(GB_precip_24hr[r, c]),
                15, 1,
            )
    _time("DMC")


def cal_dc():
    global GB_dc
    sn, we = GB_dc.shape
    for r in range(sn):
        for c in range(we):
            GB_dc[r, c] = 100 * fwi_dc( #mult
                GB_dc_yda[r, c],
                float(GB_Temp_noon[r, c]),
                float(GB_Rh_noon[r, c]),
                float(GB_precip_24hr[r, c]),
                15, 1,
            )
    _time("DC")


def cal_bui():
    global GB_bui
    sn, we = GB_bui.shape
    for r in range(sn):
        for c in range(we):
            GB_bui[r, c] = fwi_bui(float(GB_dmc[r, c]), float(GB_dc[r, c]))
    _time("BUI")


def cal_fwi():
    global GB_fwi
    sn, we = GB_fwi.shape
    for r in range(sn):
        for c in range(we):
            GB_fwi[r, c] = fwi_fwi(float(GB_isi[r, c]), float(GB_bui[r, c]))
            GB_fwi[r, c] = 100 * GB_fwi[r, c]  # scale up for better visualization
    _time("FWI")


# ----------------------  GADM reading & sampling  ----------------------------
def _read_admin_layer(geo_pkg_path: str, level: int) -> gpd.GeoDataFrame:
    """Read GADM admin layer (1,2,3) and normalize to columns GID/NAME/geometry in EPSG:4326."""
    assert level in (1, 2, 3)
    candidates = {
        1: ["ADM_ADM_1", "ADM_1", "gadm41_THA_1", "tha_adm1"],
        2: ["ADM_ADM_2", "ADM_2", "gadm41_THA_2", "tha_adm2"],
        3: ["ADM_ADM_3", "ADM_3", "gadm41_THA_3", "tha_adm3"],
    }[level]

    gdf = None
    for lyr in candidates:
        try:
            gdf = gpd.read_file(geo_pkg_path, layer=lyr)
            break
        except Exception:
            pass
    if gdf is None:
        gdf = gpd.read_file(geo_pkg_path)  # last resort

    name_c = f"NAME_{level}"
    gid_c  = f"GID_{level}"
    cols = set(gdf.columns)
    if name_c not in cols:
        name_c = next((c for c in gdf.columns if c.startswith("NAME")), None)
    if gid_c not in cols:
        gid_c  = next((c for c in gdf.columns if c.startswith("GID")), None)
    if not name_c or not gid_c:
        raise ValueError("Could not find NAME_x / GID_x columns in layer")

    gdf = gdf.rename(columns={name_c: "NAME", gid_c: "GID"}).set_geometry("geometry")
    gdf = gdf.to_crs(4326)
    return gdf[["GID", "NAME", "geometry"]]


def _sample_grid_to_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Nearest-neighbour sample model fields to polygon centroids; add metrics & classes."""
    gdf = gdf.copy()
    cent = gdf.geometry.centroid
    gdf["cent_lat"] = cent.y
    gdf["cent_lon"] = cent.x

    # KD-Tree if SciPy is available, else simple O(n) fallback
    try:
        from scipy.spatial import cKDTree
        pts = np.column_stack([GB_Lat.ravel(), GB_Lon.ravel()])
        tree = cKDTree(pts)
        q = np.column_stack([gdf["cent_lat"].to_numpy(), gdf["cent_lon"].to_numpy()])
        _, idx = tree.query(q, k=1)
        rr, cc = np.unravel_index(idx, GB_Lat.shape)
    except Exception:
        rr, cc = [], []
        for la, lo in zip(gdf["cent_lat"].to_numpy(), gdf["cent_lon"].to_numpy()):
            d2 = (GB_Lat - la) ** 2 + (GB_Lon - lo) ** 2
            r, c = np.unravel_index(np.argmin(d2), GB_Lat.shape)
            rr.append(r); cc.append(c)
        rr, cc = np.asarray(rr), np.asarray(cc)

    def pick(A): return A[rr, cc]

    # raw metrics
    gdf["FFMC"] = pick(GB_ffmc)
    gdf["ISI"]  = pick(GB_isi)
    gdf["FWI"]  = pick(GB_fwi)
    gdf["DC"]   = pick(GB_dc)
    gdf["DMC"]  = pick(GB_dmc)
    gdf["BUI"]  = pick(GB_bui)

    # classes
    gdf["FFMC_CAT_IDX"], gdf["FFMC_CAT"] = classify(gdf["FFMC"].to_numpy(), FFMC_BOUNDS)
    gdf["ISI_CAT_IDX"],  gdf["ISI_CAT"]  = classify(gdf["ISI"].to_numpy(),  ISI_BOUNDS)
    gdf["FWI_CAT_IDX"],  gdf["FWI_CAT"]  = classify(gdf["FWI"].to_numpy(),  FWI_BOUNDS)
    gdf["DC_CAT_IDX"],   gdf["DC_CAT"]   = classify(gdf["DC"].to_numpy(),   DC_BOUNDS)
    gdf["DMC_CAT_IDX"],  gdf["DMC_CAT"]  = classify(gdf["DMC"].to_numpy(),  DMC_BOUNDS)
    gdf["BUI_CAT_IDX"],  gdf["BUI_CAT"]  = classify(gdf["BUI"].to_numpy(),  BUI_BOUNDS)

    return gdf.drop(columns=["cent_lat", "cent_lon"])


def sample_to_admin(geo_pkg_path: str, level: int) -> gpd.GeoDataFrame:
    """Sample FWI data to administrative level."""
    layer = _read_admin_layer(geo_pkg_path, level)
    return _sample_grid_to_gdf(layer)


# --------------------------  Overlay management  ------------------------------
overlay_gdfs = {}
overlay_geojsons = {}

overlay_options = [
    {"label": "Overlay: None", "value": "None"},
    {"label": "Overlay: Protected areas", "value": "protected"},
    {"label": "Overlay: Reserved forests", "value": "reserved"},
]

overlay_files = {
    "protected": "protected_areas.geojson",
    "reserved": "reserved_national_reserved_forest.geojson"
}


def load_overlay(key):
    """Load overlay data on demand."""
    if key in overlay_gdfs:
        return  # Already loaded
    filename = overlay_files.get(key)
    if filename and os.path.exists(filename):
        try:
            gdf = gpd.read_file(filename)
            overlay_gdfs[key] = gdf
            overlay_geojsons[key] = json.loads(gdf.to_json())
            print(f"Loaded overlay {filename} successfully.")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    else:
        print(f"Overlay file {filename} not found. Skipping.")


# --------------------------  Files utilities  ---------------------------------
def find_two_latest_nc(data_dir: str):
    """Keep original behavior: look for files that end with 'd03.nc' in data_dir."""
    files = [f for f in os.listdir(data_dir) if f.endswith("d03.nc")]
    if not files:
        raise FileNotFoundError("No *_d03.nc files found in --data-dir")
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)))
    if len(files) == 1:
        p = os.path.join(data_dir, files[-1])
        return p, p
    file2 = os.path.join(data_dir, files[-2])
    file1 = os.path.join(data_dir, files[-1])
    return file2, file1


# --------------------------  Map generation  ---------------------------------
def generate_all_maps(gdfs_by_level: dict, output_dir: str = "./maps") -> None:
    """
    Pre-generate all metric × admin_level combinations as Plotly JSON files.
    
    Args:
        gdfs_by_level: Dictionary {1: gdf, 2: gdf, 3: gdf} from sample_to_admin()
        output_dir: Directory to save map JSON files
    """
    import plotly.express as px
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ["FWI", "DC", "DMC", "BUI", "FFMC", "ISI"]
    levels = [1, 2, 3]
    level_names = {1: "Province", 2: "District", 3: "Sub-district"}
    
    COLOR_MAP = {
        "Low": CLASS_COLORS[0],
        "Moderate": CLASS_COLORS[1],
        "High": CLASS_COLORS[2],
        "Very High": CLASS_COLORS[3],
        "Extreme": CLASS_COLORS[4],
    }
    
    total = len(metrics) * len(levels)
    count = 0
    
    for metric in metrics:
        for level in levels:
            count += 1
            print(f"[{count}/{total}] Generating {metric} map for {level_names[level]}...", end=" ", flush=True)
            
            gdf = gdfs_by_level[level]
            gj = json.loads(gdf.to_crs(4326).to_json())
            cat_col = f"{metric}_CAT"
            
            fig = px.choropleth_mapbox(
                gdf,
                geojson=gj,
                locations="GID", featureidkey="properties.GID",
                color=cat_col,
                color_discrete_map=COLOR_MAP,
                category_orders={cat_col: CLASS_SEQ},
                mapbox_style="open-street-map",
                center={"lat": 15.0, "lon": 101.0}, zoom=5, opacity=0.75,
                hover_name="NAME",
                hover_data={metric: ":.2f"},
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                title=f"Thailand Fire Danger ({level_names[level]})",
                template=None,  # Remove template to avoid plotly version compatibility issues
            )
            
            # Ensure any numpy arrays or binary-encoded arrays in traces (e.g. customdata)
            # are converted to plain Python lists so the saved JSON is frontend-friendly.
            try:
                import numpy as _np
            except Exception:
                _np = None

            for tr in fig.data:
                # Normalize customdata to plain lists (list of scalars or list of lists)
                cd = getattr(tr, 'customdata', None)
                if cd is None:
                    continue
                try:
                    # If it's already a dict (binary-encoded), try to coerce via numpy if available
                    if isinstance(cd, dict) and _np is not None:
                        # Attempt to convert via array interface, fallback to leaving as-is
                        try:
                            arr = _np.array(cd)
                            tr.customdata = arr.tolist()
                        except Exception:
                            # As a last resort, leave the dict (older plotly may still decode it client-side)
                            pass
                    else:
                        # Coerce common array-like types to lists
                        try:
                            arr = _np.array(cd) if _np is not None else cd
                            tr.customdata = arr.tolist()
                        except Exception:
                            # If cd is an iterable of scalars/iterables, convert manually
                            try:
                                tr.customdata = [list(x) if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)) else [float(x)] for x in cd]
                            except Exception:
                                # fallback: leave as-is
                                pass
                except Exception:
                    # If anything goes wrong normalizing this trace's customdata, skip it
                    pass

            # Save as JSON
            output_file = os.path.join(output_dir, f"{metric}_admin{level}.json")
            with open(output_file, 'w') as f:
                f.write(fig.to_json())
            
            print("✓")
    
    print(f"\n✓ Generated {total} maps in {output_dir}/")
    print(f"  Metrics: {', '.join(metrics)}")
    print(f"  Levels: Province (1), District (2), Sub-district (3)")
