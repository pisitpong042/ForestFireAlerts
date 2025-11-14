# app.py
# -----------------------------------------------------------------------------
# Thailand Fire Danger (Dash)
# - uses your original data-dir behavior (default ".")
# - reads two most-recent *_00UTC_d03.nc from --data-dir
# - computes FFMC, ISI, DMC, DC, BUI, FWI on the WRF grid
# - classifies with TFDRS/GWIS thresholds (fixed)
# - samples to GADM ADM_1 / ADM_2 / ADM_3 (centroid→nearest cell)
# - interactive Dash map with metric + admin level switchers
# - custom legend panel lists ALL classes + numeric thresholds (always visible)
# -----------------------------------------------------------------------------

import os, json, time, argparse
from threading import Thread
import numpy as np
import netCDF4
import geopandas as gpd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import Flask
import simplekml
import tempfile
import urllib.request

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

    def fwi_ffmc(ffmc_yda, t, rh, ws_kmh, prec):
        return max(0.0, 100 * np.exp(-0.005 * rh) + 0.1 * (t - 15) + 0.01 * ws_kmh - 0.2 * prec)

    def fwi_dmc(dmc_yda, t, rh, prec, lat, mon):
        return max(0.0, dmc_yda + 0.5 * (t - 10) - 0.3 * prec - 0.2 * (rh - 50))

    def fwi_dc(dc_yda, t, rh, prec, lat, mon):
        return max(0.0, dc_yda + 0.2 * (t - 10) - 0.1 * prec - 0.05 * (rh - 50))

    def fwi_isi(ffmc, ws_kmh, use_new=True):
        return max(0.0, 0.1 * ffmc + 0.05 * ws_kmh)

    def fwi_bui(dmc, dc):
        return max(0.0, 0.5 * dmc + 0.5 * dc)

    def fwi_fwi(isi, bui):
        return max(0.0, 0.4 * isi + 0.6 * bui)

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
    _time("ISI")

def cal_dmc():
    global GB_dmc
    sn, we = GB_dmc.shape
    for r in range(sn):
        for c in range(we):
            GB_dmc[r, c] = fwi_dmc(
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
            GB_dc[r, c] = fwi_dc(
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
    layer = _read_admin_layer(geo_pkg_path, level)
    return _sample_grid_to_gdf(layer)

# --------------------------  Dash app  ---------------------------------------
# Add an interval component for periodic updates
def build_app(gdfs_by_level: dict[int, gpd.GeoDataFrame],
              geojson_by_level: dict[int, dict],
              latest_nc_file: str,
              data_dir: str,
              gpkg_path: str,
              noon_index: int) -> dash.Dash:
    
    app = dash.Dash(__name__)
    app.server.static_folder = 'static'
    app.server.static_url_path = '/static'
    server = app.server  # Flask server for deployment

    metric_options = [
        {"label": "FWI (Fire Weather Index)", "value": "FWI"},
        {"label": "DC (Drought Code)", "value": "DC"},
        {"label": "DMC (Duff Moisture Code)", "value": "DMC"},
        {"label": "BUI (Build-Up Index)", "value": "BUI"},
        {"label": "FFMC (Fine Fuel Moisture)", "value": "FFMC"},
        {"label": "ISI (Initial Spread Index)", "value": "ISI"},
    ]

    app.layout = html.Div([
        dcc.Interval(
            id="update-interval",
            interval=5 * 60 * 1000,  # Check every 5 minutes (in milliseconds)
            n_intervals=0
        ),
        dcc.Store(id="latest-basename", data=os.path.basename(latest_nc_file)),
        html.Div([
            html.H3(
                os.path.basename(latest_nc_file),
                id="headline",
                style={"textAlign": "center", "display": "inline-block", "marginRight": "16px"},
            ),
            html.Button("Export KML", id="export-kml-btn", n_clicks=0, style={"verticalAlign": "middle", "marginLeft": "8px"}),
            dcc.Download(id="download-kml"),
            # Right-aligned About Us link within the same top container
            html.A(
                "About Us",
                href="/about",
                target="_blank",
                style={
                    "position": "absolute",
                    "right": "20px",
                    "top": "8px",
                    "textDecoration": "none",
                    "fontSize": "14px",
                },
            ),
        ], style={"textAlign": "center", "marginBottom": "12px", "position": "relative"}),

        html.Div([
            html.Span("Level: ", style={"marginRight": "8px"}),
            dcc.RadioItems(
                id="admin",
                value=1,
                options=[
                    {"label": "Province", "value": 1},
                    {"label": "District", "value": 2},
                    {"label": "Sub-district", "value": 3},
                ],
                labelStyle={"display": "inline-block", "marginRight": "12px"},
            ),
        ], style={"margin": "6px 0 12px"}),

        dcc.Dropdown(id="metric", options=metric_options, value="FWI", clearable=False),

        dcc.Loading(
            id="loading-spinner",
            type="circle",
            children=[
                html.Div([
                    dcc.Graph(id="map", style={"height": "85vh"}),
                    html.Div(
                        id="legend-box",
                        style={
                            "position": "absolute", "top": "70px", "right": "20px",
                            "backgroundColor": "rgba(255,255,255,0.95)",
                            "padding": "10px 12px",
                            "borderRadius": "8px",
                            "boxShadow": "0 2px 8px rgba(0,0,0,0.15)",
                            "fontSize": "13px",
                            "lineHeight": "1.2",
                        },
                    ),
                ], style={"position": "relative"}),
            ],
        ),
    ])

    COLOR_MAP = {
        "Low": CLASS_COLORS[0],
        "Moderate": CLASS_COLORS[1],
        "High": CLASS_COLORS[2],
        "Very High": CLASS_COLORS[3],
        "Extreme": CLASS_COLORS[4],
    }

    # Clientside callback: copy the latest-basename store value into the headline
    app.clientside_callback(
        """
        function(basename) {
            if (typeof(basename) === 'undefined' || basename === null) {
                throw window.dash_clientside.PreventUpdate;
            }
            return basename;
        }
        """,
        Output("headline", "children"),
        Input("latest-basename", "data"),
    )

    @app.callback(
        Output("map", "figure"),
        Output("legend-box", "children"),
        Input("metric", "value"),
        Input("admin", "value"),
    )
    def update_map(metric, admin_level):
        gdf = gdfs_by_level[int(admin_level)]
        gj = geojson_by_level[int(admin_level)]
        cat_col = f"{metric}_CAT"
        val_col = metric

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
            hover_data={val_col: ":.2f"},
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            title=f"Thailand Fire Danger ({ {1: 'Province', 2: 'District', 3: 'Sub-district'}[int(admin_level)] })",
        )

        rows = bounds_to_rows(BOUNDS_BY_METRIC[metric])
        legend_children = [
            html.Div([
                html.Strong(f"{metric} thresholds"),
                html.Div([
                    html.Div([
                        html.Span(style={
                            "display": "inline-block", "width": "12px", "height": "12px",
                            "backgroundColor": COLOR_MAP[label],
                            "marginRight": "8px", "border": "1px solid #999",
                        }),
                        html.Span(f"{label}: {rng}"),
                    ], style={"margin": "4px 0"})
                    for (label, rng) in rows
                ]),
            ])
        ]

        return fig, legend_children

    @app.callback(
        Output("latest-basename", "data"),
        Input("update-interval", "n_intervals"),
        State("latest-basename", "data"),
    )
    def check_for_updates(n_intervals, current_basename):
        """Periodically check for updates to the latest NetCDF file and update the store."""
        try:
            if HAVE_DOWNLOADER:
                fetch_latest_two(data_dir)
            y_nc_new, t_nc_new = find_two_latest_nc(data_dir)

            new_base = os.path.basename(t_nc_new)
            cur_base = str(current_basename or "")

            if new_base != cur_base:
                print(f"New NetCDF file detected: {t_nc_new}")
                read_nc(y_nc_new, t_nc_new, noon_index=noon_index)

                # Update global variables for admin levels
                nonlocal gdfs_by_level, geojson_by_level
                gdfs_by_level = {lvl: sample_to_admin(gpkg_path, lvl) for lvl in (1, 2, 3)}
                geojson_by_level = {lvl: json.loads(df.to_crs(4326).to_json()) for (lvl, df) in gdfs_by_level.items()}

                # Return new basename into the store; clientside callback will update the DOM
                return new_base
        except Exception as e:
            print(f"Error checking for updates: {e}")
        raise PreventUpdate

    @app.callback(
        Output("download-kml", "data"),
        Input("export-kml-btn", "n_clicks"),
        Input("admin", "value"),
        Input("metric", "value"),
        prevent_initial_call=True,
    )
    def export_kml(n_clicks, admin_level, metric):
        if not n_clicks:
            raise PreventUpdate
        gdf = gdfs_by_level[int(admin_level)]
        cat_col = f"{metric}_CAT"
        color_map = {
            "Low": 'ffb366ff',      # blue (AABBGGRR)
            "Moderate": 'ff66ff66', # green
            "High": 'ff00ffff',     # yellow
            "Very High": 'ff0000ff',# red
            "Extreme": 'ff222222',  # brown/dark
        }
        kml = simplekml.Kml()
        for idx, row in gdf.iterrows():
            geom = row.geometry
            placemark_name = f"{row.get('NAME', '')} ({row.get('GID', '')})"
            style_color = color_map.get(row.get(cat_col, 'Low'), 'ffb366ff')
            if geom.geom_type == 'Polygon':
                pol = kml.newpolygon(name=placemark_name)
                pol.outerboundaryis = [(x, y) for x, y in zip(*geom.exterior.xy)]
                pol.style.polystyle.color = style_color
                pol.style.polystyle.outline = 1
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    pol = kml.newpolygon(name=placemark_name)
                    pol.outerboundaryis = [(x, y) for x, y in zip(*poly.exterior.xy)]
                    pol.style.polystyle.color = style_color
                    pol.style.polystyle.outline = 1
            else:
                continue
            # Add all properties as extended data
            for col in gdf.columns:
                if col != 'geometry':
                    pol.extendeddata.newdata(name=col, value=str(row[col]))
        with tempfile.NamedTemporaryFile(suffix=".kml", delete=False) as tmp:
            kml.save(tmp.name)
            tmp.seek(0)
            kml_data = tmp.read()
        return dcc.send_bytes(kml_data, f"fire_danger_{metric}_admin{admin_level}.kml")

    @server.route('/about')
    def about():
        return app.server.send_static_file('about.html')

    return app

# --------------------------  Files & CLI  ------------------------------------
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

def main():
    parser = argparse.ArgumentParser(description="Thailand Fire Danger Dash App")
    parser.add_argument("--data-dir", default=".", help="Directory containing *_d03.nc files (unchanged)")
    parser.add_argument("--gpkg", default="gadm41_THA.gpkg", help="Path to GADM 4.1 GeoPackage")
    parser.add_argument("--noon-index", type=int, default=6, help="Time index for noon extraction (default 6 = 12:00 UTC)")
    parser.add_argument("--auto-download", action="store_true",
                        help="If set, fetch latest two *_00UTC_d03.nc into --data-dir (no change to structure)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Check if gadm41_THA.gpkg exists, and download it if not
    if not os.path.exists(args.gpkg):
        print(f"{args.gpkg} not found. Downloading...")
        url = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_THA.gpkg"
        try:
            urllib.request.urlretrieve(url, args.gpkg)
            print(f"Downloaded {args.gpkg} successfully.")
        except Exception as e:
            print(f"Failed to download {args.gpkg}: {e}")
            return

    if args.auto_download and HAVE_DOWNLOADER:
        os.makedirs(args.data_dir, exist_ok=True)
        fetch_latest_two(args.data_dir)

    y_nc, t_nc = find_two_latest_nc(args.data_dir)
    print("Using NetCDF:", y_nc, t_nc)

    read_nc(y_nc, t_nc, noon_index=args.noon_index)

    for fn in [cal_ffmc, cal_isi, cal_dmc, cal_dc, cal_bui, cal_fwi]:
        th = Thread(target=fn)
        th.start()
        th.join()

    gdfs = {lvl: sample_to_admin(args.gpkg, lvl) for lvl in (1, 2, 3)}
    geojson_by_level = {lvl: json.loads(df.to_crs(4326).to_json()) for (lvl, df) in gdfs.items()}

    app = build_app(gdfs, geojson_by_level, latest_nc_file=os.path.basename(t_nc),
                    data_dir=args.data_dir, gpkg_path=args.gpkg, noon_index=args.noon_index)
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
