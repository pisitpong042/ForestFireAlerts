# app.py
# -----------------------------------------------------------------------------
# Fire Danger Interactive Map for Thailand (Dash)
# - reads two most-recent *_00UTC_d03.nc from --data-dir
# - computes FFMC, ISI, DMC, DC, BUI, FWI on the WRF grid
# - classifies with TFDRS/GWIS thresholds (fixed)
# - samples to Thai sub-districts (ADM_3) by centroid (fast)
# -----------------------------------------------------------------------------

import os, json, time, argparse
from threading import Thread
import numpy as np
import netCDF4
import geopandas as gpd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output


#process.py
try:
    from fwi import bui as fwi_bui, dc as fwi_dc, dmc as fwi_dmc, isi as fwi_isi, ffmc as fwi_ffmc, fwi as fwi_fwi
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

CLASS_LABELS = np.array(["Low","Moderate","High","Very High","Extreme"])
CLASS_COLORS = ['blue','green','yellow','red','brown']
FWI_BOUNDS = [ -1e9, 17, 31, 40, 54,  1e9 ]
DC_BOUNDS  = [ -1e9, 256.1, 334.1, 450.6, 600, 1e9 ]
DMC_BOUNDS = [ -1e9, 15.7, 27.9, 53.1, 83.6, 1e9 ]
BUI_BOUNDS = [ -1e9, 24.2, 40.7, 73.3, 133.1, 1e9 ]

def classify(arr2d, bounds):
    bins = bounds[1:-1]
    idx = np.empty_like(arr2d, dtype=int)
    mask = ~np.isnan(arr2d)
    idx[~mask] = -1
    idx[mask] = np.digitize(arr2d[mask], bins, right=True)  # <= upper bound
    labels = np.full(arr2d.shape, "NA", dtype=object)
    labels[mask] = CLASS_LABELS[idx[mask]]
    return idx, labels

# Globals for the grid fields (populated by read_nc)
GB_Lat = GB_Lon = None
GB_Temp_noon = GB_Rh_noon = None
GB_u_10m_gr_noon = GB_v_10m_gr_noon = GB_Wsp_noon = None
GB_precip_24hr = None
GB_ffmc_yda = GB_dmc_yda = GB_dc_yda = None
GB_ffmc = GB_dmc = GB_dc = GB_isi = GB_bui = GB_fwi = None

_time_total = time.time()
_time_start = time.time()

def read_nc(yesterday_nc_file, today_nc_file, noon_index=6):
    global GB_Lat, GB_Lon, GB_Temp_noon, GB_Rh_noon, GB_Wsp_noon
    global GB_u_10m_gr_noon, GB_v_10m_gr_noon, GB_precip_24hr
    global GB_ffmc, GB_dmc, GB_dc, GB_isi, GB_bui, GB_fwi
    global GB_ffmc_yda, GB_dmc_yda, GB_dc_yda

    # Yesterday: accumulate precip from 14:00 UTC onward
    y_nc = netCDF4.Dataset(yesterday_nc_file)
    precip_hr_y = y_nc.variables['precip_hr']    # (time, sn, we)
    GB_precip_24hr = np.array(precip_hr_y[7, :, :])
    for t in range(8, 24):
        GB_precip_24hr += precip_hr_y[t, :, :]
    y_nc.close()

    # Today: noon met + add morning precip 07:00..noon
    t_nc = netCDF4.Dataset(today_nc_file)
    lat = t_nc.variables['lat']
    lon = t_nc.variables['lon']
    t_2m = t_nc.variables['T_2m']
    rh_2m = t_nc.variables['rh_2m']
    precip_hr_t = t_nc.variables['precip_hr']
    u_10m_gr = t_nc.variables['u_10m_gr']
    v_10m_gr = t_nc.variables['v_10m_gr']

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

def _time(msg):
    global _time_start
    print(f"{msg}: {round(time.time() - _time_start, 2)} s"); _time_start = time.time()

def cal_ffmc():
    global GB_ffmc
    sn, we = GB_ffmc.shape
    for r in range(sn):
        for c in range(we):
            GB_ffmc[r, c] = fwi_ffmc(GB_ffmc_yda[r, c], float(GB_Temp_noon[r, c]), float(GB_Rh_noon[r, c]), float(GB_Wsp_noon[r, c]), float(GB_precip_24hr[r, c]))
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
            GB_dmc[r, c] = fwi_dmc(GB_dmc_yda[r, c], float(GB_Temp_noon[r, c]), float(GB_Rh_noon[r, c]), float(GB_precip_24hr[r, c]), 15, 1)
    _time("DMC")

def cal_dc():
    global GB_dc
    sn, we = GB_dc.shape
    for r in range(sn):
        for c in range(we):
            GB_dc[r, c] = fwi_dc(GB_dc_yda[r, c], float(GB_Temp_noon[r, c]), float(GB_Rh_noon[r, c]), float(GB_precip_24hr[r, c]), 15, 1)
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

def sample_to_subdistricts(geo_pkg_path):
    """Return GeoDataFrame with per-tambon sampled values and categories."""
    # Try common layer names for ADM_3
    gdf = None
    for lyr in ["ADM_ADM_3", "ADM_3", "gadm41_THA_3", "tha_adm3"]:
        try:
            gdf = gpd.read_file(geo_pkg_path, layer=lyr)
            break
        except Exception:
            pass
    if gdf is None:
        gdf = gpd.read_file(geo_pkg_path)

    # Columns may vary; pick sensible defaults
    cols = list(gdf.columns)
    name_col = "NAME_3" if "NAME_3" in cols else next((c for c in cols if c.startswith("NAME")), None)
    gid_col  = "GID_3"  if "GID_3"  in cols else next((c for c in cols if c.startswith("GID")), None)
    if name_col != "NAME_3":
        gdf = gdf.rename(columns={name_col: "NAME_3"})
    if gid_col != "GID_3":
        gdf = gdf.rename(columns={gid_col: "GID_3"})

    gdf = gdf.to_crs(4326)
    gdf["cent_lat"] = gdf.geometry.centroid.y
    gdf["cent_lon"] = gdf.geometry.centroid.x

    # Nearest-neighbour sampling
    try:
        from scipy.spatial import cKDTree
        pts = np.column_stack([GB_Lat.ravel(), GB_Lon.ravel()])
        tree = cKDTree(pts)
        q = np.column_stack([gdf["cent_lat"].to_numpy(), gdf["cent_lon"].to_numpy()])
        dist, idx = tree.query(q, k=1)
        rr, cc = np.unravel_index(idx, GB_Lat.shape)
    except Exception:
        rr = []; cc = []
        for la, lo in zip(gdf["cent_lat"].to_numpy(), gdf["cent_lon"].to_numpy()):
            d2 = (GB_Lat - la)**2 + (GB_Lon - lo)**2
            r, c = np.unravel_index(np.argmin(d2), GB_Lat.shape)
            rr.append(r); cc.append(c)
        rr = np.array(rr); cc = np.array(cc)

    def pick(A): return A[rr, cc]

    gdf["FWI"] = pick(GB_fwi)
    gdf["DC"]  = pick(GB_dc)
    gdf["DMC"] = pick(GB_dmc)
    gdf["BUI"] = pick(GB_bui)

    # Classify to labels
    gdf["FWI_CAT_IDX"], gdf["FWI_CAT"] = classify(gdf["FWI"].to_numpy()[:, None], FWI_BOUNDS)
    gdf["DC_CAT_IDX"],  gdf["DC_CAT"]  = classify(gdf["DC"].to_numpy()[:, None],  DC_BOUNDS)
    gdf["DMC_CAT_IDX"], gdf["DMC_CAT"] = classify(gdf["DMC"].to_numpy()[:, None], DMC_BOUNDS)
    gdf["BUI_CAT_IDX"], gdf["BUI_CAT"] = classify(gdf["BUI"].to_numpy()[:, None], BUI_BOUNDS)

    for col in ["FWI_CAT_IDX","DC_CAT_IDX","DMC_CAT_IDX","BUI_CAT_IDX","FWI_CAT","DC_CAT","DMC_CAT","BUI_CAT"]:
        gdf[col] = gdf[col].apply(lambda x: x if np.isscalar(x) else np.array(x).item())

    return gdf

def build_app(gdf):
    app = dash.Dash(__name__)
    geojson = json.loads(gdf.to_crs(4326).to_json())
    metric_options = [
        {"label":"FWI (Fire Weather Index)","value":"FWI"},
        {"label":"DC (Drought Code)","value":"DC"},
        {"label":"DMC (Duff Moisture Code)","value":"DMC"},
        {"label":"BUI (Build-Up Index)","value":"BUI"},
    ]
    app.layout = html.Div([
        html.H2("Thailand Fire Danger (Sub-district)", style={"textAlign":"center"}),
        dcc.Dropdown(id="metric", options=metric_options, value="FWI", clearable=False),
        dcc.Graph(id="map", style={"height":"85vh"}),
    ])

    COLOR_MAP = {"Low":CLASS_COLORS[0], "Moderate":CLASS_COLORS[1], "High":CLASS_COLORS[2], "Very High":CLASS_COLORS[3], "Extreme":CLASS_COLORS[4]}

    @app.callback(Output("map","figure"), Input("metric","value"))
    def update_map(metric):
        cat_col = f"{metric}_CAT"; val_col = metric
        fig = px.choropleth_mapbox(
            gdf,
            geojson=geojson,
            locations="GID_3", featureidkey="properties.GID_3",
            color=cat_col,
            color_discrete_map=COLOR_MAP,
            category_orders={cat_col:["Low","Moderate","High","Very High","Extreme"]},
            mapbox_style="open-street-map",
            center={"lat":15.0, "lon":101.0}, zoom=5, opacity=0.75,
            hover_name="NAME_3",
            hover_data={val_col:":.2f" }
        )
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), legend_title_text="Fire Risk Level")
        return fig

    return app

def find_two_latest_nc(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('d03.nc')]
    if not files:
        raise FileNotFoundError("No *_d03.nc files found in --data-dir")
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(data_dir,f)))
    if len(files) == 1:
        return os.path.join(data_dir, files[-1]), os.path.join(data_dir, files[-1])
    return os.path.join(data_dir, files[-2]), os.path.join(data_dir, files[-1])

def main():
    parser = argparse.ArgumentParser(description="Thailand Fire Danger Dash App")
    parser.add_argument('--data-dir', default='.', help='Directory containing *_d03.nc files')
    parser.add_argument('--gpkg', default='gadm41_THA.gpkg', help='Path to GADM 4.1 GeoPackage')
    parser.add_argument('--noon-index', type=int, default=6, help='Time index for noon extraction (default 6 = 12:00 UTC)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    y_nc, t_nc = find_two_latest_nc(args.data_dir)
    print("Using NetCDF:", y_nc, t_nc)

    read_nc(y_nc, t_nc, noon_index=args.noon_index)

    # parallel compute
    for fn in [cal_ffmc, cal_isi, cal_dmc, cal_dc, cal_bui, cal_fwi]:
        th = Thread(target=fn)
        th.start(); th.join()

    gdf = sample_to_subdistricts(args.gpkg)
    app = build_app(gdf)
    app.run_server(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
