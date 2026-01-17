# app.py
# Thailand Fire Danger (Dash) - Main GUI Application
# Imports FWI calculations, classification, and data handling from fire_utils

import os
import json
import argparse
from threading import Thread
import urllib.request

from version import __version__

import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import simplekml
import tempfile

# Import all utilities from fire_utils
from fire_utils import (
    HAVE_DOWNLOADER,
    CLASS_COLORS,
    CLASS_SEQ,
    BOUNDS_BY_METRIC,
    bounds_to_rows,
    read_nc,
    cal_ffmc,
    cal_isi,
    cal_dmc,
    cal_dc,
    cal_bui,
    cal_fwi,
    sample_to_admin,
    load_overlay,
    find_two_latest_nc,
    overlay_gdfs,
    overlay_geojsons,
    overlay_options,
)

# Handle optional downloader
if HAVE_DOWNLOADER:
    from download_d03 import fetch_latest_two

# --------------------------  Dash app  -----------------------------------------------
def build_app(gdfs_by_level: dict,
              geojson_by_level: dict,
              latest_nc_file: str,
              data_dir: str,
              gpkg_path: str,
              noon_index: int = 6) -> dash.Dash:
    
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
        dcc.Dropdown(id="overlay", options=overlay_options, value="None", clearable=False),

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

        html.Footer(
            html.Div(
                f"ForestFireAlerts v{__version__}",
                style={
                    "textAlign": "center",
                    "fontSize": "12px",
                    "color": "#666",
                    "paddingTop": "12px",
                    "marginTop": "12px",
                    "borderTop": "1px solid #ddd",
                },
            ),
            style={"width": "100%", "marginTop": "auto"},
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
        Input("overlay", "value"),
    )
    def update_map(metric, admin_level, overlay):
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

        # Add overlay if selected
        if overlay != "None":
            load_overlay(overlay)
            gdf_overlay = overlay_gdfs[overlay]
            gj_overlay = overlay_geojsons[overlay]
            # Add overlay as a choropleth with constant color for transparency
            overlay_fig = px.choropleth_mapbox(
                gdf_overlay,
                geojson=gj_overlay,
                locations=gdf_overlay.index,  # Assuming index as locations; adjust if needed
                color_discrete_sequence=["rgba(0,0,0,0.3)"],  # Semi-transparent black
                mapbox_style="open-street-map",
                opacity=0.5,
            )
            fig.add_trace(overlay_fig.data[0])

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


# --------------------------  Main entry point  ------------------------------------
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
