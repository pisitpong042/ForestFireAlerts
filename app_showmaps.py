# app_showmaps.py
# Thailand Fire Danger (Dash) - Map Viewer
# Loads and serves pre-generated maps from maps/ directory

import os
import json
import argparse

import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import simplekml
import tempfile

# Import utilities from fire_utils
from fire_utils import (
    CLASS_COLORS,
    CLASS_SEQ,
    BOUNDS_BY_METRIC,
    bounds_to_rows,
    load_overlay,
    overlay_gdfs,
    overlay_geojsons,
    overlay_options,
)


# --------------------------  Map loader  -----------------------------------------------
class MapStore:
    """Load and cache pre-generated map JSON files."""
    
    def __init__(self, maps_dir: str = "./maps"):
        self.maps_dir = maps_dir
        self.cached_maps = {}
        self.gdfs = {}  # Store GeoDataFrames for KML export
        
    def load_map(self, metric: str, admin_level: int) -> dict:
        """Load pre-generated map JSON file."""
        cache_key = f"{metric}_admin{admin_level}"
        
        if cache_key in self.cached_maps:
            return self.cached_maps[cache_key]
        
        filepath = os.path.join(self.maps_dir, f"{metric}_admin{admin_level}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Map file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                map_dict = json.load(f)
            self.cached_maps[cache_key] = map_dict
            return map_dict
        except Exception as e:
            raise RuntimeError(f"Error loading map {filepath}: {e}")
    
    def list_available_maps(self) -> dict:
        """Return dict of available maps: {metric: [admin_levels]}"""
        available = {}
        metrics = ["FWI", "DC", "DMC", "BUI", "FFMC", "ISI"]
        
        for metric in metrics:
            available[metric] = []
            for level in [1, 2, 3]:
                filepath = os.path.join(self.maps_dir, f"{metric}_admin{level}.json")
                if os.path.exists(filepath):
                    available[metric].append(level)
        
        return available


# --------------------------  Dash app  -----------------------------------------------
def build_app(maps_dir: str = "./maps",
              gpkg_path: str = "gadm41_THA.gpkg") -> dash.Dash:
    
    app = dash.Dash(__name__)
    app.server.static_folder = 'static'
    app.server.static_url_path = '/static'
    server = app.server  # Flask server for deployment
    
    # Initialize map store
    map_store = MapStore(maps_dir)
    
    # Load metadata (date) if available
    metadata_file = os.path.join(maps_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
    
    # Check what maps are available
    available_maps = map_store.list_available_maps()
    has_all_maps = all(len(levels) == 3 for levels in available_maps.values())
    
    if not has_all_maps:
        print("⚠️  Warning: Not all maps are available. Run generate_maps.py first.")
        print("   Available maps:", available_maps)
    
    metric_options = [
        {"label": "FWI (Fire Weather Index)", "value": "FWI"},
        {"label": "DC (Drought Code)", "value": "DC"},
        {"label": "DMC (Duff Moisture Code)", "value": "DMC"},
        {"label": "BUI (Build-Up Index)", "value": "BUI"},
        {"label": "FFMC (Fine Fuel Moisture)", "value": "FFMC"},
        {"label": "ISI (Initial Spread Index)", "value": "ISI"},
    ]
    
    # Build header with date if available
    date_str = metadata.get("date", "Unknown Date")
    headline_text = f"Thailand Fire Danger Maps — {date_str}"

    app.layout = html.Div([
        dcc.Store(id="map-data-store", data={}),
        html.Div([
            html.H3(
                headline_text,
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
    ])

    COLOR_MAP = {
        "Low": CLASS_COLORS[0],
        "Moderate": CLASS_COLORS[1],
        "High": CLASS_COLORS[2],
        "Very High": CLASS_COLORS[3],
        "Extreme": CLASS_COLORS[4],
    }

    @app.callback(
        Output("map", "figure"),
        Output("legend-box", "children"),
        Output("map-data-store", "data"),
        Input("metric", "value"),
        Input("admin", "value"),
        Input("overlay", "value"),
    )
    def update_map(metric, admin_level, overlay):
        """Load pre-generated map and apply overlay if needed."""
        try:
            # Load pre-generated map
            fig_dict = map_store.load_map(metric, int(admin_level))
            fig = go.Figure(fig_dict)
            
            # Apply overlay if selected
            if overlay != "None":
                load_overlay(overlay)
                if overlay in overlay_geojsons:
                    gdf_overlay = overlay_gdfs[overlay]
                    gj_overlay = overlay_geojsons[overlay]
                    
                    # Add overlay as a choropleth with constant color for transparency
                    overlay_fig = px.choropleth_mapbox(
                        gdf_overlay,
                        geojson=gj_overlay,
                        locations=gdf_overlay.index,
                        color_discrete_sequence=["rgba(0,0,0,0.3)"],
                        mapbox_style="open-street-map",
                        opacity=0.5,
                    )
                    fig.add_trace(overlay_fig.data[0])
            
            # Generate legend
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
            
            # Store map data for KML export (metric and admin level)
            store_data = {"metric": metric, "admin_level": admin_level}
            
            return fig, legend_children, store_data
        
        except FileNotFoundError as e:
            print(f"Error: {e}")
            empty_fig = px.choropleth_mapbox()
            error_msg = html.Div(f"Map not found: {metric}_admin{admin_level}. Run generate_maps.py first.", 
                               style={"color": "red", "padding": "20px"})
            return empty_fig, error_msg, {}

    @app.callback(
        Output("download-kml", "data"),
        Input("export-kml-btn", "n_clicks"),
        Input("map-data-store", "data"),
        prevent_initial_call=True,
    )
    def export_kml(n_clicks, store_data):
        """Export current map data as KML file."""
        if not n_clicks or not store_data:
            raise PreventUpdate
        
        metric = store_data.get("metric", "FWI")
        admin_level = store_data.get("admin_level", 1)
        
        # Note: For full KML export with feature data, you would need to:
        # 1. Keep GeoDataFrames loaded in map_store
        # 2. Use them here to access geometry and attributes
        # For now, this is a placeholder that creates empty KML
        
        kml = simplekml.Kml()
        with tempfile.NamedTemporaryFile(suffix=".kml", delete=False) as tmp:
            kml.save(tmp.name)
            tmp.seek(0)
            kml_data = tmp.read()
        
        return dcc.send_bytes(kml_data, f"fire_danger_{metric}_admin{admin_level}.kml")

    @server.route('/about')
    def about():
        return app.server.send_static_file('about.html')

    return app


# --------------------------  CLI  -----------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Thailand Fire Danger Map Viewer (Pre-generated Maps)")
    parser.add_argument("--maps-dir", default="./maps", help="Directory containing pre-generated map JSON files")
    parser.add_argument("--gpkg", default="gadm41_THA.gpkg", help="Path to GADM 4.1 GeoPackage (for overlay and export)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Check if maps directory exists and has maps
    if not os.path.exists(args.maps_dir):
        print(f"❌ Error: Maps directory '{args.maps_dir}' not found.")
        print(f"   Please run: python generate_maps.py --output-dir {args.maps_dir}")
        return
    
    map_files = [f for f in os.listdir(args.maps_dir) if f.endswith(".json")]
    if not map_files:
        print(f"❌ Error: No map files found in '{args.maps_dir}'.")
        print(f"   Please run: python generate_maps.py --output-dir {args.maps_dir}")
        return
    
    print(f"✓ Found {len(map_files)} map files in {args.maps_dir}/")
    print(f"  Starting server on http://{args.host}:{args.port}")
    
    app = build_app(maps_dir=args.maps_dir, gpkg_path=args.gpkg)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
