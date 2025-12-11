# ForestFireAlerts

layout

* app.py                 # Dash app (entrypoint)
* fwi.py                 # FWI index implementations (local lib)
* download\_d03.py        # (optional) fetch the two latest \*\_00UTC\_d03.nc
* gadm41\_THA.gpkg        # GADM 4.1 Thailand (not committed; see below)



**Inputs**

* NetCDFs: two files ending with 00UTC\_d03.nc (yesterday \& today).
* GADM: gadm41\_THA.gpkg with ADM\_1/2/3 layers.

  * Store locally and point --gpkg to the path

https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41\_THA.gpkg

**Check libraries compatibility**
python3 -m pip install bs4 netCDF4 geopandas plotly dash simplekml



Running with waitress/caddy:

waitress-serve app\_test:app

See the instructions at Caddy's site to write a 4 line wrapper to serve the app by https.

