# ForestFireAlerts

layout

* app.py                 # Dash app (entrypoint)
* fwi.py                 # FWI index implementations (local lib). Should be provided by an expert, not us.
* fire\_utils.py          # helper functions
* download\_d03.py        # (optional) fetch the two latest \*\_00UTC\_d03.nc files from https://tiservice.hii.or.th/wrf-roms/netcdf
* gadm41\_THA.gpkg        # GADM 4.1 Thailand map file



**Inputs**

* NetCDFs: two files ending with 00UTC\_d03.nc (yesterday \& today).
* GADM: gadm41\_THA.gpkg with ADM\_1/2/3 layers.

  * Store locally and point --gpkg to the path

https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41\_THA.gpkg

**Check libraries compatibility**
python3 -m pip install bs4 netCDF4 geopandas plotly dash simplekml



Running as a local app:

python3 app.py --port=8080



Running with waitress/caddy:

waitress-serve app:app

See the instructions at Caddy's site to write a 4 line wrapper to serve the app by https.



Helper programs for Unix:

run\_hourly.sh - checks every hour if there is a new nc file available

generate\_maps.py - For an upcoming, faster version of the app

