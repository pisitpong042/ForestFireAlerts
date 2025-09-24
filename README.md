# ForestFireAlerts

layout
- app.py                 # Dash app (entrypoint)
- fwi.py                 # FWI index implementations (local lib)
- download_d03.py        # (optional) fetch the two latest *_00UTC_d03.nc
- gadm41_THA.gpkg        # GADM 4.1 Thailand (not committed; see below)


**Inputs**

- NetCDFs: two files ending with 00UTC_d03.nc (yesterday & today).

- GADM: gadm41_THA.gpkg with ADM_1/2/3 layers. 
  - Store locally and point --gpkg to the path

**Check libraries compatability**

