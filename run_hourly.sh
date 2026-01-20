cd "$(dirname "$0")"
source myenv/bin/activate
python3 download_d03.py
if [ $? -eq 2 ]; then
    echo "New files detected → generating maps"
    python3 generate_maps.py
    #after that, restart the server so that it refreshes the maps
    pkill -f app_showmaps.py
else
    echo "No new files → nothing to do"
fi
#are there too many files? Delete the earliest
files=(*_00UTC_d03.nc)
if (( ${#files[@]} > 5 )); then
  rm -- "${files[0]}"
fi
