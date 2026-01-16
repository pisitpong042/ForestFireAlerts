cd "$(dirname "$0")"
source myenv/bin/activate
python3 download_d03.py
if [ $? -eq 0 ]; then
    echo "New files detected → generating maps"
    python3 generate_maps.py
else
    echo "No new files → nothing to do"
fi
