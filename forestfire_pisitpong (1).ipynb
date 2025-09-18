#01_download_files.py
#Download the 2 most recent 00UTC_d03.nc files
import requests
from bs4 import BeautifulSoup
#from tqdm.notebook import tqdm # Import tqdm
import os # Import os module
import sys # Import sys module for text-based progress

url = "https://tiservice.hii.or.th/wrf-roms/netcdf"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all links on the page
links = [a['href'] for a in soup.find_all('a', href=True)]

# Filter for files ending with 'd03.nc' and sort them
target_files = sorted([link for link in links if link.endswith('00UTC_d03.nc')], reverse=False)

# Select the bottommost two files
bottommost_files = []
if len(target_files) >= 2:
    bottommost_files = target_files[-2:] # Select the last two files
    print(f"The bottommost two files ending with '00UTC_d03.nc' are: {bottommost_files}")
elif len(target_files) == 1:
    bottommost_files = target_files[-1:] # Select the last file
    print(f"Only one file ending with '00UTC_d03.nc' found: {bottommost_files}")
else:
    print("No files ending with '00UTC_d03.nc' found.")


# Download the files
for bottommost_file in bottommost_files:
    file_url = f"{url}/{bottommost_file}"
    # Check if the file already exists
    if os.path.exists(bottommost_file):
        print(f"{bottommost_file} already exists. Skipping download.")
        continue # Skip to the next file
    print(f"Downloading {file_url}...")
    with requests.get(file_url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0)) # Get the total size
        downloaded_size = 0
        with open(bottommost_file, 'wb') as f:
            #with tqdm(total=total_size, unit='B', unit_scale=True, desc=bottommost_file) as pbar: # Add tqdm progress bar
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    #pbar.update(len(chunk)) # Update the progress bar
                    progress = (downloaded_size / total_size) * 100
                    sys.stdout.write(f"\rDownloading {bottommost_file}: {progress:.2f}%")
                    sys.stdout.flush()
        sys.stdout.write('\n') # Newline after download is complete
        sys.stdout.flush()
    print(f"Downloaded {bottommost_file}")
