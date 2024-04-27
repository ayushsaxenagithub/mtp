import requests
import pandas as pd

# Base URL
base_url = "https://turbulence.oden.utexas.edu/couette2018/data/"

# List of filenames to be downloaded and converted to Excel
filenames = [
    "LM_Couette_R0093_100PI_RSTE_uu_prof.dat",
    "LM_Couette_R0093_100PI_RSTE_uu_stdev.dat",
    "LM_Couette_R0093_100PI_RSTE_vv_prof.dat",
    "LM_Couette_R0093_100PI_RSTE_vv_stdev.dat",
    "LM_Couette_R0093_100PI_RSTE_ww_prof.dat",
    "LM_Couette_R0093_100PI_RSTE_ww_stdev.dat",
    "LM_Couette_R0093_100PI_RSTE_uv_prof.dat",
    "LM_Couette_R0093_100PI_RSTE_uv_stdev.dat",
    "LM_Couette_R0093_100PI_RSTE_k_prof.dat",
    "LM_Couette_R0093_100PI_RSTE_k_stdev.dat"
]

def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def save_to_excel(data, file_name):
    lines = data.split('\n')
    columns = None
    data_list = []

    for line in lines:
        if line.startswith('%') and 'y/delta' in line:
            columns = line.lstrip('%').strip().split()
            continue
        if not line.startswith('%') and columns is not None:
            data_list.append(line.split())

    if columns:
        df = pd.DataFrame(data_list, columns=columns)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.to_excel(file_name, index=False, float_format="%.18f")

for filename in filenames:
    # Construct the complete URL
    url = base_url + filename
    # Download the data
    data = download_data(url)
    if data:
        # Create a valid Excel filename
        excel_filename = filename.replace('.dat', '.xlsx')
        # Save to Excel
        save_to_excel(data, excel_filename)
        print(f"Data from {url} has been saved to {excel_filename}")
    else:
        print(f"Failed to download data from {url}")
