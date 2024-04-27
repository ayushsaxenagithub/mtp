import requests
import pandas as pd

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
            # Extract column names
            columns = line.lstrip('%').strip().split()
            continue
        if not line.startswith('%') and columns is not None:
            # Data part
            data_list.append(line.split())

    if columns is not None:
        # Create DataFrame with correct number of columns
        df = pd.DataFrame(data_list, columns=columns)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.to_excel(file_name, index=False, float_format="%.18f")
        print(f"Data has been saved to '{file_name}'")
    else:
        print("Failed to extract columns from header")

url = "https://turbulence.oden.utexas.edu/couette2018/data/LM_Couette_R0093_100PI_RSTE_uu_prof.dat"
data = download_data(url)

if data:
    save_to_excel(data, 'output_data.xlsx')
else:
    print("Failed to download data")
