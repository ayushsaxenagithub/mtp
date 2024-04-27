import requests
import pandas as pd

def download_data(url):
    # Send a HTTP request to the URL
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        return response.text
    else:
        return None

def save_to_excel(data, file_name):
    # Split the data into lines
    lines = data.split('\n')
    
    # Find where the data starts (ignoring the header)
    for i, line in enumerate(lines):
        if line.strip().startswith('%') and 'y/delta' in line:
            start = i + 2  # Data starts after the header line
            break
    
    # Extract the data into a list of lists (each sublist is a row)
    data_list = [line.split() for line in lines[start:] if line.strip()]
    
    # Create a DataFrame
    df = pd.DataFrame(data_list, columns=["y/delta", "y^+", "Production", "Turbulent Transport", "Viscous Transport",
                                          "Pressure Strain", "Pressure Transport", "Viscous Dissipation", "Balance",
                                          "Column10", "Column11", "Column12", "Column13", "Column14", "Column15",
                                          "Column16", "Column17", "Column18", "Column19", "Column20", "Column21"])
    
    # Convert columns to numeric types (since they are initially read as strings)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Save the DataFrame to an Excel file
    df.to_excel(file_name, index=False)

# URL of the data file
url = "https://turbulence.oden.utexas.edu/couette2018/data/LM_Couette_R0093_100PI_RSTE_uu_prof.dat"

# Download the data
data = download_data(url)

if data:
    # Save to Excel
    save_to_excel(data, 'output_data.xlsx')
    print("Data has been saved to 'output_data.xlsx'")
else:
    print("Failed to download data")
