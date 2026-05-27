# Importing the tools I need to work with: pandas, os, and requests
import pandas as pd
import os
import requests

# Defining the variable that holds the URL where the CSV dataset lives online
DATA_URL = "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
# Defining the variable that holds the local file path where I want to save the dataset
LOCAL_FILE_PATH = "forecast.csv"
# Function called download data in case there is no local file
def download_data():
    if not os.path.exists(LOCAL_FILE_PATH):
        print("File not found locally. Downloading from URL...")
        response = requests.get(DATA_URL)
        with open(LOCAL_FILE_PATH, 'wb') as file:
            file.write(response.content)
        print("Download complete and file saved locally.")
    else:
        print("File already exists locally. No download needed.")

def load_and_clean():
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(LOCAL_FILE_PATH)
    
    # cleaning steps:
    # (1)Setting the times as dataframe datetimes
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df.set_index('utc_timestamp', inplace=True)
    # (2) Keep the columns DE_solar_generation_actual and DE_load_actual_entsoe_transparency
    df = df[['DE_solar_generation_actual', 'DE_load_actual_entsoe_transparency']]
    # (3) Removing any rows with missing values and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    return df
# call both funcitons and print the first 5 rows to confirm everything worked
if __name__ == "__main__":
    download_data()
    cleaned_df = load_and_clean()
    print(cleaned_df.head(5))

# Our Pipeline is working here, we cleaned our data by correcitng columns, removing doubles and missing values, and setting
