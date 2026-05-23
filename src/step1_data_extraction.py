import sqlite3
import requests
import pandas as pd
from datetime import datetime

def data_to_sqlite():
    """
    Fetches monthly historical cycling data from the AWV open data portal
    and stores it in a local SQLite database.
    """
    ## Collect the BASE URL
    base_url = "https://opendata.apps.mow.vlaanderen.be/fietstellingen/"
    db_name = "/data/fietstellingen.db"

    ## Connect to (or create) the SQLite database
    conn = sqlite3.connect(db_name)

    ## Give the date ranges
    dates = pd.date_range(start='2019-08-01', end=datetime.today(), freq='MS')
    col_names = ["Site_ID", "Direction", "Modus", "Start_Time", "End_Time", "Count"]

    first_run = True

    for date in dates:
        file_name = f"data-{date.strftime('%Y-%m')}.csv"
        file_url = base_url + file_name

        ## Check if the file exists on the server before downloading
        response = requests.head(file_url)
        
        if response.status_code == 200:
            print(f"Processing {file_name}...")
            try:
                ## Read the CSV directly into pandas
                df = pd.read_csv(file_url, sep=',', header=None, names=col_names, low_memory=False)

                ## Apply the grammar tags
                df['Site_ID'] = df['Site_ID'].astype(str).apply(lambda x: f"Location_Tag_{x}")

                ## Write to SQLite
                mode = 'replace' if first_run else 'append'
                df.to_sql('traffic_counts', conn, if_exists=mode, index=False)
                
                first_run = False

            except Exception as e:
                print(f"Error on {file_name}: {e}")

    conn.close()
    print(f"Historical data successfully stored in {db_name}")


if __name__ == "__main__":
    print("Starting data extraction pipeline...")
    data_to_sqlite()