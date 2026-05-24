import sqlite3
import requests
import pandas as pd
from datetime import datetime

## This function ensure the uniqueness of each row in the traffic_counts table. 
## If the same row is inserted twice, it will be rejected.
def init_database_constraints(db_name = "fietstellingen.db", table_name = "traffic_counts"):
    """
    Ensures the traffic_counts table exists and creates a composite unique index 
    to guarantee idempotency and prevent duplicate records during daily appends.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    ## First, check if the table exists. If not, create it to attach the index.
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            Site_ID TEXT,
            Direction TEXT,
            Modus TEXT,
            Start_Time TEXT,
            End_Time TEXT,
            Count INTEGER
        );
    """)
    
    ## Create the composite unique constraint index
    cursor.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_prevent_duplicates 
        ON {table_name} (Site_ID, Start_Time, End_Time, Direction);
    """)
    
    conn.commit()
    conn.close()
    print('Unique index is active.')

## Function to get the latest date from the database
def get_latest_date(db_name = "fietstellingen.db", table_name = "traffic_counts"):
    """
    Queries the database to find the most recent Start_Time already ingested.
    If the database or table doesn't exist, returns a baseline start date.
    """
    baseline_date = pd.to_datetime('2019-08-01')
    
    try:
        conn = sqlite3.connect(db_name)
        
        ## Query the maximum start time from the table
        query = f"SELECT MAX(Start_Time) FROM {table_name}"
        max_date_str = pd.read_sql_query(query, conn).iloc[0, 0]
        conn.close()
        
        if max_date_str:
            ## Convert string from DB to datetime object
            latest_date = pd.to_datetime(max_date_str)
            print(f"Latest record date: {latest_date.strftime('%Y-%m-%d')}")
            return latest_date
            
    except Exception:
        ## If table or DB doesn't exist, fall back to the baseline date
        print('No existing database or table found.')
        
    return baseline_date

## Function to incrementally fetch missing monthly files from the AWV portal
def data_to_sqlite_incremental():
    """
    Incrementally fetches missing monthly files from the AWV portal
    and appends them to the existing SQLite database.
    """
    base_url = "https://opendata.apps.mow.vlaanderen.be/fietstellingen/"
    db_name = "/data/fietstellingen.db"
    table_name = "traffic_counts"
    col_names = ["Site_ID", "Direction", "Modus", "Start_Time", "End_Time", "Count"]

    ## Determine our start point based on existing data
    start_date = get_latest_date(db_name, table_name)
    
    ## Generate dates from the latest date in DB up to today
    dates = pd.date_range(start = start_date, end = datetime.today(), freq = 'MS')
    
    if len(dates) <= 1 and start_date.strftime('%Y-%m') == datetime.today().strftime('%Y-%m'):
        print("Database is already completely up to date with the latest available month!")
        return

    conn = sqlite3.connect(db_name)
    
    ## We check if the table actually has data right now
    try:
        check_empty = pd.read_sql_query(f"SELECT COUNT(*) FROM {table_name}", conn).iloc[0, 0]
        is_empty = True if check_empty == 0 else False
    except Exception:
        is_empty = True

    for date in dates:
        file_name = f"data-{date.strftime('%Y-%m')}.csv"
        file_url = base_url + file_name

        response = requests.head(file_url)
        
        if response.status_code == 200:
            print(f"Fetching updates from missing file: {file_name}...")
            try:
                df = pd.read_csv(file_url, sep=',', header=None, names=col_names, low_memory=False)

                ## Apply the grammar tags
                df['Site_ID'] = df['Site_ID'].astype(str).apply(lambda x: f"Location_Tag_{x}")

                ## If the DB was empty, the first file uses 'replace' to initialize. Otherwise, always 'append'.
                mode = 'replace' if is_empty else 'append'
                df.to_sql(table_name, conn, if_exists = mode, index = False)
                
                ## After the first file is written, the next iterations append
                is_empty = False

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"File {file_name} is not available. Skipping.")

    conn.close()
    print('Data is updated successfully!')


if __name__ == "__main__":
    data_to_sqlite_incremental()