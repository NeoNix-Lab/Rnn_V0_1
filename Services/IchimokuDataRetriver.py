import sqlite3
import pandas as pd


def fetch_details(path = 'C:\\Users\\user\\OneDrive\\Desktop\\DB\\Test_Cloud_Lite.db'):
    conn = sqlite3.connect(path)
    details = pd.read_sql("SELECT * FROM IchimokuDetails", conn)
    conn.close()
    return details

def fetch_data_from_detailId(detailId, path = 'C:\\Users\\user\\OneDrive\\Desktop\\DB\\Test_Cloud_Lite.db'):
     conn = sqlite3.connect(path)
     data = pd.read_sql("SELECT * FROM Ichimoku WHERE DetailsId=?", conn, params=(detailId,))
     conn.close()
     return data
     



