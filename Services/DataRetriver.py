import sqlite3
import pandas as pd
from Services.config import Config

class DataRetriver:
    """
    Class to manage database operations including table creation from CSV
    and data insertion, using paths from the configuration.

    Attributes:
    -----------
    config : Config
        An instance of the Config class to manage configuration settings.
    PATH : str
        The path to the database file based on the environment.

    Methods:
    --------
    create_table_from_csv(table_name, csv_path)
        Creates a table in the database from a CSV file.
    insert_data(table_name, data)
        Inserts data into the specified table in the database.
    fetch_data(table_name)
        Fetches all details from the specified table.
    fetch_data_from_prop(table_name, prop_name, prop_val)
        Fetches data from the specified table based on a property value.
    """

    def __init__(self, enviroment_config:Config=Config()):
        self.config = enviroment_config
        self.PATH = self.config.data_path
        
    def create_table_from_csv(self, table_name: str, csv_path: str):
        """Creates a table in the database from a CSV file."""
        try:
            conn = sqlite3.connect(self.PATH)
            df = pd.read_csv(csv_path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            print(f"Table {table_name} created successfully from {csv_path}.")
        except Exception as e:
            print(f"Error creating table from CSV: {e}")

    def insert_data(self, table_name: str, data: pd.DataFrame):
        """Inserts data into the specified table in the database."""
        try:
            conn = sqlite3.connect(self.PATH)
            data.to_sql(table_name, conn, if_exists='append', index=False)
            conn.close()
            print(f"Data inserted successfully into table {table_name}.")
        except Exception as e:
            print(f"Error inserting data: {e}")

    def fetch_data(self, table_name: str):
        """Fetches all data from the specified table."""
        try:
            conn = sqlite3.connect(self.PATH)
            
            details = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            
            conn.close()
            return details
        except Exception as e:
            print(f'@@@@@@@@@@@@@@@@@@#########################path: {self.PATH}')
            
            print(f"Error fetching details: {e}")
            return None

    def fetch_data_from_prop(self, table_name: str, prop_name: str, prop_val):
        """Fetches data from the specified table based on a property value."""
        try:
            conn = sqlite3.connect(self.PATH)
            data = pd.read_sql(f"SELECT * FROM {table_name} WHERE {prop_name}=?", conn, params=(prop_val,))
            conn.close()
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        
    def create_A_Dedicated_Table(self, tab_name, df:pd.DataFrame):
        try:
            conn = sqlite3.connect(self.PATH)
            df.to_sql(tab_name , conn, if_exists='replace', index=False)
            conn.close()

        except Exception as e:
            print(f"Error creating table: {e}")
            return None
        
