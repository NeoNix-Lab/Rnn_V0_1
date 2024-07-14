
from enum import Enum
import pandas as pd
import json
from Services import db_Manager as dbm
from Services.DataRetriver import DataRetriver
from Services.config import Config
from Models.baseModelsClass import BaseModelsClass

class Dati(BaseModelsClass):
    """
    Represents the data used in machine learning processes,
    including training, work, and test data. Manages the data retrieval
    and storage in the database.

    Attributes:
    -----------
    DB_SCHEMA : str
        The schema for creating the 'dati' table in the database.
    INSERT_QUERY : str
        The query for inserting data into the 'dati' table.

    Methods:
    --------
    process(data):
        Processes the data according to the logic of Dati.
    convert_db_response(obj, db_config: Config):
        Converts a database response to a Dati object.
    push_on_db():
        Inserts the current Dati instance into the database.
    set_data(train_data, work_data, test_data, decrease_data):
        Sets the data split parameters.
    serializza_colonne(df_or_colonne):
        Serializes a list of columns from a pandas DataFrame to JSON.
    riduci_df_alle_colonne(data):
        Reduces a pandas DataFrame to the specified columns.
    """

    @property
    def DB_SCHEMA(self):
        return '''CREATE TABLE IF NOT EXISTS dati (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT,
                   train_data REAL,
                   work_data REAL,
                   test_data REAL,
                   decrease_data REAL,
                   db_references TEXT,
                   colonne TEXT
               );'''

    @property
    def INSERT_QUERY(self):
        return '''INSERT INTO dati (name, train_data, work_data, test_data, decrease_data, db_references, colonne)
                  VALUES (?, ?, ?, ?, ?, ?, ?);'''

    def __init__(self, tab_namereference, df_or_colonne, db_config: Config, train_data=0.5, work_data=0, test_data=0.5, decrease_data=0, name='Not_Named', id='Not_Posted_Yet'):
        self.id = id
        self.name = name
        self.train_data = train_data
        self.work_data = work_data
        self.test_data = test_data
        self.decrease_data = decrease_data
        self.db_references = tab_namereference
        self.df_or_colonne = self.serializza_colonne(df_or_colonne)
        self.db_config = db_config

        try:
            #HACK: dato che nonb salvo le impostazioni di configurazione salvo il path del df ma utilizzo solo il nome per ecuperare i dati
            retriver = DataRetriver(self.db_config)
            df = retriver.fetch_data(self.name)
            self.data = self.riduci_df_alle_colonne(df)
        except ValueError as e:
            raise ValueError(f'Error retrieving data: {e}')

        self.train_data_ = self.data[0:self.work_data]
        if self.work_data != -1:
            self.work_data_ = self.data[self.train_data:self.work_data]
        else:
            self.work_data_ = None
        if self.test_data != -1:
            self.test_data_ = self.data[self.work_data:self.test_data]
        else:
            self.test_data_ = None

    def push_on_db(self):
        """
        Inserts the current Dati instance into the database.
        """
        try:
            data_tuple = [(self.name, self.train_data, self.work_data, self.test_data, self.decrease_data, self.db_references, self.df_or_colonne)]
            #HACK: per evitare inserimenti multipli non sovrascrivo elementi con lo stesso nome ma vengono sovrascritti i set
            dbm.push(data_tuple, self.DB_SCHEMA, self.INSERT_QUERY,'name',0,'dati')
        except ValueError as e:
            raise ValueError(e)

    @staticmethod
    def convert_db_response(obj, db_config: Config):
        """
        Converts a database response to a Dati object.

        Parameters:
        -----------
        obj : any
            The database response to be converted.
        db_config : Config
            The database configuration.

        Returns:
        --------
        Dati
            An instance of the Dati class.
        """
        try:
            db_path = db_config.data_path
            df_or = obj[7]
            result = Dati(id=obj[0], name=obj[1], df_or_colonne=df_or, train_data=int(obj[2]), work_data=int(obj[3]), test_data=int(obj[4]), 
                          decrease_data=obj[5], tab_namereference=obj[6], db_config=db_config)

            return result
        except ValueError as e:
            raise ValueError(f'Error converting database record to Dati object: {e}')

    def set_data(self, train_data, work_data, test_data, decrease_data):
        """
        Sets the data split parameters.

        Parameters:
        -----------
        train_data : float
            Proportion of data used for training.
        work_data : float
            Proportion of data used for work.
        test_data : float
            Proportion of data used for testing.
        decrease_data : float
            Proportion of data to decrease.
        """
        for param in (train_data, work_data, test_data, decrease_data):
            if not 0.0000 <= param <= 0.999:
                raise ValueError(f"Parameter {param} must be between 0.001 and 0.999")

        dataset_length = len(self.data)

        if decrease_data != 0:
            length = int(dataset_length * decrease_data)
            self.data = self.data[:length]

        if train_data + work_data + test_data > 1:
            dataset_length = len(self.data)
            train_len = int(dataset_length * train_data)
            work_len = int(dataset_length * work_data)
            self.train_data_ = self.data[:train_len]
            self.work_data_ = self.data[train_len:train_len + work_len]
            self.test_data_ = self.data[train_len + work_len:]
        else:
            dataset_length = len(self.data)
            train_len = int(dataset_length * train_data)
            work_len = int(dataset_length * work_data)
            test_len = int(dataset_length * test_data)
            self.train_data_ = self.data[:train_len]
            self.work_data_ = self.data[train_len:train_len + work_len]
            self.test_data_ = self.data[train_len + work_len:train_len + work_len + test_len]

    def serializza_colonne(self, df_or_colonne):
        """
        Serializes a list of columns from a pandas DataFrame to JSON.
        Accepts a pandas DataFrame or a list of strings (column names).

        Returns:
        --------
        - str: JSON string containing the serialized columns.
        """
        if isinstance(df_or_colonne, pd.DataFrame):
            colonne = df_or_colonne.columns.tolist()
        elif isinstance(df_or_colonne, list) and all(isinstance(item, str) for item in df_or_colonne):
            colonne = df_or_colonne
        else:
            try:
                colonne = json.loads(df_or_colonne)
            except ValueError as e:
                raise ValueError(e)
        
        colonne_json = json.dumps(colonne)
        print(colonne_json)

        return colonne_json

    def riduci_df_alle_colonne(self, data):
        """
        Reduces a pandas DataFrame to the specified columns from a JSON string.

        Returns:
        --------
        - pd.DataFrame: Reduced DataFrame containing only the specified columns.
        """
        try:
            colonne = json.loads(self.df_or_colonne)
            if not isinstance(colonne, list):
                raise ValueError("The input JSON must represent a list of column names.")
            
            colonne_presenti = [col for col in colonne if col in data.columns]
            df_ridotto = data[colonne_presenti]
            
            return df_ridotto
        except json.JSONDecodeError:
            raise ValueError("The provided string is not a valid JSON.")

    def process(self, data):
        """
        Processes the data according to the logic of Dati.
        
        Parameters:
        -----------
        data : any
            The data to be processed.
        """
        print(f"Processing data in Dati: {data}")
