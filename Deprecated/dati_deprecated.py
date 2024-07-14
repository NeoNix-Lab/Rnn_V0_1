from enum import Enum
import pandas as pd
import pandera as pa
import json
from Services import Db_Manager as dbm, IchimokuDataRetriver as ichi


class Dati():
    #TODO: momentaneamente funziona con id di ichi
    #TODO: momentaneamente la compressione delle colonne e rimandata solo a streamlit

    DB_SCHEMA = '''CREATE TABLE IF NOT EXISTS dati (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT,
               train_data REAL,
               work_data REAL,
               test_data REAL,
               decrease_data REAL,
               db_references TEXT,
               colonne TEXT
           );
           '''

    INSERT_QUERY = '''INSERT INTO dati (name, train_data, work_data, test_data, decrease_data, db_references, colonne)
          VALUES (?, ?, ?, ?, ?, ?, ?);'''
          
    DATAS_DB_PATH = ''
    
# TODO: riordinare il set_data e la gestione dati e percentuali di dati
    def __init__(self, db_reference, df_or_colonne, train_data=0.5, work_data=0, test_data=0.5,  decrease_data=0, name='Not_Named', id='Not_Posted_Yet', data_db_path=''):
        self.id = id
        self.name = name
        self.train_data = train_data
        self.work_data = work_data
        self.test_data = test_data
        self.decrease_data = decrease_data
        self.db_references = db_reference
        self.df_or_colonne = self.serializza_colonne(df_or_colonne)
        self.DATAS_DB_PATH = data_db_path
        try:
            if self.DATAS_DB_PATH == '':
                df = ichi.fetch_data_from_detailId(self.db_references)
            else:
                df = ichi.fetch_data_from_detailId(self.db_references, self.DATAS_DB_PATH)
                
            self.data = self.riduci_df_alle_colonne(df)
        except ValueError as e :
            raise(f'Errore nel recupero dei dati : {e}')
        self.train_data_ = self.data[0:self.work_data]
        if  self.work_data != -1:
            self.work_data_ = self.data[self.train_data:self.work_data]
        else:
            self.work_data_ = None
        if self.test_data != -1:
            self.test_data_ = self.data[self.work_data:self.test_data]
        else:
            self.test_data_ = None
        # TODO: posso recuperare il vecchio sistema di set dei dati perche era migliore
        #try:
        #    self.set_data(train_data,work_data,test_data, decrease_data)
        #except  ValueError as e:
        #    raise(e)
        #self.attributi = self.__dict__.copy()


    def pusch_on_db(self):
        #HACK: anche questo viene verificato secondo il nome
        try:
            data_tulpe = [(self.name, self.train_data, self.work_data, self.test_data, self.decrease_data, self.db_references, self.df_or_colonne),]
            dbm.push(data_tulpe, self.DB_SCHEMA, self.INSERT_QUERY)
        except ValueError as e:
            raise(e)
       

    @staticmethod
    def convert_db_response(obj, db_path=''):
        try:
            df_or = obj[7]
            resoult = Dati(id=obj[0], name=obj[1], df_or_colonne=df_or, train_data=int(obj[2]), work_data=int(obj[3]), test_data=int(obj[4]), 
                           decrease_data=obj[5], db_reference=obj[6],data_db_path=db_path)

            return resoult

        except ValueError as e :
            pass
            #raise ValueError(f'Errore nella conversione di una funzione da db ad obj_Dati ERROR: {e}')

    def set_data(self,train_data, work_data, test_data, decrease_data):
       #verifico la coerenza dei parametri
       for param in (train_data, work_data, test_data, decrease_data):
           if not 0.0000 <= param <= 0.999:
               raise ValueError(f"Il parametro {param} deve essere compreso tra 0.001 e 0.999")

       lunghezza_dataset = len(self.data)

       if decrease_data!= 0:
           lenght = int(lunghezza_dataset*decrease_data)
           self.data = self.data[:lenght]

       if train_data+work_data+test_data > 1:
           lunghezza_dataset = len(self.data)
           trainlen = int(lunghezza_dataset*train_data)
           work_data = int(lunghezza_dataset*work_data)
           train_data_ = self.data[:trainlen]
           work_data_ = self.data[trainlen:trainlen+work_data]
           test_data_ = self.data[trainlen+work_data:]
       else:
           lunghezza_dataset = len(self.data)
           trainlen = int(lunghezza_dataset*train_data)
           work_data = int(lunghezza_dataset*work_data)
           test_data = int(lunghezza_dataset*test_data)
           self.train_data_ = self.data[:trainlen]
           self.work_data_ = self.data[trainlen:trainlen+work_data]
           self.test_data_ = self.data[trainlen+work_data:trainlen+work_data+test_data]

    def serializza_colonne(self,df_or_colonne):
        """
        Serializza una lista di colonne di un DataFrame pandas in un JSON.
        Accetta un DataFrame pandas o una lista di stringhe (nomi delle colonne).
        
        Returns:
        - str: Stringa JSON contenente le colonne serializzate.
        """
        # Se df_or_colonne è un DataFrame, estrai i nomi delle colonne
        if isinstance(df_or_colonne, pd.DataFrame):
            colonne = df_or_colonne.columns.tolist()
        elif isinstance(df_or_colonne, list) and all(isinstance(item, str) for item in df_or_colonne):
            colonne = df_or_colonne

        else:
            try:
                colonne = json.loads (df_or_colonne)
            except ValueError as e:
                raise e
            
        colonne_json = json.dumps(colonne)
        print(colonne_json)

        
        return colonne_json

    def riduci_df_alle_colonne(self, data):
        """
        Riduce un DataFrame di pandas alle sole colonne specificate in una lista fornita come stringa JSON.
        
        Returns:
        - pd.DataFrame: DataFrame ridotto contenente solo le colonne specificate.
        """
        try:
            # Deserializza la stringa JSON per ottenere la lista delle colonne
            colonne = json.loads(self.df_or_colonne)
            
            # Verifica che l'input JSON sia una lista
            if not isinstance(colonne, list):
                raise ValueError("L'input JSON deve rappresentare una lista di nomi di colonne.")
            
            # Riduci il DataFrame alle colonne specificate, ignorando quelle non presenti
            colonne_presenti = [col for col in colonne if col in data.columns]
            df_ridotto = data[colonne_presenti]
            
            return df_ridotto
        except json.JSONDecodeError:
            raise ValueError("La stringa fornita non e un JSON valido.")




    
  


