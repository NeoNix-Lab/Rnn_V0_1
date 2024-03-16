import pandas as pd
import pandera as pa
import json
from Services import Db_Manager as dbm
from io import StringIO


class Dati():
    DB_SCHEMA = '''CREATE TABLE IF NOT EXISTS dati (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT,
               data_schema_txt TEXT,
               notes TEXT
           );
           '''

    INSERT_QUERY = '''INSERT INTO dati (name, data_schema_txt, notes)
          VALUES (?, ?, ?);'''

    def __init__(self, data:pd.DataFrame=None, name='Not_Named', notes='No_Notes', id='Not_Posted_Yet'):
        self.id = id
        self.name = name
        self.data_schema = None
        self.data_schema_txt = None
        self.build_schema(data)
    
    def build_schema(self, dati:pd.DataFrame):
        if dati is not None:
            self.data_schema = pa.infer_schema(dati)
            self.data_schema_txt = self.data_schema.to_json()

    def pusch_on_db(self, notes='No Notes'):
        data_tulpe = [(self.name, str(self.data_schema_txt), notes),]
        dbm.push(data_tulpe, self.DB_SCHEMA, self.INSERT_QUERY, 'data_schema_txt', 1, 'dati')

    @staticmethod
    def convert_db_response(obj):
        try:
            resoult = Dati(id=obj[0], name=obj[1], notes=obj[3])
            resoult.data_schema_txt = obj[2]

            # Utilizza StringIO per creare un oggetto file-like dalla stringa JSON
            json_like_file = StringIO(obj[2])

            resoult.data_schema = pa.DataFrameSchema.from_json(json_like_file)

            return resoult

        except ValueError as e :
            raise ValueError(f'Errore nella conversione di una funzione da db ad obj_Dati ERROR: {e}')
     
    @staticmethod
    def retrive_all_from_db():
        response = dbm.retrive_all('dati')
        lis=[]

        for res in response:
            obj = Dati.convert_db_response(res)
            lis.append(obj)

        return lis


    # TODO: costruire lo schema relazionale dati da a f, e valida dati in ingresso per env
    # f genera due schemi da passare al modello che  verranno validati con un tentativo d'iterazione 
    # modello genera l ambiente dopo aver usato il suo schema per validare i dati in ingresso e dop aver passato 
    # il numero di timesteps utilizzato per costruire gli strati 




