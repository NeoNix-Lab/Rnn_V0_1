from enum import Enum
import pandas as pd
import pandera as pa
import json
from Services import Db_Manager as dbm
from io import StringIO


class Dati():
    class DatiVerifica(Enum):
        #TODO: l ho rimosso perceh dava problemi
        ASSENTI : -1
        VERIFICATI : 1
        NON_VERIFICATI: 0

    #TODO: probabilmente schema text e inutile
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
        self._dati = data
        self.build_schema(self._dati)
        self.pusch_on_db(f'Nuovi dati {self.name}')
        self.verifica = self.verify_data()

    def update_dati(self, new_data:pd.DataFrame):
        self._dati = new_data
        res = self.verify_data()
        self.verifica = res
        return res

    
    def build_schema(self, dati:pd.DataFrame):
        try:
            if dati is not None:
                self.data_schema = pa.infer_schema(dati)
                self.data_schema_txt = self.data_schema.to_json()
        except ValueError as e:
            raise e
    
    def verify_data(self):
        if self._dati is None:
                return 'assenti'
        else:
            try:
                self.data_schema.validate(self._dati)
                return 'verificati'
            except ValueError as e :
                raise(f'Errore nella verifica dei dati : {e}')
                return 'non_verificati'

    def pusch_on_db(self, notes='No Notes'):
        dumped_data_schema = str(self.data_schema)
        data_tulpe = [(self.name, dumped_data_schema, notes),]#str(self.data_schema_txt)
        dbm.push(data_tulpe, self.DB_SCHEMA, self.INSERT_QUERY, 'data_schema_txt', 1, 'dati')

    @staticmethod
    def convert_db_response(obj):
        try:
            #dese_sch = json.loads(obj[2])
            resoult = Dati(id=obj[0], name=obj[1], notes=obj[3])
            #resoult.data_schema_txt = dese_sch

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

    @staticmethod
    def retrive_list_by_schema(schema_txt):
        response = dbm.retive_a_list_of_recordos('data_schema_txt', 'dati', schema_txt)
        lis = []
        for r in response:
            obj = Dati.convert_db_response(r)
            lis.append(obj)

        return lis


    # TODO: costruire lo schema relazionale dati da a f, e valida dati in ingresso per env
    # f genera due schemi da passare al modello che  verranno validati con un tentativo d'iterazione 
    # modello genera l ambiente dopo aver usato il suo schema per validare i dati in ingresso e dop aver passato 
    # il numero di timesteps utilizzato per costruire gli strati 

    # HACK: implementare per recuperare e o validare dati da diverse fonti e diversi fomati db string pd.dataframe
    def validate_data(self, data:pd.DataFrame):
        try:
            return self.data_schema.validate(data)
        except ValueError as e:
            raise ValueError(f'Errore nella validazione : {e}')


