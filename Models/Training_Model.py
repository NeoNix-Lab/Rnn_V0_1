from abc import abstractmethod
from enum import Enum
from datetime import datetime
from Services import Db_Manager as dbm
from datetime import datetime as dt

class Training_statu(Enum):
    PLANNED = 'planned'
    TRAINING = 'training'
    TRAINED = 'trained'

class Training_Model():
    # TODO: aggiungere i trading data as dictionary fees, initial_balance
    DB_SCHEMA = '''CREATE TABLE IF NOT EXISTS training (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               status TEXT,
               creation_date TEXT,
               function_id INTEGER,
               process_id INTEGER,
               log_path TEXT,
               best_result TEXT,  -- Assumendo che il miglior risultato sia testuale; cambia il tipo di dati se necessario
               notes TEXT,
               name TEXT,
               FOREIGN KEY (function_id) REFERENCES functions(id),  -- Assicurati che 'functions' sia il nome corretto della tabella e 'id' il nome della colonna chiave primaria
               FOREIGN KEY (process_id) REFERENCES processes(id)  -- Assicurati che 'processes' sia il nome corretto della tabella e 'id' il nome della colonna chiave primaria
           );
           '''

    INSERT_QUERY = '''INSERT INTO training (status, creation_date, function_id, process_id, log_path, best_result, notes, name)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?);'''

    INSERT_QUERY_RELATION ='''INSERT INTO training_layers (training_id, layer_id)
           VALUES (?, ?);'''

    DB_SCHEMA_RELATION = '''CREATE TABLE IF NOT EXISTS training_layers (
                training_id INTEGER NOT NULL,
                layer_id INTEGER NOT NULL,
                FOREIGN KEY (training_id) REFERENCES training(id) ON DELETE CASCADE,
                FOREIGN KEY (layer_id) REFERENCES layers(id) ON DELETE CASCADE,
                PRIMARY KEY (training_id, layer_id) ON CONFLICT IGNORE
            );
            '''

    #NOW = datetime.datetime.now()

    def __init__(self, name, status:Training_statu, function_id, process_id, log_path, id='not_posted_yet',
                 creation_data=dt.now().strftime('%Y-%m-%d %H:%M:%S'), best_resoult='Not_Trained_Yet'):
        self.id = id
        self.status = status
        self.function_id = function_id
        self.process_id = process_id
        self.creation_date = creation_data
        self.log_path = log_path
        self.best_resoult = best_resoult
        self.name = name
        self.attributi = self.__dict__.copy()

    # TODO: recuperare id di layers e addestramento
    # TODO: manca il sistema per evitare i doppioni (possibilie implementazione tramite il nome)
    def pusch_on_db(self, layers_id:list, notes='No Notes'):
        dbm.try_table_creation(self.DB_SCHEMA_RELATION)

        data_tulpe = [(self.status.value, self.creation_date, self.function_id, self.process_id, self.log_path, self.best_resoult, notes, self.name)]
        dbm.push(data_tulpe, self.DB_SCHEMA, self.INSERT_QUERY)

        last_id = dbm.retrive_last('training')[0]

        data_lay_tulpe = []
        # HINT: sto aggiungendo una lista di entry
        for i in layers_id:
            data_lay_tulpe.append((last_id, i)) 

        # HINT: non c'e volutamente controllo di doppioni per permettere piu inserimenti per ogni tipo
        dbm.push(data_lay_tulpe, self.DB_SCHEMA_RELATION, self.INSERT_QUERY_RELATION)

    @staticmethod
    def convert_db_response(response):
        status = Training_statu(response[1])
        obj = Training_Model(response[8], status, response[3], response[4], response[5], id=response[0], creation_data=response[2],
                             best_resoult=response[6])
        return obj

    # TODO: some implementation not implemented
    def update_status():
        pass

    def update_best_resoult(self):
        pass

    @abstractmethod
    def retrive_list_records_by_name(names:list[str]):
        return dbm.retive_a_list_of_recordos('name', 'processes', names)

    def build_training_model_from_record(record):
        try:
            # Assumi che record sia una tupla con tutti i valori necessari per Training_Model
            training_model = Training_Model(
                status=record[1],
                creation_data=record[2],
                function_id=record[3],
                process_id=record[4],
                log_path=record[5],
                best_resoult=record[6],
                notes=record[7],
                name=record[8],
                id=record[0] 
            )
            return training_model

        except Exception as e:
            print(f"Errore durante la mappatura del record su Training_Model: {record}: {e}")
            return None

    def verifty_exisistence(self):
        pass


