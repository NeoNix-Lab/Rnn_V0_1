from abc import abstractmethod
from enum import Enum
from datetime import datetime
from Services import Db_Manager as dbm

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
               model_id INTEGER,  -- Chiave esterna che fa riferimento a 'models'
               log_path TEXT,
               best_result TEXT,  -- Assumendo che il miglior risultato sia testuale; cambia il tipo di dati se necessario
               notes TEXT,
               name TEXT,
               FOREIGN KEY (function_id) REFERENCES functions(id),  -- Assicurati che 'functions' sia il nome corretto della tabella e 'id' il nome della colonna chiave primaria
               FOREIGN KEY (process_id) REFERENCES processes(id),  -- Assicurati che 'processes' sia il nome corretto della tabella e 'id' il nome della colonna chiave primaria
               FOREIGN KEY (model_id) REFERENCES models(id)  -- Assicurati che 'models' sia il nome corretto della tabella e 'id' il nome della colonna chiave primaria
           );
           '''

    INSERT_QUERY = '''INSERT INTO training (status, creation_date, function_id, process_id, model_id, log_path, best_result, notes, name)
           VALUES (?,?, ?, ?, ?, ?, ?, ?, ?);'''

    NOW = datetime.now()

    def __init__(self, name, status:Training_statu, function_id, process_id, log_path, model_id):
        self.id = 'not_posted_yet'
        self.status = Training_statu.PLANNED
        self.function_id = function_id
        self.process_id = process_id
        self.modele_id = model_id
        self.creation_date = self.now.strftime('%Y-%m-%d %H:%M:%S')
        self.log_path = log_path
        self.best_resoult = 'Not_Trained_Yet'
        self.name = name


    def pusch_on_db(self, notes='No Notes'):
        data_tulpe = [(self.status, self.creation_date, self.function_id, self.process_id, self.modele_id, self.log_path, self.best_resoult, notes)]
        dbm.push(data_tulpe, self.DB_SCHEMA, self.INSERT_QUERY)

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
                creation_date=record[2],
                function_id=record[3],
                process_id=record[4],
                model_id=record[5],
                log_path=record[6],
                best_result=record[7],
                notes=record[8],
                name=record[9],
                _id=record[0]  # Assumi che il primo elemento del record sia l'ID
            )
            return training_model

        except Exception as e:
            print(f"Errore durante la mappatura del record su Training_Model: {record}: {e}")
            return None

    def verifty_exisistence(self):
        pass


