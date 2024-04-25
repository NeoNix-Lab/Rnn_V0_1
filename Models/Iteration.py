import json
from Services import Db_Manager as dbm

class Iterazione:
    """
    Rappresenta un'iterazione di un processo di machine learning,
    collegando risultati di training, test e work a specifici set di dati.
    """
    DB_SCHEMA = '''CREATE TABLE IF NOT EXISTS iterazioni (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT,
                   dati_id INTEGER,
                   training_id INTEGER,
                   train_result REAL,
                   test_result REAL,
                   work_result REAL,
                   log_path TEXT,
                   FOREIGN KEY(dati_id) REFERENCES dati(id),
                   FOREIGN KEY(training_id) REFERENCES training(id)
               );
               '''

    INSERT_QUERY = '''INSERT INTO iterazioni (name, dati_id, training_id, train_result, test_result, work_result, log_path)
                      VALUES (?, ?, ?, ?, ?, ?, ?);'''

    def __init__(self, name, dati_id, training_id, train_result=0, test_result=0, work_result=0, id='Not_Posted_Yet', log_path='Not_posted_yet'):
        self.id = id
        self.name = name
        self.dati_id = dati_id
        self.training_id = training_id
        self.train_result = train_result
        self.test_result = test_result
        self.work_result = work_result
        self.log_path = log_path
        #self.attributi = self.__dict__.copy()


    def push_on_db(self):
        """
        Inserisce l'iterazione corrente nel database.
        """
        data_tuple = [(self.name, self.dati_id, self.training_id, self.train_result, self.test_result, self.work_result, self.log_path)]
        dbm.push(data_tuple, self.DB_SCHEMA, self.INSERT_QUERY)

    @staticmethod
    def convert_db_response(obj):
        """
        Converte una risposta del database in un oggetto Iterazione.
        """
        try:
            result = Iterazione(name=obj[1], dati_id=obj[2], training_id=obj[3], train_result=obj[4], 
                                test_result=obj[5], work_result=obj[6], id=obj[0], log_path=obj[7])
            return result
        except ValueError as e:
            raise ValueError(f'Errore nella conversione del record del database in oggetto Iterazione: {e}')

    @staticmethod
    def update_resoult(iteration_id, update_dict:dict):
        dbm.new_update_record('iterazioni', update_dict, {'id':iteration_id})
        