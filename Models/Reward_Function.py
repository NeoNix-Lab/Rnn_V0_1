from abc import abstractclassmethod
from Services import Db_Manager as dbm
import sqlite3


class Rewar_Function():
    DB_SCHEMA = '''CREATE TABLE IF NOT EXISTS functions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT,
               function TEXT,
               data_schema TEXT,
               action_schema TEXT,
               status_schema TEXT,
               notes TEXT
           );
           '''

    INSERT_QUERY = '''INSERT INTO functions (name, function, data_schema, action_schema, status_schema, notes)
           VALUES (?, ?, ?, ?, ?, ?);'''

    def __init__(self, name, function, data_schema, action_schema, status_schema, id = 'not_posted_yet'):
        self.id = id
        self.name = name
        self.funaction = function
        self.data_schema = data_schema
        self.action_schema = action_schema
        self.status_schema = status_schema

    def pusch_on_db(self, notes='No Notes'):
        data_tulpe = [(self.name, self.funaction, self.data_schema, self.action_schema, self.status_schema, notes),]
        dbm.push(data_tulpe, self.DB_SCHEMA, self.INSERT_QUERY, 'function', 1, 'functions')

    #def push(self, note='No'):
    #    obj = (self.name, self.funaction, self.data_schema, self.action_schema, self.status_schema, note)
    #    try:
    #        conn = sqlite3.connect('C:\\Users\\user\\OneDrive\\Desktop\\DB\\RNN_Tuning_V01.db')
    #        cursor = conn.cursor()
    #        cursor.execute(self.INSERT_QUERY, obj)

    #        conn.commit()
    #        print(obj)

    #    except sqlite3.Error as e:
    #        print(f"Errore durante il push di {obj} : {e}")
    #        if conn:
    #            conn.rollback()  # Annulla le modifiche in caso di errore

    #    finally:
    #        if conn:
    #            conn.close()

    def get_specific_funtion(self):
        pass

    @staticmethod
    def convert_db_response(obj):
        try:
            return Rewar_Function(obj[1],obj[2],obj[3],obj[4],obj[5],obj[0])
        except ValueError as e :
            raise ValueError(f'Errore nella conversione di una funzione da db ad obj_Function ERROR: {e}')
        

    def verifty_exisistence(self):
        pass

    def build_env(self, data):
        pass


