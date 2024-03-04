from Services import Db_Manager as dbm

class Rewar_Function():

    DB_SCHEMA = '''CREATE TABLE IF NOT EXISTS functions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT,
               function TEXT,
               data_schema TEXT,
               action_schema TEXT,
               status_schema TEXT,  -- Chiave esterna che fa riferimento a 'models'
               notes TEXT,
           );
           '''

    INSERT_QUERY = '''INSERT INTO functions (name, function, data_schema, action_schema, status_schema, notes)
           VALUES (?, ?, ?, ?, ?, ?);'''

    def __init__(self, name, function, data_schema, action_schema, status_schema):
        self.id = 'not_posted_yet'
        self.name = name
        self.funaction = function
        self.data_schema = data_schema
        self.action_schema = action_schema
        self.status_schema = status_schema

    def pusch_on_db(self, notes='No Notes'):
        data_tulpe = [(self.name, self.function, self.data_schema, self.action_schema, self.status_schema, notes)]
        dbm.push(data_tulpe, self.DB_SCHEMA, self.INSERT_QUERY, 'function', 1, 'functions')

    def get_specific_funtion(self):
        pass

    def convert_db_resoult_into_obbj(self):
        pass

    def verifty_exisistence(self):
        pass

    def build_env(self, data):
        pass


