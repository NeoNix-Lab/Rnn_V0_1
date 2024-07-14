from abc import abstractclassmethod
from Services import db_Manager as dbm
import sqlite3
import json
from Models.baseModelsClass import BaseModelsClass as BCM

#TODO: manca la documentazione


class Rewar_Function(BCM):
    #TODO: attenzione all ordine degli schemi
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

    def __init__(self, name, function, data_schema:dict, action_schema:dict, status_schema:dict, id = 'not_posted_yet'):
        self.id = id
        self.name = name
        self.funaction = function
        self.data_schema = data_schema
        self.action_schema = action_schema
        self.status_schema = status_schema

    def push_on_db(self, notes='No Notes'):
        try:
            dat = json.dumps(self.data_schema)
            act = json.dumps(self.action_schema)
            stat = json.dumps(self.status_schema)

            data_tulpe = [(self.name, self.funaction, dat, act, stat, notes),]
            dbm.push(data_tulpe, self.DB_SCHEMA, self.INSERT_QUERY, 'function', 1, 'functions')
        except ValueError as e:
            raise e
       

    def get_specific_funtion(self):
        pass

    @staticmethod
    def convert_db_response(obj,notes=''):
        try:
            act_sch = json.loads(obj[4])
            dat_sch = json.loads(obj[3])
            stat_sch = json.loads(obj[5])

            return Rewar_Function(obj[1],obj[2],dat_sch, act_sch,stat_sch,obj[0])
        except ValueError as e :
            raise ValueError(f'Errore nella conversione di una funzione da db ad obj_Function ERROR: {e}')
        

    def verifty_exisistence(self):
        pass

    def build_env(self, data):
        pass


