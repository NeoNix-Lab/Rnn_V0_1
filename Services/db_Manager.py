import sqlite3
import os
import json
from typing import List, Tuple, Union, Any
from Services.config import Config

# TODO: make it as a class in order to menage config
config = Config()

# HACK: separare i metodi aiuterebbe la notifica degli errori
# Db base path
DB_BASE_PATH = config.models_path

def try_table_creation(tab_schema):
     try:
        conn = sqlite3.connect(DB_BASE_PATH)
        cursor = conn.cursor()
        cursor.execute(tab_schema)

        conn.commit()

     except sqlite3.Error as e:
         print(f"Errore del database: {e}")
         if conn:
             conn.rollback()  # Annulla le modifiche in caso di errore

     finally:
         if conn:
             conn.close()

def change_db_path(db_file_path):
    global DB_BASE_PATH
    directory = os.path.dirname(db_file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    DB_BASE_PATH = db_file_path

def retrive_all(tab_name):
    try:
        conn = sqlite3.connect(DB_BASE_PATH)
        cursor = conn.cursor()
        # Prepara la query SQL per cercare un layer con lo stesso nome e configurazione
        query = f'''SELECT * FROM {tab_name}'''
        cursor.execute(query)
        all = cursor.fetchall()
        
        return all

    except sqlite3.Error as e:
        print(f"Errore durante il recupero di tutti gli oggetti da: {tab_name}: {e}")

    finally:
       if conn:
           conn.close()

def exists_retrieve(prop_name, val_name, tab_name, obj_values) -> List[Tuple[str, bool, Any]]:

    """
    Verifica l'esistenza di uno o piu oggetti nella tabella specificata e restituisce i dettagli.
    
    Args:
        prop_name (str): Nome della proprieta chiave nella tabella per il confronto.
        val_name (str): Nome della colonna chiave nella tabella per il confronto.
        tab_name (str): Nome della tabella in cui cercare l'oggetto.
        obj_values (Union[str, List[str]]): Valore(i) dell'oggetto da cercare. Puo essere una stringa singola o una lista di stringhe.
    
    Returns:
        List[Tuple[str, bool, Any]]: Una lista di tuple dove ogni tupla contiene il valore di obj_value analizzato,
                                     un booleano che indica se l'oggetto esiste o non esiste nel database,
                                     e il parametri dell'oggetto (di qualsiasi tipo) se esiste, altrimenti None.
    """


    conn = None
    try:
        conn = sqlite3.connect(DB_BASE_PATH)
        cursor = conn.cursor()
        
        if not isinstance(obj_values, list):
            obj_values = [obj_values]  # Trasforma in lista se non lo è
        
        results = []
        for obj_value in obj_values:
            query = f'''SELECT {prop_name} FROM {tab_name} WHERE {val_name}=? LIMIT 1;'''
            cursor.execute(query, (obj_value,))
            result = cursor.fetchone()
            if result:
                results.append((obj_value, True, result[0]))  # Oggetto esistente con ID
            else:
                results.append((obj_value, False, None))  # Oggetto non esistente
        return results

    except sqlite3.Error as e:
        print(f"Errore durante la verifica dell esistenza degli oggetti in {tab_name}: {e}")
        return [(obj_value, False, None) for obj_value in obj_values]  # Restituisce False e None per ogni valore in caso di errore

    finally:
       if conn:
           conn.close()

def retive_a_list_of_recordos(val_name:str, tab_name:str, obj_values:list) -> List[any]:

    #"""
    #Verifica l'esistenza di uno o piu oggetti nella tabella specificata e restituisce tutti i dettagli.
    
    #Args:
    #    val_name (str): Nome della colonna chiave nella tabella per il confronto.
    #    tab_name (str): Nome della tabella in cui cercare l'oggetto.
    #    obj_values (List[any]]): Valore(i) dell'oggetto da cercare. Puo essere una stringa singola o una lista di stringhe.
    
    #Returns:
    #    List[any]: Una lista di Any dove Rappresentanti la risposta alla ricerca di uno specifico oggetto
    #"""


    conn = None
    try:
        conn = sqlite3.connect(DB_BASE_PATH)
        cursor = conn.cursor()
        
        if not isinstance(obj_values, list):
            obj_values = [obj_values]
        
        results = []
        for obj_value in obj_values:
            query = f'''SELECT * FROM {tab_name} WHERE {val_name}=?'''
            cursor.execute(query, (obj_value,))
            records = cursor.fetchall() 
            for record in records:
                results.append(record)

        return results

    except sqlite3.Error as e:
        print(f"Errore durante il recupero degli oggetti oggetti in {tab_name}: {e}")
        return None

    finally:
       if conn:
           conn.close()

# TODO: ho perso i rilanci delle eccezioni!!!!!!!!!!!!!!!!!!!
def push(obj_list:list, tab_schema:str, query, unique_colum=None, unique_value_index=None, tabb_name=None):
    """
    Inserisce qualsiasi lista di oggetti sulla base di una query ed uno schema , evita inserimenti multipli se unique_colum, unique colum e tabb name sono 
        diversi da zero

    Args:
        unique_colum (str): Nome della propieta chiave nella tabella per il confronto di unicita
        unique_value_index (int): Indice del valore di confronto di unicita, indice della tulpa d'inserimento.
        tab_name (str): Nome della tabella in cui cercare l'oggetto.
   
    """
    # TODO: ritorno gli elementi esistenti ma senza un associazione , in caso di liste
    
    try:
        returned_objs = []
        conn = sqlite3.connect(DB_BASE_PATH)
        cursor = conn.cursor()
        cursor.execute(tab_schema)

        for obj in obj_list:
            if unique_colum is not None and unique_value_index is not None and tabb_name is not None:
                check_query = f"SELECT EXISTS(SELECT 1 FROM {tabb_name} WHERE {unique_colum}=?)"
                check_values = (obj[unique_value_index],)
                cursor.execute(check_query, check_values)
                exists = cursor.fetchone()[0]

                if exists == 0:
                    try:
                        cursor.execute(query, obj)
                    except ValueError as e :
                        raise(f'@@@@@@@@@@@@@@@@@[[[[[[[[[@{e}')
                
                check_query_ = f"SELECT * FROM {tabb_name} WHERE {unique_colum}=?"
                check_values_ = (obj[unique_value_index],)
                cursor.execute(check_query_, check_values_)
                existing_obj = cursor.fetchone()
                returned_objs.append(existing_obj)

            else :
                cursor.execute(query, obj)
            
        conn.commit()
        

    except ValueError as e:
        raise(f"Errore durante il push di : {e}")

        if conn:
            conn.rollback()  # Annulla le modifiche in caso di errore

    finally:
        return returned_objs
        if conn:
            conn.close()

def retrive_last(tab_name, prop_name='id'):
    try:
        conn = sqlite3.connect(DB_BASE_PATH)
        cursor = conn.cursor()
        query = f"SELECT {prop_name} FROM {tab_name} ORDER BY id DESC LIMIT 1"
        cursor.execute(query)
        record  = cursor.fetchone()

        return record
    except ValueError as e:
        print(f"##################################Errore durante il recupero dell ultimo di {prop_name} nella ttabella {tab_name}: {e}")
        raise(f"Errore durante il recupero dell ultimo di {prop_name} nella ttabella {tab_name}: {e}")
        if conn:
            conn.rollback()  # Annulla le modifiche in caso di errore

    finally:
        if conn:
            conn.close()

#def update_record(tabb_name, prop_name, prop_value, comparation_prop_name, comparation_prop_value):
#    try:
#        conn = sqlite3.connect(DB_BASE_PATH)
#        cursor = conn.cursor()

        
#        check_query = f"UPDATE {tabb_name} SET {prop_name} = {prop_value} WHERE {comparation_prop_name} = {comparation_prop_value}"

#        cursor.execute(query)

#        conn.commit()

#    except ValueError as e:
#        raise(f"Errore durante il push di : {e}")

#        if conn:
#            conn.rollback()  

#    finally:
#        if conn:
#            conn.close()

def push_debugger(obj_list:list, tab_schema:str, query):
     try:
         conn = sqlite3.connect(DB_BASE_PATH)
         cursor = conn.cursor()
         cursor.execute(tab_schema)

         for obj in obj_list:
                 cursor.execute(query, obj)
         conn.commit()

     except ValueError as e:
         #print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Errore durante il push di {obj_list} sull oggetto {obj} in {query}: {e}")
         raise(f"Errore durante il push di : {e}")

         if conn:
             conn.rollback()  # Annulla le modifiche in caso di errore

     finally:
         if conn:
             conn.close()

def new_update_record(table_name: str, updates: dict, conditions: dict):
    """
    Aggiorna i record specificati in una tabella.

    Args:
    table_name (str): Nome della tabella in cui aggiornare il record.
    updates (dict): Dizionario delle colonne e dei nuovi valori da aggiornare.
    conditions (dict): Dizionario delle condizioni da rispettare per l'aggiornamento (WHERE clause).

    Returns:
    None: Aggiorna i record nel database e gestisce le eccezioni internamente.
    """
    # HINT: Esempio di utilizzo:
    # update_record('nome_tabella', {'colonna_da_aggiornare': 'nuovo_valore'}, {'colonna_condizione': 'valore_condizione'})
    try:
        conn = sqlite3.connect(DB_BASE_PATH)
        cursor = conn.cursor()

        # Preparazione della query di aggiornamento
        update_clause = ', '.join([f"{key} = ?" for key in updates.keys()])
        condition_clause = ' AND '.join([f"{key} = ?" for key in conditions.keys()])
        query = f"UPDATE {table_name} SET {update_clause} WHERE {condition_clause}"

        # Preparazione dei valori per la query
        update_values = list(updates.values())
        condition_values = list(conditions.values())
        query_values = update_values + condition_values

        # Esecuzione della query
        cursor.execute(query, query_values)
        conn.commit()

    except sqlite3.Error as e:
        print(f"Errore durante l'aggiornamento del record in {table_name}: {e}")
        if conn:
            conn.rollback()  # Annulla le modifiche in caso di errore

    finally:
        if conn:
            conn.close()


