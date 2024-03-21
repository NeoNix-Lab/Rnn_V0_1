import sqlite3
import os
import json
from typing import List, Tuple, Union, Any

# Db base path
# TODO: probabilmente sara necessario aggiornare la path del db on Run
DB_BASE_PATH = 'C:\\Users\\user\\OneDrive\\Desktop\\DB\\RNN_Tuning_V01.db'

#@staticmethod
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

def change_db_path(path):
    global DB_BASE_PATH

    if not os.path.exists(path):
        os.makedirs(path)

    DB_BASE_PATH = path

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

    """
    Verifica l'esistenza di uno o piu oggetti nella tabella specificata e restituisce tutti i dettagli.
    
    Args:
        val_name (str): Nome della colonna chiave nella tabella per il confronto.
        tab_name (str): Nome della tabella in cui cercare l'oggetto.
        obj_values (List[any]]): Valore(i) dell'oggetto da cercare. Puo essere una stringa singola o una lista di stringhe.
    
    Returns:
        List[any]: Una lista di Any dove Rappresentanti la risposta alla ricerca di uno specifico oggetto
    """


    conn = None
    try:
        conn = sqlite3.connect(DB_BASE_PATH)
        cursor = conn.cursor()
        
        if not isinstance(obj_values, list):
            obj_values = [obj_values]
        
        results = []
        for obj_value in obj_values:
            query = f'''SELECT * FROM {tab_name} WHERE {val_name}=?;'''
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

#@staticmethod
def push(obj_list:list, tab_schema:str, query, unique_colum=None, unique_value_index=None, tabb_name=None):
    """
    Inserisce qualsiasi lista di oggetti sulla base di una query ed uno schema , evita inserimenti multipli se unique_colum, unique colum e tabb name sono 
        diversi da zero

    Args:
        unique_colum (str): Nome della propieta chiave nella tabella per il confronto di unicita
        unique_value_index (int): Indice del valore di confronto di unicita, indice della tulpa d'inserimento.
        tab_name (str): Nome della tabella in cui cercare l'oggetto.
   
    """
    # TODO: manca una verifica di esistenza su tutti gli elementi
    try:
        conn = sqlite3.connect(DB_BASE_PATH)
        cursor = conn.cursor()
        cursor.execute(tab_schema)

        for obj in obj_list:
            if unique_colum is not None and unique_value_index is not None and tabb_name is not None:
                check_query = f"SELECT EXISTS(SELECT 1 FROM {tabb_name} WHERE {unique_colum}=?)"
                check_values = (obj[unique_value_index],)
                cursor.execute(check_query, check_values)
                exists = cursor.fetchone()[0]
                #print(exists)

                if not exists:
                    cursor.execute(query, obj)

            else :
                cursor.execute(query, obj)

        conn.commit()

    except ValueError as e:# sqlite3.Error as e:
        raise(f"Errore durante il push di {obj_list} sull oggetto {obj} in {query}: {e}")
        if conn:
            conn.rollback()  # Annulla le modifiche in caso di errore

    finally:
        if conn:
            conn.close()

#def layer_all():
#        try:
#            conn = sqlite3.connect(DB_BASE_PATH)
#            cursor = conn.cursor()
#            # Prepara la query SQL per cercare un layer con lo stesso nome e configurazione
#            query = '''SELECT layer FROM layers'''
#            cursor.execute(query)
#            layers = cursor.fetchall()
            
#            return layers

#        except sqlite3.Error as e:
#            print(f"Errore durante la verifica dell'esistenza del layer: {e}")

#        finally:
#           if conn:
#               conn.close()

#def model_all():
#        try:
#            conn = sqlite3.connect(DB_BASE_PATH)
#            cursor = conn.cursor()
#            # Prepara la query SQL per cercare un layer con lo stesso nome e configurazione
#            query = '''SELECT model FROM models'''
#            cursor.execute(query)
#            layers = cursor.fetchall()
            
#            return layers

#        except sqlite3.Error as e:
#            print(f"Errore durante la verifica dell'esistenza del model: {e}")

#        finally:
#           if conn:
#               conn.close()

#def layer_exists(layer_config_json):
#        print(layer_config_json)
#        try:
#            conn = sqlite3.connect(DB_BASE_PATH)
#            cursor = conn.cursor()
#            # Prepara la query SQL per cercare un layer con lo stesso nome e configurazione
#            query = '''SELECT EXISTS(SELECT 1 FROM layers WHERE layer=?)'''
#            cursor.execute(query, (str(layer_config_json)))
#            exists = cursor.fetchone()[0]
#            return exists == 1

#        except sqlite3.Error as e:
#            print(f"Errore durante la verifica dell'esistenza del layer: {e} : {exists}")

#        finally:
#           if conn:
#               conn.close()




#def push_Layers(dict_layer_list):
#    # TDOD: manca la tabella relazionale models_layers
#    try:
#        conn = sqlite3.connect(DB_BASE_PATH)
#        cursor = conn.cursor()
#        cursor.execute('''
#            CREATE TABLE IF NOT EXISTS layers (
#                id INTEGER PRIMARY KEY AUTOINCREMENT,
#                layer_name TEXT,
#                layer TEXT
#            )
#        ''')

#        layers = layer_all()
#        xl_layer = []

#        if xl_layer is not None:
#            for lay in layers:
#                xl_layer.append(lay[0])

#        for dict_layer in dict_layer_list:

#            if 'name' in  dict_layer:
#                nome = dict_layer['name']
#            else:
#                nome = 'Non Nominato'

#            json_layer = json.dumps(dict_layer)

#            if not (xl_layer.__contains__(json_layer)):
#                cursor.execute('''
#                INSERT INTO layers (layer_name, layer)
#                VALUES (?, ?)
#            ''', (nome, json_layer))

#                conn.commit()


#    except sqlite3.Error as e:
#        print(f"Errore del database: {e}")
#        if conn:
#            conn.rollback()  # Annulla le modifiche in caso di errore
#    finally:
#        if conn:
#            conn.close()

#def push_Model(dict_model, name):
#    # TODO: manca un vincolo sul tipo di dati ___ Implementare metadati
#    try:
#        conn = sqlite3.connect(DB_BASE_PATH)
#        cursor = conn.cursor()
#        cursor.execute('''
#            CREATE TABLE IF NOT EXISTS models (
#                id INTEGER PRIMARY KEY AUTOINCREMENT,
#                model_name TEXT,
#                model TEXT
#            )
#        ''')

#        models = model_all()
#        xl_model = []

#        if models is not None:
#            for mod in models:
#                xl_model.append(mod[0])

#        json_model = json.dumps(dict_model)

#        if not (xl_model.__contains__(json_model)):
#            cursor.execute('''
#            INSERT INTO models (model_name, model)
#            VALUES (?, ?)
#        ''', (name, json_model))

#            conn.commit()


#    except sqlite3.Error as e:
#        print(f"Errore del database: {e}")
#        if conn:
#            conn.rollback()  # Annulla le modifiche in caso di errore
#    finally:
#        if conn:
#            conn.close()
