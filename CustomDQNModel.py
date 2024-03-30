from enum import Enum
import json
from os import name
import tensorflow as tf
from Services import Db_Manager as dbm

class layers_type(Enum):

        INPUT = 'input'
        HIDDEN = 'hidden'
        OUTPUT = 'output'

class Layers():

    DB_SCHEMA = '''CREATE TABLE IF NOT EXISTS layers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              layer TEXT,
              name TEXT,
              note TEXT,
              type TEXT,
              schema TEXT
          );
          '''

    INSERT_QUERY = '''INSERT INTO layers (layer, name, note, type, schema)
           VALUES (?, ?, ?, ?, ?);'''
    
    # TODO: schema non sara una stringa andra tipizzato serializzato , deserializzato
    def __init__(self, layer:dict, name, type:layers_type, schema:dict, notes='no_notes', id='Not_posted'):
        self.id = id
        self.layer = layer
        self.name = name
        self.type = type
        self.schema = schema
        self.note = notes

    def push_layer(self):
        layer_json = json.dumps(self.layer)
        schema_json = json.dumps(self.schema)
        tulp = [(layer_json, self.name, self.note, self.type.value, schema_json)]

        # TODO : ho eliminato il controllo sui duplicati perche mi inmpediva troppi push
        dbm.push(tulp, self.DB_SCHEMA, self.INSERT_QUERY, 'layer', 0, 'layers')
    
    @staticmethod
    def convert_db_response(obj):
       try:
           schema_des = json.loads(obj[5])
           lay_deserialized = json.loads(obj[1])
           enum_type = layers_type(obj[4])
           return Layers(layer=lay_deserialized, name=obj[2], type=enum_type, schema=schema_des, notes=obj[3], id=obj[0])
       except ValueError as e :
           raise ValueError(f'Errore nella conversione di un layer da db ad obj_Layer ERROR: {e}')

    def print_attributo(self, nome_attributo):
        # Utilizza getattr per ottenere il valore dell'attributo dal suo nome
        if hasattr(self, nome_attributo):
            valore = getattr(self, nome_attributo)
            tipo = type(valore).__name__
            print(f'"nome atr:{nome_attributo} : valore: {valore} tipo:"{tipo}')
        else:
            print(f"L'attributo '{nome_attributo}' non esiste.")

class CustomDQNModel(tf.keras.Model):

    DB_SCHEMA = '''CREATE TABLE IF NOT EXISTS models (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              model TEXT,
              name TEXT,
              note TEXT,
          );
          '''

    INSERT_QUERY = '''INSERT INTO models (model, name, note)
           VALUES (?, ?, ?);'''

    DB_RELATION_SCHEMA = '''CREATE TABLE IF NOT EXISTS model_layer_relation  (
              id_model INTEGER,
              id_layer INTEGER,
              PRIMARY KEY (id_model, id_layer),
              FOREIGN KEY (id_model) REFERENCES models(id) ON DELETE CASCADE,
              FOREIGN KEY (id_layer) REFERENCES layers(id) ON DELETE CASCADE
              );
            '''

    DB_RELATION_INSERT_QUERY = '''INSERT INTO model_layer_relation  (id_model, id_layer)
           VALUES (?, ?);'''
    
    # TODO creare i layers da lista oggetti
    def __init__(self, lay_obj, name=None):
        super().__init__(name=name)
        self.schema_data = None
        self.schema_input = None
        self.schema_output = None
        self.lay_obj = lay_obj
        self.model_layers = []

        # TODO : fixare la conversione da dizionario ad oggetto
        #self.find_sschemas()

    def build_layers(self, input_shape=None, notes='No Notes', layer_notes='no layer notes'):
        print(self.lay_obj)
        if input_shape is not None:
            try:
                #HINT: tento la sovrascrittura della shape per manggiare diversi windows_size
                # TODO : definizione dei layer tramite dizionario e sovrascrittura della size troppo delicata
                self.lay_obj[0]['params']['input_shape'][0] = input_shape
            except ValueError as e :
                raise ValueError(f'Errore nella sovrascrittura della forma : {e}')

        for layer_config in self.lay_obj:
            layer_type = layer_config['type']
            config = {key: value for key, value in layer_config['params'].items() if key != 'type'}
            try:
                layer = getattr(tf.keras.layers,layer_type)(**config)
                self.model_layers.append(layer)
            except ValueError as e:
                raise ValueError(f'Layer non Instanziato {e}')
        # TODO : sospeso il pusch, inutile dei layer, rimandato pusch intero apost compilazione
        self.push_on_db(notes=notes, layer_notes=layer_notes)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)

        return x

    def print_attributo(self, nome_attributo):
        # Utilizza getattr per ottenere il valore dell'attributo dal suo nome
        if hasattr(self, nome_attributo):
            valore = getattr(self, nome_attributo)
            tipo = type(valore).__name__
            print(f'"nome atr:{nome_attributo} : valore: {valore} tipo:"{tipo}')
        else:
            print(f"L'attributo '{nome_attributo}' non esiste.")

    def find_sschemas(self):

        for lay in self.lay_obj:
            if lay.type == layers_type.HIDDEN:
                self.schema_data = lay.schema

            if lay.type == layers_type.INPUT:
                self.schema_input= lay.schema

            if lay.type == layers_type.OUTPUT:
                self.schema_output= lay.schema

        self.chek_schemas()

    def push_on_db(self, notes='No Notes', layer_notes='no layer notes'):

        dbm.try_table_creation(self.DB_SCHEMA)

        serializzati = self.serialize_to_json()[0]
        print(serializzati)

        exsist = dbm.exists_retrieve('model', serializzati,'models')
        if not exsist[1]:

            # Recupero le tulpe per l insereimento dei layer
            obj_tulpe_list = [(self.serialize_to_json()[0], self.name, notes)]
            dbm.push(obj_tulpe_list, self.DB_SCHEMA, self.INSERT_QUERY)

            ser_lay = self.serialize_Layers_to_json()

            # Devo puschare gòi oggetti copo verifica di corrispondenza 
            if ser_lay.count != ser_lay.count:
                raise ValueError('Uncorresponding N_ Layers')
            
            else:
                for i in range(len(ser_lay)):
                    self.lay_obj[i].layer = ser_lay[i]
                    self.lay_obj[i].name = ser_lay[i].name
                    self.lay_obj[i].push_layer()

            # Recupero gli id inseriti per creare la tabella relazionale
            ids = dbm.exists_retrieve('id', 'layer', 'layers', ser_lay)

            # Carico il modello:
            #TODO: Aggiungi i prametri di push necessaria ad evitare doppioni
            mod_tulpa = [(self.serialize_to_json[0], self.serialize_to_json[1], notes)]
            dbm.push(mod_tulpa, self.DB_SCHEMA, self.INSERT_QUERY)

            mod_id = dbm.exists_retrieve('id', 'models', 'model', mod_tulpa[0][0])

            # carico le relazioni
            ids_tulpe = []
            for id in ids:
                ids_tulpe.append((mod_id[2], id[2]))

            dbm.push(ids_tulpe, self.DB_RELATION_SCHEMA, self.DB_RELATION_INSERT_QUERY)

    ############################## METODI DI SERIALIZZAZIONE #######################
    # Serializzazione
    def serialize_Layers_to_json(self):
        serialized_l = []
        for layer in self.model_layers:
            layer_config = layer.get_config()  # Ottiene la configurazione del layer come dizionario
            lc = layer_config
            serialized_l.append(json.dumps(lc))

        return serialized_l

    def serialized_layers_dict(self):
        serialized_layers_json = self.serialize_Layers_to_json()  # Ottiene la lista delle stringhe JSON
        serialized_layers_dict = [json.loads(layer_json) for layer_json in serialized_layers_json]  # Deserializza ogni stringa JSON in un dizionario
        return serialized_layers_dict

    # Serializzo l intero modello
    def serialize_to_json(self):
        name = 'Not Named'
        # Serializzazione dell'architettura del modello
        model_config = self.to_json()
        # Conversione della configurazione in un dizionario
        model_dict = json.loads(model_config)
        # Aggiunta del nome del modello al dizionario, se presente
        if self.name:
            name = self.name
        return json.dumps(model_dict), name  # Ritorno della stringa JSON con il nome incluso

    @staticmethod
    def seialayze_single_layer(layer, type='JSON'):

        if type == 'DICT':
            return json.load(layer.to_json())
        else:
            return layer.to_json()

    # Deserialize
    def deserialize_Layers(self, list_serialized_layers:list):
        layers = []

        if self.is_json(list_serialized_layers[0]):
            
            for lay in list_serialized_layers:
                des = tf.keras.deserialize(json.loads(lay))
                layers.append(des)

        else:
            if isinstance(list_serialized_layers[0], dict):
                for lay in list_serialized_layers:
                    layers.append(tf.keras.deserialize(lay))
            else:
                raise ValueError('Unsupported Type Error, please provide dict or json lists()')

    ######################### Metodi Ausiliari
    #TODO: pusch degli oggetti layers

    # verifico la scrittura degli schemi:
    def chek_schemas(self):
        if self.schema_data == None or self.schema_output == None or self.schema_input == None:
            raise ValueError('Missed schema')

    # Verifico se e dict o json
    def is_json(self, myjson):
        try:
            json_ob = json.loads(myjson)
        except ValueError as e:
            return False

        return True

    # Deserializzo l' intero modello
    @staticmethod
    def deserialize_from_json(json_str):
        model_dict = json.loads(json_str)
        name = model_dict.pop('name', None)  # Estrazione e rimozione del nome dal dizionario
        # Ricreazione del modello da JSON senza il nome
        model = tf.keras.models.model_from_json(json.dumps(model_dict))
        model.name = name  # Assegnazione del nome al modello deserializzato
        return model

    # Load and Save
    def save_model(self, file_path, save_format='h5'):
        self.save(file_path, save_format=save_format)

    @staticmethod
    def load_model(file_path):
        return tf.keras.models.load_model(file_path)

    # Metodo per salvare solo i pesi del modello
    def save_weights_only(self, file_path):
        self.save_weights(file_path)

# costruzione di dinamica del modello tramite i layer che verranno salvati a parete restituendo un dizionario
# metodo astratto di serializzazione e deserializzazione 
# il db verifichera eventuale ugualianza della lista di layer per evitare dupplicati
# [id, description, [layers_list]]
# eventuali sistemi di callback custom
