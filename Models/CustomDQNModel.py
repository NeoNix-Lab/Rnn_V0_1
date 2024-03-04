from abc import abstractclassmethod
import json
from os import name
import tensorflow as tf
from keras.layers import deserialize
from Services import Db_Manager as dbm

class Layers():

    DB_SCHEMA = '''CREATE TABLE IF NOT EXISTS layers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              layer TEXT,
              name TEXT,
              note TEXT,
          );
          '''

    INSERT_QUERY = '''INSERT INTO layers (layer, name, note)
           VALUES (?, ?, ?);'''

    def __init__(self, layer, name, notes='no_notes'):
        self.id = 'Not_posted'
        self.layer = layer
        self.name = name
        self.note = notes

    def push_layer(self):
        tulp = [(self.layer, self.name, self.note)]

        dbm.push(tulp, self.DB_SCHEMA, self.INSERT_QUERY, 'layer', 0, 'layers')

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

    def __init__(self, layer_config, name=None, notes='No Notes', layer_notes='no layer notes'):
        super().__init__(name=name)
        self.model_layers = []
        for config in layer_config:
            layer_type = config['type']
            config.pop('type')
            try:
                layer = getattr(tf.keras.layers,layer_type)(**config)
                self.model_layers.append(layer)
            except:
                raise ValueError('Layer non Instanziato')

        self.push_on_db(notes=notes, layer_notes=layer_notes)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)

        return x

    def push_on_db(self, notes='No Notes', layer_notes='no layer notes'):

        dbm.try_table_creation(self.DB_SCHEMA)

        exsist = dbm.exists_retrieve('model', self.serialize_to_json[0],'models')
        if not exsist[1]:

            # Recupero le tulpe per l insereimento dei layer
            obj_tulpe_list = [(self.serialize_to_json()[0], self.name, notes)]
            dbm.push(obj_tulpe_list, self.DB_SCHEMA, self.INSERT_QUERY)

            tulpe_lay_list = []
            ser_lay = self.serialize_Layers_to_json()
            for i in range(len(ser_lay)):
                tulpe_lay_list.append((ser_lay[i], self.layers[i].name, layer_notes))

            dbm.push(ser_lay, Layers.DB_SCHEMA, Layers.INSERT_QUERY)

            # Recupero gli id inseriti per creare la tabella relazionale
            ids = dbm.exists_retrieve('id', 'layer', 'layers', ser_lay)

            # Carico il modello:
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
                des = deserialize(json.loads(lay))
                layers.append(des)

        else:
            if isinstance(list_serialized_layers[0], dict):
                for lay in list_serialized_layers:
                    layers.append(deserialize(lay))
            else:
                raise ValueError('Unsupported Type Error, please provide dict or json lists()')

    # Metodi Ausiliari
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
