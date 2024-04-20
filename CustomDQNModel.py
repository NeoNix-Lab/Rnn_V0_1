from enum import Enum
import json
from os import name
import tensorflow as tf
from Services import Db_Manager as dbm
import keras as k


class layers_type(Enum):
        INPUT = 'input'
        HIDDEN = 'hidden'
        OUTPUT = 'output'

#region layers
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
#endregion

class CustomDQNModel(tf.keras.Model):

    DB_SCHEMA = '''CREATE TABLE IF NOT EXISTS models (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              model TEXT,
              name TEXT,
              note TEXT
          );
          '''

    INSERT_QUERY = '''INSERT INTO models (model, name, note)
           VALUES (?, ?, ?);'''

    DB_RELATION_SCHEMA = '''CREATE TABLE IF NOT EXISTS model_layer_relation  (
              id_model INTEGER,
              id_layer INTEGER,
              layer_index INTEGER,
              PRIMARY KEY (id_model, layer_index) ON CONFLICT IGNORE
              FOREIGN KEY (id_model) REFERENCES models(id) ON DELETE CASCADE,
              FOREIGN KEY (id_layer) REFERENCES layers(id) ON DELETE CASCADE
              );
            '''

    DB_RELATION_INSERT_QUERY = '''INSERT INTO model_layer_relation  (id_model, id_layer, layer_index)
           VALUES (?, ?, ?);'''
    
    # TODO creare i layers da lista oggetti
    def __init__(self, lay_obj, input_shape, name, id='Not_Posted', push:bool = True, **kwargs):
        super(CustomDQNModel, self).__init__(name=name, **kwargs)
        self.name = name
        self.id = id
        self.window_size = input_shape
        self.lay_obj = lay_obj
        self.model_layers = []
        self.layers_id = []
        self.set_up_layers(self.lay_obj)
        self.push = push

    def build_layers(self, notes='No Notes'):
        try:
            layer_counter = 0
            for i in self.lay_obj:
                if i.type == layers_type.INPUT:
                    i.layer['params']['input_shape'][0] = self.window_size
        except ValueError as e :
            raise ValueError(f'Errore nella sovrascrittura della forma : {e}')

        for layer_config in self.lay_obj:
            layer_counter += 1
            layer_type = layer_config.layer['type']
            unique_name = f'{layer_type}_{layer_counter}'
            config = {key: value for key, value in layer_config.layer['params'].items() if key != 'type'}
            config['name'] = unique_name
            try:
                layer = getattr(tf.keras.layers,layer_type)(**config)
                self.model_layers.append(layer)
            except ValueError as e:
                raise ValueError(f'Layer non Instanziato {e}')

        if self.push:
            self.push_on_db(notes='Model_Compile')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)

        return x

    def set_up_layers(self, list_of_layers):
        self.layers_id = []

        #TODO: ho escluso questo metodo quando viene creata la rete target perche... non ce la fa 
        try:
            for index, i in enumerate(list_of_layers):
                if isinstance(i.id,str) and self.push:
                    i.push_layer()
                    record = dbm.retrive_last('layers', '*')
                    obj = Layers.convert_db_response(record)
                    list_of_layers[index] = obj

                self.layers_id.append(i.id)
        except :
            pass
        
        #region metodi necessari per la serializazzione e deserializzazione della classe custom
    def get_config(self):
        config = super(CustomDQNModel, self).get_config()
        config.update({
            'input_shape': self.window_size,
            'name': self.name,
            'id': self.id,
            'lay_obj': [(layer.get_config(), layer.__class__.__name__) for layer in self.model_layers],
            'push': self.push
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        lay_obj_config = config.pop('lay_obj')
        lay_obj = [getattr(tf.keras.layers, layer_name)(**layer_config) for layer_config, layer_name in lay_obj_config]
        
        return cls(lay_obj=lay_obj, **config)
        #endregion

    #region METODI DI SERIALIZZAZIONE #######################
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

    #endregion

    #region Metodi Ausiliari
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

    def push_on_db(self, notes='No Notes'):
        serializzati = self.serialize_to_json()[0]

        # Recupero le tulpe per l insereimento dei layer
        obj_tulpe_list = [(self.serialize_to_json()[0], self.name, notes)]
        # HACK: attenzione il nome e il parametro di controllo dei duplicati, impedisce i push
        dbm.push(obj_tulpe_list, self.DB_SCHEMA, self.INSERT_QUERY, 'name', 1, 'models')

        last_id = dbm.retrive_last('models', 'id')[0]
        self.id = last_id
        
        ids_tulpe = []
        for index, id in enumerate(self.layers_id):
            ids_tulpe.append((last_id, id, index))

        dbm.push(ids_tulpe, self.DB_RELATION_SCHEMA, self.DB_RELATION_INSERT_QUERY)

    #def extract_standard_keras_model(self, shape):
    #    """Estrae e restituisce un modello standard Keras dai layer di questo modello personalizzato."""
    #    try:
    #        # Crea un nuovo input layer che corrisponde all'input del modello originale
    #        new_input =  k.Input(shape=shape.shape)

    #        # Collega l'input ai layer del modello personalizzato
    #        x = new_input
    #        for layer in self.model_layers:
    #            x = layer(x)

    #        # Crea un nuovo modello standard Keras
    #        standard_model = k.Model(inputs=new_input, outputs=x)
            
    #        return standard_model
    #    except Exception as e:
    #        print(f'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ {e}')

    def extract_standard_keras_model(self, shape):
        """Extracts and returns a standard Keras model from this custom model's layers."""
        try:
            # Assuming 'self.model_layers' is a list of the layers you want to include
            if not self.model_layers:
                raise ValueError("Model layers are undefined or empty")

            # Create a new input layer that matches the input shape of the original model
            # Assuming the first layer of your model is configured correctly
            #input_shape = self.model_layers[0].input_shape[1:]  # Exclude the batch size dimension
            new_input = k.Input(shape=shape.shape)

            # Connect the input to the first layer and then to each subsequent layer
            x = new_input
            for layer in self.model_layers:
                x = layer(x)

            # Create a new standard Keras model
            standard_model = k.Model(inputs=new_input, outputs=x)
            print(f"WWWWWWWWWWWWWWWWWWWWWWWWWWWWW TYPE {type(standard_model)}")

            return standard_model
        except Exception as e:
            print(f"WWWWWWWWWWWWWWWWWWWWWWWWWWWWWError creating standard Keras model: {str(e)}")
            return None
        
    #endregion

