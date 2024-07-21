from abc import abstractmethod
from enum import Enum
from uu import Error
from Services import db_Manager as dbm
from Models.baseModelsClass import BaseModelsClass as BCM

#TODO: manca la documentazione

class process_type(Enum):

    BATCH = 'batch'
    SERIE = 'serie'
    STEP = 'step'

class ProcessOptimizer(Enum):
    ADAM = 'Adam'
    SGD = 'SGD'
    RMSPROP = 'RMSprop'
    ADAGRAD = 'Adagrad'
    ADADELTA = 'Adadelta'
    NADAM = 'Nadam'
    FTRL = 'Ftrl'
    ADAMAX = 'Adamax'

class ProcessLossFunction(Enum):
    MEAN_SQUARED_ERROR = 'mean_squared_error'
    BINARY_CROSSENTROPY = 'binary_crossentropy'
    CATEGORICAL_CROSSENTROPY = 'categorical_crossentropy'
    SPARSE_CATEGORICAL_CROSSENTROPY = 'sparse_categorical_crossentropy'
    MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'
    HINGE = 'hinge'
    HUBER = 'huber'
    LOGCOSH = 'logcosh'
    KULLBACK_LEIBLER_DIVERGENCE = 'kullback_leibler_divergence'

class Process(BCM):

    DB_SCHEMA = '''CREATE TABLE IF NOT EXISTS processes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                description TEXT,
                epsilon_start REAL,
                epsilon_end REAL,
                epsilon_reduce REAL,
                gamma REAL,
                tau REAL,
                learning_rate REAL,
                optimizer TEXT,
                loss TEXT,
                n_episode INTEGER,
                epochs INTEGER,
                type TEXT,
                windows_size REAL,
                fees REAL,
                initialBalance REAL,
                batch_size REAL
          );
          '''

    INSERT_QUERY = '''INSERT INTO processes (name, description, epsilon_start, epsilon_end, epsilon_reduce, gamma, tau, learning_rate, optimizer, loss, n_episode, epochs, type, windows_size, fees, initialBalance,batch_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'''

    def __init__(self, name, episodi=1, epoche=5, notes='No notes', epsilon_start=1.0, epsilon_end=0.01, epsilon_reduce=0.995, gamma=0.95, tau=0.125, learning_rate=0.001, optimizer:ProcessOptimizer=ProcessOptimizer.ADAM, 
        loss_functions:ProcessLossFunction=ProcessLossFunction.MEAN_SQUARED_ERROR, epochs=1000, type_=process_type.SERIE, _id='not posted yet', window_size=20, fees=0.01, initial_balance=100000, batch_size = 30):

        self.name = name
        self.description = notes
        self.epochs = epoche
        self.epsilo_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_reduce = epsilon_reduce
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss_functions 
        self.n_episode = episodi
        self.type = type_
        self.id = _id
        self.window_size = window_size
        self.fees=fees
        self.initial_balance = initial_balance
        self.batch_size = batch_size


    def push_on_db(self):
        try:
            tulp = [(
                self.name, self.description, self.epsilo_start, self.epsilon_end, self.epsilon_reduce, self.gamma, self.tau, 
                self.learning_rate, self.optimizer.value, self.loss.value, self.n_episode, self.epochs, self.type.value, self.window_size, self.fees, self.initial_balance, self.batch_size
            )]
            
            dbm.push(tulp, self.DB_SCHEMA, self.INSERT_QUERY, 'name', 1, 'processes')

        except ValueError as e:
            print(e)
        

    @staticmethod
    def convert_db_response(record,db_config):
        try:
            process = Process(name=record[1], notes=record[2], epsilon_start=record[3], epsilon_end=record[4], 
                epsilon_reduce=record[5], gamma=record[6], tau=record[7], learning_rate=record[8], 
                optimizer=record[9], loss_functions=record[10], episodi=record[11], epoche=record[12], 
                type_=record[13], _id=record[0], window_size=record[14], fees=record[15], initial_balance=record[16], batch_size=record[17])

            return process
        except Error as e:
            print(f"Errore durante la mappattura del record: {record}: {e}")

    @staticmethod
    def retrive_list_records_by_name(names:list[str]):
        return dbm.retive_a_list_of_recordos('name', 'processes', names)

    def print_attributo(self, nome_attributo):
        # Utilizza getattr per ottenere il valore dell'attributo dal suo nome
        if hasattr(self, nome_attributo):
            valore = getattr(self, nome_attributo)
            tipo = type(valore).__name__
            print(f'"nome atr:{nome_attributo} : valore: {valore} tipo:"{tipo}')
        else:
            print(f"L'attributo '{nome_attributo}' non esiste.")
        