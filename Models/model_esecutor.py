from abc import abstractmethod
from csv import writer
from gc import callbacks
import sys
from tabnanny import verbose
import numpy as np
import tensorflow as tf
import time
from CustomDQNModel import CustomDQNModel as model
from Flex_Envoirment import EnvFlex as envoirment
from ReplayBuffer import ReplayBuffer
from Models import logs_classes as lgc
from datetime import datetime , timedelta
import math
import pandas as pd
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from matplotlib import animation
import time
import streamlit as st

###############  LEGENDA °°°°°°°°°°°°°°°°°
# tau aggiorna i pesi
# epsilon aggiorna esplorativa, esploitativa   VAlori fra 0 e 1
# lr va passato dal compilatore
# gamma e il fattore di sconto futuro

class Trainer():
    def __init__(self, epsilon_start, epsilon_end, epsilon_reduce, gamma, tau, base_logs_path:str, epoche=1, 
                 customcallback=0, replay_cap = 30000, esclude_from_logs=[]):
        self.replayer = ReplayBuffer(replay_cap)
        self.epsilon = epsilon_start
        self.epochs = epoche
        self.epsilo_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_reduce = epsilon_reduce
        self.data = None
        #self.train_data_ = 0
        #self.work_data_ = 0
        #self.test_data_ = 0
        self.env : envoirment
        self.main_network : model
        self.target_network : model
        # aggiunge callback
        self.CustomCallback = customcallback
        self.esclude_from_Logs = esclude_from_logs
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = 0
        self.optimizer =  [
                        'Adam',
                        'SGD',
                        'RMSprop',
                        'Adagrad',
                        'Adadelta',
                        'Nadam',
                        'Ftrl',
                        'Adamax']
        self.loss = loss_functions = [
                        'mean_squared_error',
                        'binary_crossentropy',
                        'categorical_crossentropy',
                        'sparse_categorical_crossentropy',
                        'mean_absolute_error',
                        'hinge',
                        'huber',
                        'logcosh',
                        'kullback_leibler_divergence'
                    ]
        self.ep_report = list()

        # Additional metrics
        self.episode = 0
        self.current_rew = 0

        # HACK tento l'aggiunta di un callback >>>>>>>>>> spostata alla fine della compilazione della rete per avere il nome del modello
        timestamp = time.time()
        self.date_str = time.strftime('%d_%H_%M_%S', time.localtime(timestamp))
        self.path_2 = f'{base_logs_path}/{self.date_str}/'
        self.writer = tf.summary.create_file_writer(self.path_2+'env_metrics')
        self.writer_tabulari = tf.summary.create_file_writer(self.path_2+'tabulari')
    
    #region MAIN METHODS ###############
     # Aggiornamento della Rete Target
    def Update_target_network(self):
        print(f'##################################      Update Target  ########################')
        main_weights = self.main_network.get_weights()
        target_weights = self.target_network.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = self.tau * main_weights[i] + (1 - self.tau) * target_weights[i]

        self.target_network.set_weights(target_weights)

    # Aggiornamento Rete Principale
    def Aggiornamento_Main(self, stati, azioni, ricompense, stati_successivi, terminati):

        print(f'##################################      Update Main  ########################')
        tensore_stati_successivi, ricompense, terminati = self.Estrai_Stati(stati_successivi, ricompense, terminati)
        stati_correnti, _, _ = self.Estrai_Stati(stati, ricompense, terminati)
    
        # TODO: verificare che q-target venga effettivamente scontato
        terminati_int = tf.cast(terminati, tf.float32)
        Q_target = ricompense + self.gamma * np.max(self.target_network.predict(tensore_stati_successivi), axis=1)*(1-terminati_int) 
    
        # NODO: Utilizzo del modello principale per ottenere le stime Q per le azioni in ogni stato del batch corrente.
        Q_stime = self.main_network.predict(tensore_stati_successivi, verbose=2)
    
        # NODO: Aggiornamento delle stime Q con i valori target Q per le azioni effettivamente intraprese.
        Q_stime[np.arange(len(Q_stime)), azioni] = Q_target

        writer_defoult = tf.keras.callbacks.TensorBoard(log_dir=self.path_2 + f'default{self.episode}', histogram_freq=1, update_freq='epoch'),

    
        if self.CustomCallback != 0:
            fitness = self.main_network.fit(x=stati_correnti, y=Q_stime, epochs=self.epochs, verbose=1, callbacks=self.CustomCallback)
        else:
            fitness = self.main_network.fit(x=stati_correnti, y=Q_stime, epochs=self.epochs, verbose=1, callbacks=writer_defoult)

        # Inizializzo gli hyperparametri 
        self.main_network.get_hyperparameter()

    # Using Policy Decision
    def epsylon_greedy_policy(self, state, model):
        #TODO: testare la coerenza dei vari training mode
        n_action = self.env.coutnaction()

        if np.random.rand() < self.epsilon:
            print(f'##################################      Randomly Selected ########################')
            x = np.random.randint(n_action)
            azione_one_hot = np.zeros(n_action)
            azione_one_hot[x] = 1
            return azione_one_hot
        else:
            Q_values = model.predict(state, verbose=2)
            x = np.argmax(Q_values[0])
            azione_one_hot = np.zeros(n_action)
            azione_one_hot[x] = 1
            # HACK : Sembra che predict funzioni
            return azione_one_hot

    def Train(self, n_episodi, mode, batch_size, logpath=0):
       # registro i log per episodio
       self.ep_report.clear()

       if mode != 'batch' and mode != 'step' and mode != 'serie':
           raise ValueError('Train mode : batch |  step | serie')

       if mode == 'serie':
           # TODO: Verificare che avvenga l'addestramento e cha i dati siano giuti
           batch_size = len(self.env.data)
           # tento di stabilizzare gli step di riduzione di epsilon
           if n_episodi > 1:
                self.epsilon_reduce = n_episodi-1
           else:
               self.epsilon_reduce = n_episodi

       for episodio in range(n_episodi):

           self.episode = episodio
           
           batch_count = 0

           # TODO: verificare se lo stato viene effettivamente sovrascritto
           stato = self.env.reset() # Reset

           # ottengo il tensore dello stato
           stato = self.estrapola_tensore(stato[0], stato[1], stato[2])

           # HINT: Libreria Math per operazioni fra tensori
           while not tf.math.equal(stato[2] , True):

               # Debug
               done = tf.math.equal(stato[2] , True)
               print(f'##################################    done : {done}  ########################')
               print(f'##################################    current_step : {self.env.current_step}  ########################')
               print(f'##################################    corrent_balance : {self.env.current_balance}  ########################')
               print(f'##################################    reward : {stato[1]}  ########################')


               # Aggiungo una dimensione al tensore
               tens = tf.expand_dims(stato[0], axis=0)
               
               # Selezione Azione
               self.epsilon = self.reduce_epsilon()
               azione = self.epsylon_greedy_policy(state=tens, model=self.main_network) 

               # TODO: verificala!!!
               azione = np.argmax(azione)

               # Esecuzione Azione / Ossevazione
               nuovo_stato, ricompensa, done, _ = self.env.step(azione) 
               self.current_rew += ricompensa
               print(f'##################################     azione: {azione}  ########################')


               # Estraggo E Aggiungo Una dimensione al nuvo stato
               nuovo_stato = self.estrapola_tensore(nuovo_stato, ricompensa, done)
               nuovo_stato_tens = tf.expand_dims(nuovo_stato[0], axis=0)

               #seguendo quanto sopra pongo la ricompensa e done pari all ultima ricompensa ottenuta
               ricompensa = nuovo_stato[1]
               done = nuovo_stato[2]

               # Memorizzazione
               if mode != 'step':
                    self.replayer.push(state=tens, action=azione, reward=ricompensa, next_state=nuovo_stato, done=done )
                    
                    ## TODO: qua scelgo quando aggiornare main Campionamento dei batch
                    if  tf.math.equal(done , True):
                        batch = self.campionamento(batch_size-self.env.window_size)

                        # Aggiornamento rete principale 
                        self.Aggiornamento_Main(*batch)

                    # Aggiorno la target in base al  numero dei batch
                    if mode == 'batch':

                        if batch_count % batch_size == 0:
                            self.Update_target_network()

               if mode == 'step':

                   self.Aggiornamento_Main(state=tens, action=azione, reward=ricompensa, nex_state=nuovo_stato, done=done)

                   if self.env.current_step % batch_size == 0:
                        self.Update_target_network()

               stato = nuovo_stato

           # HINT: posto i log per tensorboard
           self.post_logs(self.writer, self.env.Obseravtion_DataFrame, f'{episodio}')

           # Aggiorno la target in base al  numero dei batch
           if mode == 'serie':

               self.current_rew = 0

               if episodio % 1 == 0:
                   self.Update_target_network()

               self.replayer.clear()

           # HINT: Salvo il il df per l episodio
           self.ep_report.append(self.env.Obseravtion_DataFrame)

       # Creo la log tab
       self.build_log_tab(self.writer_tabulari,mode)

       # HINT: Creo il custom log per il db
       _start_data = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

       if logpath != -1:
            self.logg_all(batch_size, n_episodi, _start_data, logpath)
    #endregion

    #region METODI AUSILIARI ###############
    def set_data(self, train_data, work_data, test_data=0, decrese_data = 0):
       #verifico la coerenza dei parametri
       for param in (train_data, work_data, test_data, decrese_data):
           if not 0.0000 <= param <= 0.999:
               raise ValueError(f"Il parametro {param} deve essere compreso tra 0.001 e 0.999")

       lunghezza_dataset = len(self.data)

       if decrese_data != 0:
           lenght = int(lunghezza_dataset*decrese_data)
           self.data = self.data[:lenght]

       if train_data+work_data+test_data > 1:
           lunghezza_dataset = len(self.data)
           trainlen = int(lunghezza_dataset*train_data)
           work_data = int(lunghezza_dataset*work_data)
           self.train_data_ = self.data[:trainlen]
           self.work_data_ = self.data[trainlen:trainlen+work_data]
           self.test_data_ = self.data[trainlen+work_data:]
       else:
           lunghezza_dataset = len(self.data)
           trainlen = int(lunghezza_dataset*train_data)
           work_data = int(lunghezza_dataset*work_data)
           test_data = int(lunghezza_dataset*test_data)
           self.train_data_ = self.data[:trainlen]
           self.work_data_ = self.data[trainlen:trainlen+work_data]
           self.test_data_ = self.data[trainlen+work_data:trainlen+work_data+test_data]

    def compile_networks(self, optimaizer_, loss_, metrics):
        # HACK: salvo opt / loss per le metriche 
        self.optimizer = optimaizer_
        self.loss = loss_

        if hasattr(self.optimizer, 'lr'):
            self.learning_rate = self.optimizer.lr
        elif hasattr(self.optimizer, 'learning_rate'):
            self.learning_rate = self.optimizer.learning_rate

        try:
            n_variabili = self.env.coutobservationVar()
            n_output = self.env.coutnaction()
            time_steps = self.env.window_size
            n_features = self.env.coutobservationVar()
            self.main_network = model(time_steps, n_features,n_output)

            self.main_network.compile(optimizer=optimaizer_, loss=loss_, metrics=metrics)

            self.target_network = tf.keras.models.clone_model(self.main_network)
            self.target_network.set_weights(self.main_network.get_weights())

            #self.path_2 = f'logs/{self.main_network.name_model}/{self.date_str}/'
            #self.writer = tf.summary.create_file_writer(self.path_2+'env_metrics')

        except :
            raise ValueError('modelli non compilati, verificare che l ambinte sia inizializzato')

    # costruzione dell ambiente
    def build_env(self, data, encode_actions=['wait','buy','sell'], encode_position_status=['flat','long','short'],
                 windows_size=20, fees=0.01, initial_balance = 100000, 
                first_reword=0,use_additional_reward_colum:bool = False, custom_env=None, alternative_logic=0):
       if isinstance(custom_env, (envoirment)):
           self.env = custom_env
       else:
           if alternative_logic == 0:
               logic = self.premia
           else:
               logic = alternative_logic

           self.env = envoirment(data,logic, action_spaces_tab=encode_actions, position_space_tab=encode_position_status,
                           windows_size=windows_size, fees=fees, initial_balance=initial_balance, first_reword=first_reword,
                           use_additional_reward_colum=use_additional_reward_colum,reward_colums=[])

    # Campionamento del Batch
    def campionamento(self, batch_size):
        batch = self.replayer.sample(batch_size)
        stati, azioni, ricompense, stati_successivi, terminati = zip(*batch)
        return stati, azioni, ricompense, stati_successivi, terminati

    def Estrai_Stati(self, tulpa_di_stati_successivi, ricompense, terminati):
        # estraggo tutti i tensori dall indice zero di ogni elemento della lista
        tensori = [t[0] for t in tulpa_di_stati_successivi]

        # Li combino in un unico tensore
        tensore = tf.stack(tensori)
        ricompense = tf.stack(ricompense)
        terminati = tf.stack(terminati)

        return tensore , ricompense, terminati

    # TODO: Richiede una normalizzazione nell estrazione dei dati
    def estrapola_tensore(self, stato : np.array , ricompensa, terminato):
        posizioni = np.array(stato)

        ricompensa_t = tf.convert_to_tensor(ricompensa, dtype=tf.float32)

        terminato_t = tf.convert_to_tensor(terminato, dtype=tf.bool)

        stato_t = tf.convert_to_tensor(posizioni)

        return stato_t , ricompensa_t , terminato_t

    # Verificare la funzione di riduzione di epsilon
    def reduce_epsilon(self):
        decay_rate = (self.epsilon_end - self.epsilon_end) / self.epsilon_reduce

        val = max(self.epsilon - decay_rate , self.epsilon_end)
        self.epsilon = val
        return val

    #######################   LOG E SALVATAGGI   ###############
    # TODO: categorizzare meglio i modelli salvati
    def save(self, destination = 0, tipo = 0):

        if (destination == 0) and (tipo == 0):
            self.main_network.save(f'Models/{self.main_network.name}_{self.date_str}.h5')
        if tipo == 0:
            self.main_network.save(destination)
        else:
            self.main_network.save(destination,tipo)

    def test_existing_model(self, path, env=0):

        network = model = tf.keras.models.load_model(path)

        if env == 0:
            env = self.env

        stato = env.reset()

        stato = self.estrapola_tensore(stato[0], stato[1], stato[2])
        
        
        while not tf.math.equal(stato[2] , True):
        
            tens = tf.expand_dims(stato[0], axis=0)
            Q_values = self.main_network.predict(tens, verbose=0)
            x = np.argmax(Q_values[0])
            azione_one_hot = np.zeros(len(env.action_space_tab))
            azione_one_hot[x] = 1
            azione = np.argmax(azione_one_hot)
            stato_successivo, ricompensa, finito, _ = env.step(azione)
        
            # Estraggo E Aggiungo Una dimensione al nuvo stato
            nuovo_stato = self.estrapola_tensore(stato_successivo, ricompensa, finito)
            nuovo_stato_tens = tf.expand_dims(nuovo_stato[0], axis=0)
        
            #seguendo quanto sopra pongo la ricompensa e done pari all ultima ricompensa ottenuta
            ricompensa = nuovo_stato[1]
            done = nuovo_stato[2]
            stato = nuovo_stato
            print(f'##### current balance : {env.current_balance}')
            print(f'##### azione : {azione}')

        return env.Obseravtion_DataFrame
    #endregion

    #region Logs
    ### Custom Tensorboard _ log
    def post_logs(self, callback, list_of_list_of_value, iteration_index):
        with callback.as_default():
            for coll_name in list_of_list_of_value.columns:
                if coll_name in self.esclude_from_Logs:
                    continue
                for step, val in enumerate(list_of_list_of_value[coll_name]):
                    if isinstance(val, str):
                        if val == 'wait' or val == 'flat':
                            val = 0
                        elif val == 'long' or val == 'buy':
                            val = 1
                        elif val == 'short' or val == 'sell':
                            val = 2
                        else:
                            val = 200
                    tf.summary.scalar(name=coll_name+iteration_index, data=val, step=step)
            tf.summary.flush()

    def build_log_tab(self, callback, mode):

        with callback.as_default():
           table = "| Parametro           | Valore       |\n|---------------------|--------------|\n"
           table += f"| Model Name          | {self.main_network.name_model} |\n"
           table += f"| Model description   | {self.main_network.description} |\n"
           table += f"| Epsilon Inizio      | {self.epsilo_start} |\n"
           table += f"| Epsilon Fine        | {self.epsilon_end}   |\n"
           table += f"| Decadimento Epsilon | {self.epsilon_reduce} |\n"
           table += f"| Gamma               | {self.gamma}         |\n"
           table += f"| Tau                 | {self.tau}           |\n"
           table += f"| Ottimizzatore       | {self.optimizer}     |\n"
           table += f"| Learning Rate       | {self.learning_rate} |\n"
           table += f"| Funzione Loss       | {self.loss} |\n"
           table += f"| Metodo Iterazione   | {mode} |\n"
           
           tf.summary.text('Parametri', table, step=0)

        if hasattr(self.main_network, 'hyperparameters'):
            with callback.as_default():
                table = "| Parametro           | Valore       |\n|---------------------|--------------|\n"
                for key, value in self.main_network.hyperparameters.items():
                    table += f'| {key} | {value} |\n'

                tf.summary.text('Hyper-Parametri', table, step=0)

    def reset_callback_dir(self):
        # Tento la scrittura di call back organizzati
        # Sovrascrivi il percorso di log per ogni callback nel dizionario
        if self.CustomCallback != 0:
            for key in self.CustomCallback.keys():
                if hasattr(self.CustomCallback[key], 'log_dir'):
                    self.CustomCallback[key].log_dir = self.path_2 + key
    
    # Tento la costruzione del logger personalizzato
    def logg_all(self, batc_size, epoche, start_data, db_path=0):
        if db_path != 0:
            lgc.change_db_path(db_path)

        _end_data = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Per calcolare e formattare _time_span
        _time_span_seconds = int(time.mktime(time.strptime(_end_data, "%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime(start_data, "%Y-%m-%d %H:%M:%S")))
        _time_span_formatted = str(timedelta(seconds=_time_span_seconds)) 
        _process = lgc.AddestramentoRNN(batc_size, epoche,self.tau,note='none')
        _dati = lgc.Dati(str(self.env.data), 'Nessuno')
        _env = lgc.Ambiente(str(self.env.action_space_tab),str(self.env.observation_space),str(self.flex_buy_andSell), note='none')
        _layers = self.main_network.get_custom_logs()
        _model = lgc.Model(_layers,self.learning_rate,self.loss,self.optimizer,batc_size,epoche)

        # TODO: dati hard coded _ dict not supported, may serialize it
        _test_log = lgc.Test_Log(start_data,_end_data,_time_span_formatted,0000,1000,_model,
                                 _env,_dati,_process,_layers)

        _test_log.Push()

    #endregion

    
    
   

