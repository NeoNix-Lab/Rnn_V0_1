from abc import abstractmethod
from csv import writer
from gc import callbacks
import sys
from tabnanny import verbose
import numpy as np
import tensorflow as tf
import time
from CustomDQNModel import CustomDQNModel as model
from Models.Flex_Envoirment import EnvFlex as envoirment
from Models.ReplayBuffer import ReplayBuffer
#from Models import logs_classes as lgc
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
    def __init__(self, env:envoirment, network:model,  epsilon_start, epsilon_end, epsilon_reduce, gamma, tau, training_name:str, epoche=1, replay_cap = 30000):
        self.replayer = ReplayBuffer(replay_cap)
        self.epsilon = epsilon_start
        self.epochs = epoche
        self.epsilo_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_reduce = epsilon_reduce
        self.env = env
        self.main_network = network
        self.target_network = None
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
        self.action_report = None
        self.action_report_for_episode = []
        self.action_count = 0

        # Additional metrics
        self.episode = 0
        self.current_rew = 0

        # HACK tento l'aggiunta di un callback >>>>>>>>>> spostata alla fine della compilazione della rete per avere il nome del modello
        timestamp = time.time()
        self.date_str = time.strftime('%d_%H_%M_%S', time.localtime(timestamp))
        self.path = f'logs/{training_name}'
        self.path_2 = f'{self.path}/{self.date_str}/'
        self.writer = tf.summary.create_file_writer(self.path_2+'env_metrics')
        self.writer_tabulari = tf.summary.create_file_writer(self.path_2+'tabulari')
    
    #region MAIN METHODS ###############
    def Update_target_network(self):
        print(f'##################################      Update Target Weight  (soft_updating) ########################')
        main_weights = self.main_network.get_weights()
        target_weights = self.target_network.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = self.tau * main_weights[i] + (1 - self.tau) * target_weights[i]

        self.target_network.set_weights(target_weights)

    # Aggiornamento Rete Principale
    def Aggiornamento_Main(self, stati, azioni, ricompense, stati_successivi, terminati):
        #region reshape
        #HACK: reshape di ricompense da (79,) a (79,16)
        #ricompense = np.expand_dims(ricompense, axis=1)
        terminati_int = tf.cast(terminati, tf.float32)
        term = 1-terminati_int
        #matching tens shape
        _term = np.expand_dims(term, axis=1)
        #endregion

        #TODO: Verifica d aver scelto l asse giusto
        # NODO 1 : HINT: otterngo tutti i q per tutti gli stati successivi shape (size,n_action)
        prediction = self.target_network.predict(stati_successivi) # (batch_size, n_action)

        # NODO 2 : HINT: otterngo tutti i q relativi alle azioni realmente intraprese (size,)
        selected_max_prediction = prediction[np.arange(prediction.shape[0]), azioni] #(batch_size)

        print(f'XXXXXXXXXXXXXXXXXXXXXXX Update Main  ########################')
        print(f'XXXXXXXXXXXXXXXXXXXXXXX Model_summary {self.target_network.summary()}########################')
        print("XXXXXXXXXXXXXXXXXXXXXXX Shape of predictions:", prediction.shape)
        print("XXXXXXXXXXXXXXXXXXXXXXX Shape of terminati_int:", _term.shape) #(1,1)
        print("XXXXXXXXXXXXXXXXXXXXXXX Shape of terminati_flatten:", _term.flatten().shape) #(1)
        print("XXXXXXXXXXXXXXXXXXXXXXX Shape of azioni:", azioni.shape) #(batch_size, 1)
        print("XXXXXXXXXXXXXXXXXXXXXXX Shape of selected_max_prediction:", selected_max_prediction.shape) #(batch_size,1)
        print("XXXXXXXXXXXXXXXXXXXXXXX Shape of ricompense:", ricompense.shape) #(batch_size,1)

        # TODO: verificare che q-target venga effettivamente scontato
        # NODO 3 : HINT: sconto i q delle azioni intraprese per poterli porre come target d apprendimento(size,)
        # ricompense (batch_size,1) ; (real); (batch_size,1); (1)
        Q_target = ricompense + (self.gamma * selected_max_prediction* _term.flatten())
    
        # NODO: Utilizzo del modello principale per ottenere le stime Q per le azioni in ogni stato del batch corrente.
        # TODO: verificare
        Q_stime = self.main_network.predict(stati_successivi, verbose=2)
        print("XXXXXXXXXXXXXXXXXXXXXXX Shape of main_network predictions:", Q_stime.shape)


        print("XXXXXXXXXXXXXXXXXXXXXXX Q_target:", Q_target.shape)
        print("XXXXXXXXXXXXXXXXXXXXXXX Q_stime:", Q_stime.shape)

    
        # NODO: Aggiornamento delle stime Q con i valori target Q per le azioni effettivamente intraprese.
        Q_stime[np.arange(len(Q_stime)), azioni] = Q_target

        ##HINT: callback
        writer_defoult = tf.keras.callbacks.TensorBoard(log_dir=self.path_2 + f'default{self.episode}', histogram_freq=1, update_freq='epoch'),

        # TODO: ridefinire y=Q_stime
        fitness = self.main_network.fit(x=stati, y=Q_stime, epochs=self.epochs, verbose=1, callbacks=writer_defoult)
        print(f'XXXXXXXXXXXXXXXXXXXXXXX Main_Model_summary {self.main_network.summary()}########################')


        self.Update_target_network()

        # Inizializzo gli hyperparametri 
        #self.main_network.get_hyperparameter()

    # Using Policy Decision
    def epsylon_greedy_policy(self, state, model):
        n_action = self.env.coutnaction()
        self.action_count+=1
        self.action_report['selection'][self.env.current_step] = self.epsilon 

        if np.random.rand() < self.epsilon:
            print(f'################################## Action Randomly Selected ########################')
            self.action_report['selection'][self.env.current_step] = 'random'
            

            x = np.random.randint(n_action)
            azione_one_hot = np.zeros(n_action)
            azione_one_hot[x] = 1
            return azione_one_hot
        else:
            print(f'################################## Action Model Selected ########################')
            self.action_report['selection'][self.env.current_step] = 'model'

            Q_values = model.predict(state, verbose=2)
            x = np.argmax(Q_values[0])
            azione_one_hot = np.zeros(n_action)
            azione_one_hot[x] = 1

            self.action_report['action'][self.env.current_step] = np.argmax(azione_one_hot)
            return azione_one_hot

    def Train(self, n_episodi, mode, batch_size):
       self.ep_report.clear()

       if mode != 'batch' and mode != 'step' and mode != 'serie':
           raise ValueError('Train mode : batch |  step | serie')

       if mode == 'serie':
           #HACK: qui sto riducendo epsilon del numero di episodi, sei sicuro??
           #HACK: pongo la dimensione del batch == alla lunghezza della serie
           batch_size = len(self.env.data)-(self.env.window_size+1)
           if n_episodi > 1:
                self.epsilon_reduce = n_episodi-1
           else:
               self.epsilon_reduce = n_episodi

       for episodio in range(n_episodi):
           print(f'##################################    New Episode: {episodio}  ########################')
           self.action_report = pd.DataFrame(data=np.zeros((len(self.env.data),3)),columns=['action','selection','epsilon'])

           self.episode = episodio
           #HACK: non sto mai pulendo la queque
           # TODO: verificare se lo stato viene effettivamente sovrascritto
           stato = self.env.reset() # Reset

           # ottengo il tensore dello stato
           stato = self.estrapola_tensore(stato[0], stato[1], stato[2])
           _done = stato[2]

           # HINT: Libreria Math per operazioni fra tensori
           while not tf.math.equal(_done, True):

               # Debug
               done = tf.math.equal(stato[2] , True)
               print(f'##################################    done : {done}  ########################')
               print(f'##################################    current_step : {self.env.current_step}/{len(self.env.data)}  ########################')
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
               stato_t , ricompensa_t , terminato_t = self.estrapola_tensore(nuovo_stato, ricompensa, done)
               #TODO: verifica della struttura e del metodo di campionamento e selezione del campionamento
               self.replayer.push(state=stato[0], action=azione, reward=ricompensa_t, next_state=stato_t, done=terminato_t )

               # Memorizzazione
               if mode == 'step':
                    coda = len(self.replayer.buffer) 

                    if coda >= batch_size:
                        batch = self.campionamento(batch_size)
                        self.Aggiornamento_Main(*batch)

               if mode != 'step':
                   coda = len(self.replayer.buffer) 
                   print(f'coda : {coda}')
                   print(f'batch_size : {batch_size}')
 
                   if coda >= batch_size:
                        batch = self.campionamento(batch_size)
                        self.Aggiornamento_Main(*batch)
               
               _done = terminato_t

           self.action_report_for_episode.append(self.action_report)
          
           self.ep_report.append(self.env.Obseravtion_DataFrame)
       
       self.main_network.save(f'{self.path_2}/Modello.h5')
    #endregion

    #region Utils
    def compile_networks(self, optimaizer_, loss_, metrics):
        # HACK: salvo opt / loss per le metriche 
        self.optimizer = optimaizer_
        self.loss = loss_

        try:

            self.main_network.compile(optimizer=optimaizer_, loss=loss_, metrics=metrics)

            stato = self.env.reset()
            stato = self.estrapola_tensore(stato[0], stato[1], stato[2])
            tens = stato[0]

            self.main_network = self.main_network.extract_standard_keras_model(shape=tens)

            self.main_network.compile(optimizer=optimaizer_, loss=loss_, metrics=metrics)

            #TODO: per ora verificare la congruenza dei pesi
            # TODO: ho dovuto rimandare la crazione della target ad una seconda istanza della main
            self.target_network = tf.keras.models.clone_model(self.main_network)
            self.target_network.set_weights(self.main_network.get_weights())
            self.target_network.compile(optimizer=optimaizer_, loss=loss_, metrics=metrics)

        except Exception as e:
            print(f'Model Complie Error {str(e)}')

    # Campionamento del Batch
    def campionamento(self, batch_size):
        stati, azioni, ricompense, stati_successivi, terminati = self.replayer.sample(batch_size)#zip(*batch)
        return stati, azioni, ricompense, stati_successivi, terminati

    # TODO: Richiede una normalizzazione nell estrazione dei dati
    def estrapola_tensore(self, stato : np.array , ricompensa, terminato):
        posizioni = np.array(stato)

        ricompensa_t = tf.convert_to_tensor(ricompensa, dtype=tf.float32)

        terminato_t = tf.convert_to_tensor(terminato, dtype=tf.bool)

        stato_t = tf.convert_to_tensor(posizioni)

        return stato_t , ricompensa_t , terminato_t

    # Verificare la funzione di riduzione di epsilon
    def reduce_epsilon(self):
        decay_rate = (self.epsilo_start - self.epsilon_end) / self.epsilon_reduce

        val = max(self.epsilon - decay_rate , self.epsilon_end)
        self.epsilon = val
        return val

    #######################   LOG E SALVATAGGI   ###############
    # TODO: categorizzare meglio i modelli salvati
    def save(self, destination = 0, tipo = 0):

        if (destination == 0) and (tipo == 0):
            self.main_network.save(f'Modelli/{self.main_network.name}_{self.date_str}.h5')
        if tipo == 0:
            self.main_network.save(destination)
        else:
            self.main_network.save(destination,tipo)

    def reset_callback_dir(self):
        # Tento la scrittura di call back organizzati
        # Sovrascrivi il percorso di log per ogni callback nel dizionario
        if self.CustomCallback != 0:
            for key in self.CustomCallback.keys():
                if hasattr(self.CustomCallback[key], 'log_dir'):
                    self.CustomCallback[key].log_dir = self.path_2 + key
    
    def test_existing_model(self, path, data, env):

        network = tf.keras.models.load_model(path)

        env.data = data

        stato = env.reset()

        stato = self.estrapola_tensore(stato[0], stato[1], stato[2])
        
        
        while not tf.math.equal(stato[2] , True):
        
            tens = tf.expand_dims(stato[0], axis=0)
            Q_values = network.predict(tens, verbose=0)
            x = np.argmax(Q_values[0])
            #azione_one_hot = np.zeros(len(env.action_space_tab))
            #azione_one_hot[x] = 1
            azione = np.argmax(x)
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

    def ispeziona_layers(self, modello):
        for i, layer in enumerate(modello.layers):
            print(f"Layer {i} | Name: {layer.name} | Type: {type(layer).__name__} | Output Shape: {layer}")
            print(f"Config: {layer.get_config()}")
            print("--------------------------------------------------")
#endregion

#region logs
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

     ## Tento la costruzione del logger personalizzato
    #def logg_all(self, batc_size, epoche, start_data, db_path=0):

    #    if db_path != 0:
    #        lgc.change_db_path(db_path)

    #    _end_data = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #    # Per calcolare e formattare _time_span
    #    _time_span_seconds = int(time.mktime(time.strptime(_end_data, "%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime(start_data, "%Y-%m-%d %H:%M:%S")))
    #    _time_span_formatted = str(timedelta(seconds=_time_span_seconds)) 
    #    _process = lgc.AddestramentoRNN(batc_size, epoche,self.tau,note='none')
    #    _dati = lgc.Dati(str(self.env.data), 'Nessuno')
    #    _env = lgc.Ambiente(str(self.env.action_space_tab),str(self.env.observation_space),str(self.flex_buy_andSell), note='none')
    #    _layers = self.main_network.get_custom_logs()
    #    _model = lgc.Model(_layers,self.learning_rate,self.loss,self.optimizer,batc_size,epoche)

    #    # TODO: dati hard coded _ dict not supported, may serialize it
    #    _test_log = lgc.Test_Log(start_data,_end_data,_time_span_formatted,0000,1000,_model,
    #                             _env,_dati,_process,_layers)

    #    _test_log.Push()
    #endregion
   

