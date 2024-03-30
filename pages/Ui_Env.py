import streamlit as st
from Models.Flex_Envoirment import EnvFlex as env
from Models.Reward_Function import Rewar_Function as Rw
import pandas as pd
import numpy as np
import CustomDQNModel as model
from Services import IchimokuDataRetriver as ichi, Db_Manager as db
from CustomDQNModel import CustomDQNModel as mo
from pages.Ui_Process import show_summary_in_sidebar
from Models.Process import Process as pr
from Models.Training_Model import Training_Model as training, Training_statu as status

#region Tentativo di iterazione definitivo
LOG_PATH = 'log fittizzia di base'
LOG_PATH = st.sidebar.text_input('Insert_your_log_path', value=LOG_PATH)
#region Recupero della logica 
st.title('QUI RICOSTRUIAMO UNA VERA ITERAZIONE')

if 'Obj_Function' not in st.session_state:
     objs = db.retrive_all('functions')

     lis = []
     lis_name = []
     
     for obj in objs:
         var = Rw.convert_db_response(obj)
         lis.append(var)
         lis_name.append(f'{var.id}- {var.name}')
     
     box = st.selectbox('Select Your Function', lis_name)
     if box:
         index = lis_name.index(box)
     
         if 'Obj_Function' not in st.session_state:
             st.session_state.Obj_Function = lis[index]
             st.balloons()
         else:
             st.session_state.Obj_Function = lis[index]
             st.balloons()
else :
     #region data
     if 'Data' not in st.session_state:
        st.subheader('Select_your_Data')
        set_dati = ichi.fetch_details() 
        lis_dati = set_dati['Id']
        
        sel = st.selectbox('Select your set', lis_dati)
        
        if sel:
            data = ichi.fetch_data_from_detailId(sel)
            st.session_state.Data = data
        
     if 'Data' in st.session_state:
         #TODO: vado a verificare solo la coerenza fra i nomi delle colonne e non il tipo di valori al loro interno
         colonne_df = set(st.session_state.Data.columns)
         chiavi_dict = set(st.session_state.Obj_Function.data_schema.keys())

         corrispondenza = colonne_df == chiavi_dict

         if corrispondenza == False:
            st.warning('Il set di dati non e compatibile con la funzione')
            if st.button('Tenta di forzare la corrispondenza'):
                new_dataframe = []
                for i in colonne_df:
                    if i in chiavi_dict:
                        new_dataframe.append(i)
                if chiavi_dict == set(new_dataframe):
                    new = st.session_state.Data[new_dataframe]
                    st.session_state.Data = new

                    colonne_df = set(st.session_state.Data.columns)
                    chiavi_dict = set(st.session_state.Obj_Function.data_schema.keys())

                    corrispondenza = colonne_df == chiavi_dict

         #region  Corrispondenza verificata Carica layers
         st.subheader('Load your layers')
         objs = db.retrive_all('layers')
         
         lis = []
         lis_name = []
         
         for obj in objs:
             var = model.Layers.convert_db_response(obj)
         
             lis.append(var)
             lis_name.append(f'{var.id}- {var.name}')
         
         box = st.multiselect('Select Your Layer', lis_name)
         if st.button('Save your selection'):
         
             indexes = [lis_name.index(sel)for sel in box]
         
             #selected = lis[indexes]
         
             if 'Layers' not in st.session_state:
                 st.session_state.Layers = []
                 for i in indexes:
                     st.session_state.Layers.append(lis[i])
             else:
                 st.session_state.Layers = [] #HACK: resetto la lista ogni volta
                 for i in indexes:
                     st.session_state.Layers.append(lis[i])

         if 'Layers' in st.session_state:
             lis_l = []
             for i in st.session_state.Layers:
                 lis_l.append(f'{i.name}: {i.type.value}')
             #lis_l = [l.name for l in st.session_state.Layers]
             inn = st.sidebar.selectbox('List_Of_Layers', lis_l)

             if st.sidebar.button('Clear layers list'):
                 st.session_state.pop('Layers')
                 st.experimental_rerun()

             try:
                 # HACK: sto passando solo i dizionari dei layers, perdendo cosi tutti i dettagli degli oggetti
                 list_layer = []
                 for i in st.session_state.Layers:
                     list_layer.append(i.layer)
                 modello = mo(list_layer,'Test_1')
                 st.session_state['Modello'] = modello

             except ValueError as e:
                 st.warning(e)

             if 'Modello' in st.session_state:
                 #region Processo
                 if 'Process' not in st.session_state:
                     st.subheader('Load your process')

                     processi = db.retrive_all('processes')
                     #HACK: visualizzazione grezza ma efficace
                     processo = st.selectbox('All processes', processi, help='id, name, description, epsilon_start, epsilon_end, epsilon_reduce, gamma, tau, learning_rate, optimizer, loss, n_episode, epochs, type, windows_size')
                     
                     if st.button('Save Selection'):
                         obj = pr.build_process_from_record(processo)
                     
                         if 'Process' not in st.session_state :
                             st.session_state.Process = obj
                         else:
                             st.session_state.Process = obj

                         st.experimental_rerun()
                 else:
                    show_summary_in_sidebar(st.session_state.Process)

                    try:
                        st.session_state.Modello.build_layers(st.session_state.Process.window_size)
                        st.success('Layers Compiled')
                    except ValueError as e:
                        st.warning(e)
                        raise ValueError(f'################## {e}')

                    if st.sidebar.button('Clear_Process_'):
                        st.session_state.pop('Process')

                    #region tento di salvare l iterazione sfruttando gli id dei layers
                    tra_name = st.text_input('insert_training_name')
                    if st.button('Pusch new training'):
                       train = training(tra_name,status.PLANNED,st.session_state.Obj_Function.id,st.session_state.Process.id, LOG_PATH)
                       l = []
                       for i in st.session_state.Layers:
                           l.append(i.id)
                       # TODO: il nome di questa ui dovra cambiare presto
                       train.pusch_on_db(l,'Pusch from Ui_Env')
                    #endregion

                 #endregion

                 if st.sidebar.button('Clear_Modello'):
                     st.session_state.pop('Modello')
         #endregion
         
         if st.sidebar.checkbox('Display_data'):
              st.write(st.session_state.Data.head(5))
              remover = st.multiselect('Remove Columns', st.session_state.Data.columns)
              remove = st.button('Save new D_Frame')
              if remove:
                  new_data = st.session_state.Data.drop(columns=remover)
                  st.session_state.Data = new_data
     
         if st.sidebar.button('Clear_Data'):
             st.session_state.pop('Data')
      #endregion
     
     if st.sidebar.button('Clear current logic'):
        st.session_state.pop('Obj_Function')

#endregion

    #region layers
    #endregion

#endregion
#if 'Modello' not in st.session_state or 'Schemas' not in st.session_state:
#    st.warning('Modello or Schema not in session')
#else:
#    x=str(st.session_state.Schemas[0].items())
#    dat_0 = pd.DataFrame([st.session_state.Schemas[0]])
#    dat_1 = pd.DataFrame([st.session_state.Schemas[1]])
#    dat_2 = pd.DataFrame([st.session_state.Schemas[2]])

#    st.write(st.session_state.Data)
#    st.write(dat_0)
#    st.write(dat_1)
#    st.write(dat_2)

#    #Fimnd_schemas non funziona

#    st.write(st.session_state.Modello.schema_data)
#    st.write(st.session_state.Modello.schema_input)

#    #TODO: diobocia e sbagliato l ordine delle azioni  e degli stati

#    actions = ['wait', 'buy', 'sell']
#    statusss = ['flat', 'long', 'short']


#    ambiente = env(st.session_state.Data,st.session_state.Premia,[],actions,statusss)

#    ambiente.step(1)
#    ambiente.step(1)
#    ambiente.step(1)
#    ambiente.step(2)
#    ambiente.step(0)
#    ambiente.step(1)
#    ambiente.step(2)
#    ambiente.step(1)
#    ambiente.step(2)
#    ambiente.step(1)
#    ambiente.step(2)
#    ambiente.step(0)
#    ambiente.step(0)
#    ambiente.step(1)

#    st.write(ambiente.Obseravtion_DataFrame)



