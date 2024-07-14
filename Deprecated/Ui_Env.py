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
import time
from Services import st_utils as utils

# TODO: il nome di questa ui dovra cambiare presto
# TODO: i db log non raggiungono streamlit
#region Tentativo di iterazione definitivo
LOG_PATH = 'log fittizzia di base'
st.title('Iteration')
LOG_PATH = st.text_input('Insert_your_log_path', value=LOG_PATH)
#region Recupero della logica 
st.write(st.session_state.Obj_Function.funaction)
if 'Obj_Function' in st.session_state :
     #region data
     if 'Data' in st.session_state:
         if st.sidebar.checkbox('Display_data'):
             st.write(st.session_state.Data.head(5))
             remover = st.multiselect('Remove Columns', st.session_state.Data.columns)
             remove = st.button('Save new D_Frame')
             if remove:
                 new_data = st.session_state.Data.drop(columns=remover)
                 st.session_state.Data = new_data
     
         if st.sidebar.button('Clear_Data'):
             st.session_state.pop('Data')

         utils.Try_Force_Corrispondenza(st.session_state.Obj_Function)

         #region  Corrispondenza verificata Carica layers
         if 'Layers' in st.session_state:
             lis_l = []
             for i in st.session_state.Layers:
                 lis_l.append(f'{i.name}: {i.type.value}')
             inn = st.sidebar.selectbox('List_Of_Layers', lis_l)
             #region Processo
             if 'Process' in st.session_state:
                show_summary_in_sidebar(st.session_state.Process)
                if 'Pusched' not in st.session_state:
                    model_name = st.text_input('Unique Model Name')
                    if st.button('Push_Model'):
                        try:
                            #TODO:Costruzione del modello ambiente
                            modello = mo(st.session_state.Layers,input_shape=st.session_state.Process.window_size, name=model_name)
                            st.session_state['Modello'] = modello
                            st.session_state.Modello.build_layers('Ui_Model creation')
                            #HACK: sarebbe bello poter avere un log dal db anziche inserire un True
                            st.session_state['Pusched'] = True
                        except ValueError as e:
                            st.warning(e)
                            raise ValueError(f'################## {e}')
                if st.sidebar.button('Clear_Process_'):
                    st.session_state.pop('Process')
                #region tento di salvare l iterazione sfruttando gli id dei layers
                if 'Pusched' in st.session_state:
                    st.success(f'Layers Succesfuly trained with id: {st.session_state.Modello.id}')
                    tra_name = st.text_input('insert_training_name')
                    #TODO: sarebbe bello avere un logh_path indipendente dall ui
                    if st.button('Pusch new training'):
                       train = training(tra_name,status.PLANNED,st.session_state.Obj_Function.id,st.session_state.Process.id, 
                                        st.session_state.Modello.id, LOG_PATH)

                       pu = train.push_on_db('Pusch from Ui_Env')

                       # HACK: spostato nella classe
                       #if pu != []:
                       #    train = training.convert_db_response(pu[0])

                       if 'Training' not in st.session_state:
                            st.session_state['Training'] = pu
                       else:
                            st.session_state['Training'] = pu

                       if 'Training' in st.session_state:
                           # st.success('move to home')
                           st.switch_page('Ui_Exe.py')
                #endregion
             #endregion             
         #endregion
     #endregion
     if 'Layers' in st.session_state:
         if st.sidebar.button('Clear_Layers'):
             st.session_state.pop('Layers')
     if st.sidebar.button('Clear_logic'):
        st.session_state.pop('Obj_Function')

#endregion
if 'Data' not in st.session_state:
    st.switch_page('pages/1Ui_function.py')

if 'Process' not in st.session_state:
    st.switch_page('pages/Ui_Process.py')

if 'Obj_Function' not in st.session_state:
    st.switch_page('pages/1Ui_function.py')

if 'Layers' not in st.session_state:
    st.switch_page('pages/1Ui_layers.py')
#endregion
