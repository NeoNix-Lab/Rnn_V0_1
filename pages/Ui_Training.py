import streamlit as st
import streamlit_shadcn_ui as ui
from Services import IchimokuDataRetriver as ichi, Db_Manager as db
from streamlit_ace import st_ace
import CustomDQNModel as model
import ast
from Services import IchimokuDataRetriver as ichi
import json
from Services import st_utils as utils
import pandas as pd
from Models import dati, Iteration as iteration, DQL_v_0_2 as model

st.set_page_config(
    page_title="trainer",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/library/api-reference/layout',
        'Report a bug': "https://https://shadcn.streamlit.app/DatePicker",
        'About': "# Here you will able to build Customs RL Functions"
    }
)

st.title('Train')

if 'Selected_Iteration' not in st.session_state:
    st.switch_page('Ui_Exe.py')
iter = st.session_state['Selected_Iteration']
function, process, model_, _train = utils.build_training_from_tr_record(iter)

train_or_test = st.select_slider('Train or Test', ['New_Train','Test_Exisisting_model'],label_visibility='collapsed')

if train_or_test == 'New_Train':

    utils.Load_Data()
    utils.Try_Force_Corrispondenza(function)
    
    if 'Data' in st.session_state and 'Env' not in st.session_state:
        
        st.subheader('Build_your_data_Iteration')
        st.write(st.session_state.ichi_ref)
    
        iteration_name = st.text_input('Iteration_Name', value='Test_Iteration')
        name = st.text_input('Data_Iteration_Name', value='Test_Data_Iteration')
        train_data = st.slider('Train_Data',0.000,1.000,value=0.330)
        work_Data = st.slider('Work_Data',0.000,1.000,value=0.330)
        test_Data = st.slider('Test_Data',0.000,1.000,value=0.330)
    
        if st.button('Build_Iteration'):
            d = dati.Dati(st.session_state.ichi_ref,st.session_state.Data, train_data,work_Data,test_Data, name=name)
            d.pusch_on_db()
            d_response= db.retive_a_list_of_recordos('name', 'dati', name)
            #HINT: non sto registrando i dati nella sezione
            iter_data_obj = dati.Dati.convert_db_response(d_response[0])
            id = iter_data_obj.id
            i = iteration.Iterazione(iteration_name,1,_train.id)
            i.push_on_db()
            iter = db.retive_a_list_of_recordos('name','iterazioni',iteration_name)

            if 'Iter' not in st.session_state:
                st.session_state.Iter = iteration.Iterazione.convert_db_response(iter[0])
            else:
                st.session_state.Iter = iteration.Iterazione.convert_db_response(iter[0])
    
            if 'Env' not in st.session_state:
                st.session_state.Env = env, test_env = utils.build_and_test_envoirment(st.session_state.Data, function, process)
            else :
                st.session_state.Env = env, test_env = utils.build_and_test_envoirment(st.session_state.Data, function, process)

            st.experimental_rerun()

    if 'Env' in st.session_state:
        st.write(st.session_state.Env[1])
        st.success('Enviroment correctly setted up')

        if st.button('Start_Training'):
            #TODO: correggere questa necessita di costruire i layers ogni volta
            model_.build_layers('Pre_Training')

            EPSILON_START = process.epsilo_start
            EPSILON_END = process.epsilon_end
            EPSILON_REDUCE = process.epsilon_reduce
            GAMMA = process.gamma
            TAU = process.tau
            EPOCHE = process.epochs
            #TODO: manca la replay capacity

            _mod = model.Trainer(st.session_state.Env[0],model_ ,EPSILON_START,EPSILON_END,EPSILON_REDUCE, GAMMA, TAU, epoche=EPOCHE)
            # TODO: manca la selezione delle metriche 
            # TODO: verificare che i dati dell ambiente siano corretti perche passano da li
            # TODO: per ora non viene gestita la parte di log
            # TODO: manca il batch_size
            _mod.compile_networks(process.optimizer, process.loss, metrics=['accuracy', 'precision', 'recall'])
            _mod.Train(process.n_episode, process.type,'batch')


        if st.sidebar.button('clear env'):
            st.session_state.pop('Env')
            st.session_state.pop('Iter')
            st.experimental_rerun()

        
        


