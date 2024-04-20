import io
import sys
import streamlit as st
import streamlit_shadcn_ui as ui
from Services import IchimokuDataRetriver as ichi, Db_Manager as db, st_utils
from streamlit_ace import st_ace
import CustomDQNModel as model
import ast
from Services import IchimokuDataRetriver as ichi
import json
from Services import st_utils as utils
import pandas as pd
from Models import dati, Iteration as iteration, Training_Model as trainingMod
from Models import Mod_esecutor as model
import numpy as np

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

        work_Data = -1
        test_Data = -1

        train_data = st.slider('Train_Data',0,len(st.session_state.Data),value=int(len(st.session_state.Data)/3), step=1)
        if train_data != len(st.session_state.Data):
            work_Data = st.slider('Work_Data', train_data, len(st.session_state.Data) ,int(train_data+(len(st.session_state.Data)-train_data)/2), step=1)
            if work_Data != len(st.session_state.Data):
                test_Data = st.slider('Test_Data',work_Data,len(st.session_state.Data),int(work_Data+(len(st.session_state.Data)-work_Data)/2), step=1)

        st.write(work_Data)
    
        if st.button('Build_Iteration'):
            d = dati.Dati(st.session_state.ichi_ref,st.session_state.Data, train_data,work_Data,test_Data, name=name)
            d.pusch_on_db()
            d_response= db.retive_a_list_of_recordos('name', 'dati', name)
            dati_i = dati.Dati.convert_db_response(d_response[0])
            dati_id = dati_i.id
            
            #HINT: non sto registrando i dati nella sezione
            i = iteration.Iterazione(iteration_name,dati_id,_train.id)
            i.push_on_db()
            iter = db.retive_a_list_of_recordos('name','iterazioni',iteration_name)


            if 'Iter' not in st.session_state:
                st.session_state.Iter = iteration.Iterazione.convert_db_response(iter[0])
            else:
                st.session_state.Iter = iteration.Iterazione.convert_db_response(iter[0])
    
            if 'Env' not in st.session_state:
                st.session_state.Env = env, test_env = utils.build_and_test_envoirment(dati_i.train_data_, function, process)
            else :
                st.session_state.Env = env, test_env = utils.build_and_test_envoirment(dati_i.train_data_, function, process)

            st.experimental_rerun()

    if 'Env' in st.session_state:
        st.success('Enviroment correctly setted up')
        show_out = st.sidebar.checkbox('show console output')

        if st.button('Start_Training'):
            #TODO: correggere questa necessita di costruire i layers ogni volta
            #TODO: e soprattutto di doverlo fasre per due reti

            model_.build_layers('Pre_Training')

            EPSILON_START = process.epsilo_start
            EPSILON_END = process.epsilon_end
            EPSILON_REDUCE = process.epsilon_reduce
            GAMMA = process.gamma
            TAU = process.tau
            EPOCHE = process.epochs
            #TODO: manca la replay capacity
            _mod = model.Trainer(st.session_state.Env[0],  model_, EPSILON_START,EPSILON_END,EPSILON_REDUCE, GAMMA, TAU, _train.name, epoche=EPOCHE)

            # TODO: manca la selezione delle metriche 
            # TODO: verificare che i dati dell ambiente siano corretti perche passano da li
            # TODO: per ora non viene gestita la parte di log
            # TODO: manca il batch_size
            if show_out:
                old_stdout = sys.stdout
                redirected_output = io.StringIO()
                sys.stdout = redirected_output
                output_placeholder = st.empty()

            _mod.compile_networks(process.optimizer, process.loss, metrics=['accuracy'])
            _mod.Train(process.n_episode, process.type,'batch')

            if show_out:
                output_placeholder.text_area("Captured Output", value=redirected_output.getvalue(), height=300)
                redirected_output.close()
                sys.stdout = old_stdout

            if '_mod' not in st.session_state:
                st.session_state._mod = _mod
            else:
                st.session_state._mod = _mod

            best_gain = []

            for i in range(len(st.session_state._mod.ep_report)):
                g = st.session_state._mod.ep_report[i]['balance'].iloc[-1] - st.session_state._mod.env.initial_balance
                best_gain.append(g)

            #region Updating
            st.session_state.Iter.update_resoult(st.session_state.Iter.id, {'train_result':st.session_state.Env[0].current_balance })
            st.session_state.Iter.update_resoult(st.session_state.Iter.id, {'log_path':_mod.path_2 })
            _train.update_best_resoult(max(best_gain))
            _train.update_status(trainingMod.Training_statu.TRAINED)
            _train.update_path(_mod.path)
            #endregion

        episodes = [i for i in range(len(st.session_state._mod.ep_report))]
        episode = st.selectbox('Select_Episode', episodes)
        if episodes:
            details_type = st.select_slider(label='Show_Details',label_visibility='collapsed', options=['Observatio_Df', 'Action_Selection'])

            if details_type == 'Observatio_Df':
                lines = [i for i in st.session_state._mod.ep_report[episode].columns]
                linee = st.multiselect('Select_your_Lines', lines, default=['Price', 'balance'])
        
                if len(linee) > 0:
                    grapf = st_utils.display_stats(st.session_state.Env[0].Obseravtion_DataFrame,linee,facecolor='#3CB371FF',plot_color='#888888FF')

                gain = st.session_state._mod.ep_report[episode]['balance'].iloc[-1] - st.session_state._mod.env.initial_balance
                st.warning(f'Guadagno durante l addestramento: {gain}')

                if st.checkbox('Show Observatio DF'):
                    st.write(st.session_state._mod.ep_report[episode])

            else:
                st.write(st.session_state._mod.action_report_for_episode[episode])
                model_count = (st.session_state._mod.action_report_for_episode[episode]['selection'] == 'model').sum()
                random_count = (st.session_state._mod.action_report_for_episode[episode]['selection'] == 'random').sum()

                st.warning(f'{model_count} azioni sono state selezionate dal modello')
                st.warning(f'{random_count} azioni sono state selezionate randomicamente')

        #if st.button('Save_Model'):
        #    st.session_state._mod.save()


        if st.sidebar.button('clear env'):
            st.session_state.pop('Env')
            st.session_state.pop('Iter')
            st.experimental_rerun()

        
        


