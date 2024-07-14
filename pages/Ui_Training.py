import io
from operator import indexOf
import sys
import streamlit as st
import streamlit_shadcn_ui as ui
from Services import db_Manager as db
from streamlit_ace import st_ace
import CustomDQNModel as model
import ast
from Services import DataRetriver as ichi
import json
from Services import St_utils as st_utils
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

#TODO: Semplicemente mal fatto 
st.title('Train')
tarining = db.retrive_all('training')
traing_objs = []
traing_objs_names = []

if 'Selected_Iteration' not in st.session_state:
    st.switch_page('Ui_Exe.py')
    

# for i in tarining:
#     tr_obj = trainingMod.Training_Model.convert_db_response(i)
#     traing_objs.append(tr_obj)
#     traing_objs_names.append(tr_obj.name)

# st.divider()
# col_1, col_2 = st.columns(2)
# st.write(st.session_state.Selected_Iteration[0][9])
# defoult = next((p for p in traing_objs if p.name == st.session_state.Selected_Iteration[0][9]),None)
# idx = indexOf(traing_objs,defoult)

# with col_1:
#     st.subheader('Select A Different Training_Set_UP:')
# with col_2:
#     selet_Iter = st.selectbox('training selection', traing_objs_names, label_visibility='collapsed',index=idx)

# record = db.retive_a_list_of_recordos('id','training',traing_objs[indexOf(traing_objs_names,selet_Iter)].id)
# rec = record[0]

# st.session_state['Selected_Iteration'] = tarining[idx]
    
st.write(st.session_state['Selected_Iteration'])
iter = st.session_state['Selected_Iteration']
function, process, model_, _train = st_utils.build_training_from_tr_record(iter)

ex = st.expander('Process_Details')
with ex:
    st_utils.show_process_details(process)

ex_f = st.expander('Function_Details')
with ex_f:
    st_utils.show_function_details(function)

ex_l = st.expander('Layers_Details')
with ex_l:
    st_utils.show_model_details(model_)

ex_t = st.expander('Training_Details')
with ex_t:
    st_utils.show_train_details(_train)

untrained, res = st_utils.find_not_trained_iters(_train.id)

if res == 'untrained':
    st.warning('untrained iters')
elif res == 'empty':
    st.error('no builded iters')

train_or_test = st.select_slider('Train or Test', ['New_Train', 'Work_On_Model','Test_Exisisting_model'],label_visibility='collapsed')
    
if res == 'untrained':
    ex_i = st.expander('Untrained_Iters')
    with ex_i:
        names = [i.name for i in untrained]
        selected = st.selectbox('Untrained', names) 
        if selected:
            datti_id = (untrained[names.index(selected)].dati_id)
            dats = db.retive_a_list_of_recordos('id', 'dati', datti_id)
            dats = dati.Dati.convert_db_response(dats[0])
            # st.write(dats.test_data_.head(5))

            c1, c2, c3 = st.columns(3)

            with c1:
                st.write(f'Train Data : {dats.train_data}')
            with c2:
                st.write(f'Test Data : {dats.test_data}')
            with c3:
                st.write(f'Work Data : {dats.work_data}')

            if st.button('Train This'):
                 # TODO: need implementation
                 st.error('not implemented yet')
                 #if 'Iter' not in st.session_state:
                 #    st.session_state.Iter = iteration.Iterazione.convert_db_response(dats)
                 #else:
                 #    st.session_state.Iter = iteration.Iterazione.convert_db_response(dats)

if train_or_test == 'New_Train':

    st_utils.Load_Data()
    st_utils.Try_Force_Corrispondenza(function)

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
            d_response= db.retrive_last('dati', '*')
            dati_i = dati.Dati.convert_db_response(d_response)
            dati_id = dati_i.id
            
            #HINT: non sto registrando i dati nella sezione
            i = iteration.Iterazione(iteration_name,dati_id,_train.id)
            i.push_on_db()
            iter = db.retrive_last('iterazioni','*')


            if 'Iter' not in st.session_state:
                st.session_state.Iter = iteration.Iterazione.convert_db_response(iter)
            else:
                st.session_state.Iter = iteration.Iterazione.convert_db_response(iter)
    
            if 'Env' not in st.session_state:
                st.session_state.Env = env, test_env = st_utils.build_and_test_envoirment(dati_i.train_data_, function, process)
            else :
                st.session_state.Env = env, test_env = st_utils.build_and_test_envoirment(dati_i.train_data_, function, process)

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
            BATCH_SIZE = int(process.window_size)
            #TODO: manca la replay capacity
            #TODO: il batch_size e la dimensione della finestra

            _mod = model.Trainer(st.session_state.Env[0],  model_, EPSILON_START,EPSILON_END,EPSILON_REDUCE, GAMMA, TAU, _train.name, epoche=EPOCHE)

            #HACK:SHow profiler option
            st.write(_mod.profile)

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
            _mod.Train(process.n_episode, process.type, BATCH_SIZE)

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

        if '_mod' in st.session_state:
            note = st.text_area('Update Training Notes')
            p_notes = st.button('Push_Notes')
            
            if p_notes:
                _train.update_notes(note)

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

        if st.sidebar.button('clear env'):
            st.session_state.pop('Env')
            st.session_state.pop('Iter')
            st.experimental_rerun()

else:
    st.write(_train.id)
    lis_of_iters = db.retive_a_list_of_recordos('training_id','iterazioni', [_train.id])
    _iteration_obj = []
    _iteration_names = []

    for i in lis_of_iters:
        obj = iteration.Iterazione.convert_db_response(i)
        _iteration_obj.append(obj)
        _iteration_names.append(obj.name)

    selected_iter = st.selectbox('Select_your_iter', _iteration_names)
    if selected_iter:
        index = _iteration_names.index(selected_iter)
        st.write(f'indice del elemento selezionato: {_iteration_names.index(selected_iter)}')
        st.write(f'path: {_iteration_obj[index].log_path}')

    if st.button('test'):
        dat = st.session_state.Env[0].data
        en = st.session_state.Env[0]
        res = st.session_state._mod.test_existing_model(f'{_iteration_obj[index].log_path}Modello.h5', dat, en)
        st.write(res)

        
        


