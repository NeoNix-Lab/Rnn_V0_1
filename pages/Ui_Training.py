import io
from operator import indexOf
from Services import Utils as logic_Utils
import sys
import streamlit as st
import streamlit_shadcn_ui as ui
from Services import St_utils, db_Manager as db
from streamlit_ace import st_ace
from Models.Model_Static import CustomDQNModel as model
import ast
from Services import DataRetriver as ichi
import json
from Services import St_utils as st_utils
import pandas as pd
from Models import dati, Iteration as iteration, Training_Model as trainingMod
from Models import Mod_esecutor as model
import numpy as np
from Services.config import Config as conf
from Models.dati import Dati
from Services.DataRetriver import DataRetriver as dataret


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

#TODO: Menage Db configuration
# HINT: tento l utilizzo di un config globale
retriver = dataret()
config = retriver.config

st_utils.header('Training Area', st_utils.PageName.SETTINGS)
tarining = db.retrive_all('training')
traing_objs = []
traing_objs_names = []

#region intestazione
if 'Selected_Iteration' not in st.session_state:
    st.switch_page('Ui_Exe.py')

iter = st.session_state['Selected_Iteration']
function, process, model_, _train = st_utils.build_training_from_tr_record(iter, config)

t1,_,t2 = st.columns(3,gap='large')

with t1:
    st.subheader('Selected Training Details :')
with t2:
    hide = st.checkbox('hide')

if hide == False:
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

if res == 'untrained':
    ex_i = st.expander('Untrained_Iters')
    with ex_i:
        names = [i.name for i in untrained]
        selected = st.selectbox('Untrained', names, label_visibility='collapsed') 
        selected_obj = untrained[indexOf(names,selected)]
        if selected:
            datti_id = selected_obj.dati_id
            dats = db.retive_a_list_of_recordos('id', 'dati', datti_id)
            dats = dati.Dati.convert_db_response(dats[0],config)
            data___ = retriver.fetch_data(dats.name)
           
            lista = [f'Train Data : {dats.train_data}', f'Test Data : {dats.test_data}', f'Work Data : {dats.work_data}']

            set_selector =  st.select_slider('Data set selector', lista, label_visibility='collapsed')

            if set_selector:
                if indexOf(lista,set_selector) == 0:
                    selected = dats.train_data_
                elif indexOf(lista,set_selector) == 2:
                    selected = dats.work_data_
                elif indexOf(lista,set_selector) == 1:
                    selected = dats.test_data_
            # with c1:
            #     st.write(f'Train Data : {dats.train_data}')
            # with c2:
            #     st.write(f'Test Data : {dats.test_data}')
            # with c3:
            #     st.write(f'Work Data : {dats.work_data}')

            if st.button('Train This'):
                 st_utils.st_sessions_states('Dati',dats)
                 st_utils.st_sessions_states('Data',data___)
                 st_utils.st_sessions_states('Iter',selected_obj)
                 env, test_env = st_utils.build_and_test_envoirment(st.session_state.Data, function, process)
                 st_utils.st_sessions_states('Env',[env, test_env])
                 st.rerun()
                 
st.divider()
#endregion

    #TODO: Fix this process
    # st_utils.Load_Data()

if 'Dati' not in st.session_state:
    st.subheader('Grab_your_Data :')

    col1,col2 = st.columns(2,gap='small')
    with col1:
        st.write('Data Load Mode :')
    with col2:
        mode_selector = st.select_slider('slider', ['Use Data Reference', 'Load CSV'], label_visibility='collapsed')
    
    if mode_selector == 'Load CSV':
        if 'Data' in st.session_state:
            st.session_state.pop('Data')
        uploaded_file = st.file_uploader("Load a CSV", type="csv", label_visibility='collapsed')

        # Verifica se un file e stato caricato
        if uploaded_file is not None:
            # Leggi il CSV in un DataFrame Pandas
            df = pd.read_csv(uploaded_file)


            if 'Data' not in st.session_state:
                st_utils.st_sessions_states('Data',df)
                
            data, removed = st_utils.remove_columns(df)
            
            if removed:
                st_utils.st_sessions_states('Data',data)
                
            
            colonne_df = set(st.session_state.Data.columns)
            chiavi_dict = set(function.data_schema.keys())
            
            dataespander = st.expander('Display Data Schema')

            with dataespander:
                dats1,dats2 = st.columns(2 , gap='Large')
                with dats1:
                    st.subheader('Data Schema :')
                    st.write(list(colonne_df))
                with dats2:
                    st.subheader('Function Data Schema :')
                    st.write(list(chiavi_dict))
             
            corrispondenza = colonne_df == chiavi_dict
            
            if removed:
                st_utils.Try_Force_Corrispondenza(function)
                colonne_df = set(st.session_state.Data.columns)
                chiavi_dict = set(function.data_schema.keys())
                 
                corrispondenza:bool = colonne_df == chiavi_dict
                
            if corrispondenza:
                sd1, sd2 = st.columns(2,gap='small')

                with sd2:
                    ex1 = st.expander('Sow Your Data')
                    with ex1:
                        st.write(st.session_state.Data)
                
                with sd1:
                    ex = st.expander('Proportion Your Data')
                    with ex:
                        work_Data = -1
                        test_Data = -1

                        train_data = st.slider('Train_Data',0,len(st.session_state.Data),value=int(len(st.session_state.Data)/3), step=1)
                        if train_data != len(st.session_state.Data):
                            work_Data = st.slider('Work_Data', train_data, len(st.session_state.Data) ,int(train_data+(len(st.session_state.Data)-train_data)/2), step=1)
                            if work_Data != len(st.session_state.Data):
                                test_Data = st.slider('Test_Data',work_Data,len(st.session_state.Data),int(work_Data+(len(st.session_state.Data)-work_Data)/2), step=1)
        
                st.divider()
                st.warning('Using Identical Names Tables will be overritten')
            else:
                st.error('Unmatching Data Schema')

            col1b, col2b = st.columns(2,gap='large')
            with col1b:
                dati_name = st.text_input('label',label_visibility='collapsed',placeholder='Dati Table Name')
            with col2b:
                if st.button('Save New Data Table', type='secondary',disabled= not corrispondenza):
                    retriver.create_A_Dedicated_Table(dati_name, st.session_state.Data)
                    dato = Dati(retriver.PATH, st.session_state.Data,conf,train_data,work_Data,test_Data, name=dati_name)
                    dato.push_on_db()
                    st_utils.st_sessions_states('Dati',dato)
                    
    if mode_selector == 'Use Data Reference':
            
            col1,col2 = st.columns(2,gap='small')

            tulpe = db.retrive_all('dati')
            dati_objs = []
            nomi = []

            for i in tulpe:
                var = Dati.convert_db_response(i,config)
                if logic_Utils.compare_function_to_dati(function,var):
                    dati_objs.append(var)
                    nomi.append(var.name)

            with col1:
                st.write('')
                st.write('')
                st.write('')
                st.write('Select Your Data Reference:')

            with col2:
                st.write('')
                selection = st.selectbox('Names', nomi,label_visibility='collapsed')
                procede = st.button('Proced')
                ind = indexOf(nomi,selection)
                
            if procede:
                data = retriver.fetch_data(selection)
                st_utils.st_sessions_states('Data',data)
                obj = dati_objs[ind]
                st_utils.st_sessions_states('Dati',obj)
                st_utils.st_sessions_states('ichi_ref',obj.name)
                st.divider()
                st.write(data.head(5))
            
        
if 'Data' in st.session_state and 'Dati' in st.session_state and 'Env' not in st.session_state:
    st.subheader('Build_your_Iteration :')
    st.divider()

    train_data = st.session_state.Dati.train_data
    work_Data = st.session_state.Dati.work_data
    test_Data = st.session_state.Dati.test_data
    
    ds1,_, ds2 = st.columns(3,gap='large')
    
    with ds1:
        iteration_name = st.text_input('Iteration_Name', placeholder='Iteration_Name', label_visibility='collapsed')
        data_selection = [f'Train Data : {train_data}',f'Work Data : {work_Data}', f'Test Data : {test_Data}']
        data_set_selector = st.selectbox('Select yuor data Portion', data_selection)
        
        
    with ds2:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        click_btn = st.button('Build_Iteration')

    if click_btn:
        dati_i = st.session_state.Dati
        dati_id = dati_i.id
        
        selected = None

        if indexOf(data_selection,data_set_selector) == 0:
            selected = dati_i.train_data_
        elif indexOf(data_selection,data_set_selector) == 1:
            selected = dati_i.work_data_
        elif indexOf(data_selection,data_set_selector) == 2:
            selected = dati_i.test_data_
        
        #HINT: non sto registrando i dati nella sezione
        i = iteration.Iterazione(iteration_name,st.session_state.Dati.id,_train.id)
        i.push_on_db()
        iter = db.retrive_last('iterazioni','*')


        if 'Iter' not in st.session_state:
            st.session_state.Iter = iteration.Iterazione.convert_db_response(iter)
        else:
            st.session_state.Iter = iteration.Iterazione.convert_db_response(iter)

        if 'Env' not in st.session_state:
            st.session_state.Env = env, test_env = st_utils.build_and_test_envoirment(selected, function, process)
        else :
            st.session_state.Env = env, test_env = st_utils.build_and_test_envoirment(selected, function, process)

        st.experimental_rerun()


if 'Env' in st.session_state and 'Iter' in st.session_state:
    st.subheader('Run_your_Iteration :')
    st.success('Enviroment correctly setted up')
    st.divider()    
    show_out = st.sidebar.checkbox('show console output')
    
    b1,_,b2 = st.columns(3, gap='large')
    
    with b1:
        layer_notes = st.text_input('txtinput', placeholder='Insert Your Layers Note', label_visibility='collapsed')
        if st.button('Change Iter'):
            st.session_state.pop('Iter')
            st.session_state.pop('Dati')
            st.session_state.pop('Data')
            st.session_state.pop('Env')
            
            st.rerun()
    
    with b2:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        bottone_b2 = st.button('Start_Training')
    
    if bottone_b2:
            #TODO: correggere questa necessita di costruire i layers ogni volta
            #TODO: e soprattutto di doverlo fasre per due reti
            if layer_notes == '':
                layer_notes = 'Pre_Training'
            model_.build_layers(layer_notes)

            EPSILON_START = process.epsilo_start
            EPSILON_END = process.epsilon_end
            EPSILON_REDUCE = process.epsilon_reduce
            GAMMA = process.gamma
            TAU = process.tau
            EPOCHE = process.epochs
            BATCH_SIZE = int(process.batch_size)
            #TODO: manca la replay capacity

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
            
if 'Data' in st.session_state:
    if st.sidebar.button('Clear Data'):
        st.session_state.pop('Data')
        st.session_state.pop('Dati')
        st.rerun()

