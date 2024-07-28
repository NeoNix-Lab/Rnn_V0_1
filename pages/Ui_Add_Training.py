from operator import indexOf
from sre_constants import RANGE
import streamlit as st
from streamlit.type_util import OptionSequence
from tensorflow.python.ops.ragged.ragged_tensor import session
from Services import db_Manager as db
from Models.Reward_Function import Rewar_Function as Rw
from Services import St_utils as utils, config as conf
from Models.process import Process as pr
from Models.Model_Static import CustomDQNModel as model, Layers as ly, layers_type as lt
from Models.Training_Model import Training_Model as training, Training_statu as status

st.set_page_config(
    page_title='Compose',
    page_icon="random",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/library/api-reference/layout',
        'Report a bug': "https://https://shadcn.streamlit.app/DatePicker",
        'About': "# Here you will able to build Customs RL Functions"
    }
    )

utils.header('Compose your Training',utils.PageName.COMPOSE)

function = st.expander('Select Your Function')

#TODO: Gestione dei db config
config = conf.Config()

functions = db.retrive_all('functions')
models = db.retrive_all('models')

with function:
    lis = []
    lis_name = []
    
    for obj in functions:
        var = Rw.convert_db_response(obj)
        lis.append(var)
        lis_name.append(f'id:{var.id}__name:{var.name}__')
        
    col1, col2 = st.columns(2, gap='large')

    with col2:
        ex_details = st.checkbox('Show Function')
    
    with col1:
        box = st.selectbox('Select Your Function', lis_name, label_visibility='collapsed')

    if box:
        index = lis_name.index(box)
        utils.ShowFunctionForm(lis[index])
        
        if ex_details:
           st.code(lis[index].funaction, line_numbers=True)

       
            
    st.divider()
            
    if 'FUNCTION' not in st.session_state:
        if st.button('Select your Function'):
            utils.st_sessions_states('FUNCTION',lis[index])
            FUNCTION = lis[index]
            st.experimental_rerun()
            
    if 'FUNCTION' in st.session_state:
        if st.button('Reset Function'):
            st.session_state.pop('FUNCTION')
            st.experimental_rerun()
            
        
if 'FUNCTION' in st.session_state:
    processes = db.retrive_all('processes')
    ex_process = st.expander('Process_Details')
    
    with ex_process:
        proc_objs = []
        proc_name = []
        
        for i in processes:
            st.write(i)
            pr_ob = pr.convert_db_response(i,None)
            proc_objs.append(pr_ob)
            proc_name.append(pr_ob.name)
            
        process = st.selectbox('SelectProcess', proc_name, label_visibility='collapsed')
        prc_obj = proc_objs[indexOf(proc_name,process)]
        utils.show_process_details(prc_obj)
        
        st.divider()

        if 'PROCESS' not in st.session_state:
            if st.button('Select process'):
                utils.st_sessions_states('PROCESS',prc_obj)
                st.experimental_rerun()
                
        if 'PROCESS' in st.session_state:
            if st.button('Reset Process'):
                st.session_state.pop('PROCESS')
                st.experimental_rerun()
                
            

if 'PROCESS' in st.session_state:

    layer_s = db.retive_a_list_of_recordos('type','layers','output')
    modelli = []
    
    for i in models:
        list_of_index = db.retive_a_list_of_recordos('id_model', 'model_layer_relation', i[0])
        sorted = utils.sort_list_of_layers_from_record(list_of_index)
        layers_ = db.retive_a_list_of_recordos('id', 'layers', sorted)
    
        for x in layers_:
            obj = ly.convert_db_response(x)
            if obj.type == lt.OUTPUT:
                if obj.schema == st.session_state['FUNCTION'].action_schema:
                    modelli.append(i)
        
        mod_obj = []
        mod_obj_names = []
        
        for m in modelli:
            obj = utils.build_static_model_from_id(m[0],st.session_state['PROCESS'].window_size)
            mod_obj.append(obj)
            mod_obj_names.append(obj.custom_name)
                    
    model_expander = st.expander('Select your model')

    with model_expander:
        c1, c2 = st.columns(2)
        with c1:
            selected_model = st.selectbox('Select Model', mod_obj_names, label_visibility='collapsed')
            
        
        idx = indexOf(mod_obj_names,selected_model)
        model_obj = mod_obj[idx]
        
        utils.show_model_details(model_obj)
        
        st.divider()
        
        if 'MODEL' not in st.session_state:
            if st.button('Selec Model'):
                utils.st_sessions_states('MODEL', model_obj)
                st.experimental_rerun()
                
                
        if 'MODEL' in st.session_state:
           if st.button('Reset Model'):
                st.session_state.pop('PROCESS')
                st.experimental_rerun()
                
if 'MODEL' in st.session_state:

    st.divider()
    st.subheader('Save New Iteration')
    
    co_1, co_2 = st.columns(2)
    log_p = config.logs_path

    with co_2:
        TRAINING_NAME = st.text_input('Training name', label_visibility='collapsed')
        #HINT: disabled custom log path
        # custom_log_path = st.checkbox('custom_path', label_visibility='collapsed')
        # if custom_log_path:
        #     log_p = st.text_input('Training path', label_visibility='collapsed')
            
        save_training = st.button('Save Your Training')
        
    with co_1:
        st.write('Training Name:')
        st.write('Use Custom Log Path:')
        # if custom_log_path:
        #     st.write('Log Path:')
            
    if save_training:
        train = training(TRAINING_NAME ,status.PLANNED,st.session_state.FUNCTION.id,st.session_state.PROCESS.id, 
                                        st.session_state.MODEL.id, log_p)
        train.push_on_db('Pusch from Add Training')
        retrived = db.retrive_last('training','*')
        retrived_obj = training.convert_db_response(retrived)
        st.success('Training Set Up Saved Updated In Home')
        utils.st_sessions_states('Training',retrived_obj)

