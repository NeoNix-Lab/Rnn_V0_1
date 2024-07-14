from operator import indexOf
import streamlit as st
import json
from Services.St_utils import PageName
import Services.db_Manager as dbm
from streamlit_ace import st_ace
from Services import DataRetriver as ichi
from Services import St_utils as utils
import streamlit_shadcn_ui as ui
from Models import Training_Model as tm, Reward_Function as rw, process as pr
import pandas as pd
from st_aggrid import AgGrid as Ag, grid_options_builder
from CustomDQNModel import Layers as l, CustomDQNModel as model
import time
from Services.config import Config
from Services import Utils as logic_utils
import os

st.set_page_config(
    page_title='Home',
    page_icon=''
    )

utils.st_sessions_states('CONFIG', Config())

#region Init_Method
# HINT: le funzioni ag_grid devono essere json serializabbili
style_by_status = """
function(params) {
    if (params.value === 'planned') {
        return {backgroundColor: 'limegreen', color: 'white'};
    } else {
        return {backgroundColor: 'transparent', color: 'black'};
    }
}
"""


def pulisci_e_filtra_dizionari(lista_dizionari, colonne_desiderate):
    lista_modificata = []

    try:
         for index, dizionario in enumerate(lista_dizionari):
            dizionario_modificato = {chiave: valore for chiave, valore in dizionario.items() if chiave in colonne_desiderate}
        
            # Estrazione del valore dallo status e sua assegnazione
            if 'status' in dizionario_modificato:
                dizionario_modificato['status'] = lista_dizionari[index]['status'].value

            if 'function_id' in dizionario_modificato:
                resoult = dbm.retive_a_list_of_recordos('id','functions',lista_dizionari[index]['function_id'])
                name = rw.Rewar_Function.convert_db_response(resoult[0]).name
                dizionario_modificato['function_id'] = name
                if  dizionario_modificato['function_id'] == '':
                    dizionario_modificato['function_id'] = 'not_named'
            if 'process_id'in dizionario_modificato:
                 resoult = dbm.retive_a_list_of_recordos('id','processes',lista_dizionari[index]['process_id'])[0]
                 name = pr.Process.convert_db_response(resoult).name
                 dizionario_modificato['process_id'] = name
                 if  dizionario_modificato['process_id'] == '':
                     dizionario_modificato['process_id'] = 'not_named'
            if 'model_id' in dizionario_modificato:
                 resoult = dbm.retive_a_list_of_recordos('id','models',lista_dizionari[index]['model_id'])
                 name = resoult[0][2]
                 dizionario_modificato['model_id'] = name
                 if  dizionario_modificato['model_id'] == '':
                     dizionario_modificato['model_id'] = 'not_named'
            
            lista_modificata.append(dizionario_modificato)
    
         return lista_modificata

    except ValueError as e:
        raise (f'errore nella pulizia dei training item per la tabella : {e}')
    
   
#endregion
radi = False
t_iteration = dbm.retrive_all('training')
obj_converted = []
obj_converted_attr_dict = []
    
for i in t_iteration:
    obj = tm.Training_Model.convert_db_response(i)
    obj_converted.append(obj)
    
    obj_converted_attr_dict.append(obj.attributi)

utils.header('Training Home', PageName.HOME)
        
explorer = st.select_slider('Esplorer', ['Train','Explore'], label_visibility='collapsed')
# explorer = st.select_slider('Esplorer', ['Train','Explore'], label_visibility='collapsed')

if explorer == 'Train':
    z1,_,_,_,z3 = st.columns(5,gap='small')
    with z1: 
        st.subheader('Training :')
   

    if 'Training' in st.session_state:
        radi = st.checkbox('Use Current Traing Set_Up?')
        # TODO: QUESTA OPZIONE AL MOMENTO E INUTILE E CMQ TRAINING E UN PUSH ON DB
        if radi:
            espa_tr = st.expander('Traing Details')
            with espa_tr:
                utils.show_train_details(st.session_state.Training)
    
    #region tabella
   
    # if radi == False or 'Training' in st.session_state:
            
        # st.write(obj_converted_attr_dict)
            
    for obj in obj_converted_attr_dict:
        if isinstance(obj['status'], tm.Training_statu):
            obj['status'] = obj['status'].name
    
    # HINT: La pulizia del dizionario non funziona    
    #new_attr_dict = pulisci_e_filtra_dizionari(obj_converted_attr_dict, list(obj_converted_attr_dict[0].keys()))
    
    df = pd.DataFrame(obj_converted_attr_dict)
    gb = grid_options_builder.GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection('single', use_checkbox=True)
    grid_options = gb.build()
    grid_options['headerHeight'] = 50
    
    grid_options['defaultColDef'] = {
        'editable': False, # Set to True if you want columns to be editable
        'resizable': True, # Allows resizing columns
        'width': 150,      # Default width of each column
        'autoWidth': True,
        'autoHeight': True
    }
    grid_options['groupHeaderHeight'] = 75
    #grid_options['columnDefs'] = [
    #    {**col, 'rowStyle': style_by_status if col['field'] == 'status' else None}
    #    for col in grid_options['columnDefs']
    #]
    
    response = Ag(df,height=(30+(50*len(df))), gridOptions=grid_options, theme='alpine',
                      enable_enterprise_modules=True, update_mode='SELECTION_CHANGED', fit_columns_on_grid_load=True, allow_unsafe_jscode=True)
    
    try:
        if response.selected_rows_id is not None:
            id = int(response.selected_rows_id[0])
          
        
           
    except :
        if radi == False:
            st.warning('No currently selected training')
    
    if radi == True:
        st.warning('Using State Saved Training Set_Up')
        
    #endregion
    
    col1, col2, col3 = st.columns(3, gap='large')
    
    with col3:
        if st.button('Run Your Training'):
            if radi == False or 'Training' not in st.session_state:
                st.session_state['Selected_Iteration'] = t_iteration[id]
                
            elif radi:
                sel_item = next((item for item in obj_converted if item.name == st.session_state.Training.name))
                selected_tulp_obj_index = indexOf(obj_converted, sel_item)
                st.session_state['Selected_Iteration'] = t_iteration[selected_tulp_obj_index]
                
            st.switch_page('pages/Ui_Training.py')

                
            
else:
    st.subheader('Training Logs:')
    
    obj_trained = [o for o in obj_converted if o.status != tm.Training_statu.PLANNED]

    obj_name = [o.name for o in obj_trained]
    training_selection = st.selectbox('Select Your Training', obj_name)
    
    index = indexOf(obj_name, training_selection)
    
    # List all directories in the log path
    directories = [d for d in os.listdir(obj_converted[index].log_path)]
    iter_selection = st.selectbox('Select your Iter', directories)
    var = os.path.join(obj_converted[index].log_path, iter_selection)
    

    # Filter directories whose names start with 'episodio_'
    valid_folder = [f for f in os.listdir(var) if os.path.isdir(os.path.join(var, f)) and f.startswith('episodio_')]
    # List all directories in the valid folder
    ep_directory = [e for e in valid_folder]
    
    report = st.selectbox('select your episode',ep_directory)
    
    selected_report_path = os.path.join(var, report)
    
    _type = ['actions','resoult','tensorboard'] 
    selected_type = st.selectbox('Select Your Report', _type)
    
    last_path = os.path.join(selected_report_path, selected_type)
    
    st.subheader('Your Logs Path:')
    st.success(last_path)

    if selected_type == 'actions' or selected_type == 'resoult': 
        uploaded_file = st.file_uploader("Observation CSV file", type="csv",accept_multiple_files=True)

        if uploaded_file is not None:
            
            for i in range(len(uploaded_file)):
                try:
                    ex = st.expander(f'Data_Settings_{i}', False)
                    df = pd.read_csv(uploaded_file[i])
                    df_len = len(df.iloc[:, 0])
                    with ex:
                        mindatarange = st.slider(f'min data range_{i}', min_value=0, max_value=df_len, value=0)
                        maxdatarange = st.slider(f'max data range_{i}', min_value=mindatarange, max_value=df_len, value=df_len)
                        tables = st.checkbox(f'Show Tables_{i}')
                        lines = [i for i in df.columns]
                        linee = st.multiselect('Select_your_Lines', lines)
                    
                    new_df = df.iloc[mindatarange:maxdatarange]
                    
                    if tables:
                        st.write("Loaded DataFrame:", df)
                    if new_df is not None:
                        
                        if len(linee) > 0:
                            grapf = utils.display_stats(new_df,linee,facecolor='#3CB371FF',plot_color='#888888FF')
        
                except Exception as e:
                    st.write(f"Error loading CSV file: {e}")

    elif selected_type == 'tensorboard':
        _,_,t = st.columns(3,gap='large')
        succes = None
        message = 'Not Loaded Yet'

        with t:
            if st.button('Run Tensorboard'):
                message, succes = logic_utils.run_tensorboard(last_path)
        if succes == True:
            st.success(message)
        elif succes == False:
            st.error(message)

            
    