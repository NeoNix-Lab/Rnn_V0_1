import streamlit as st
import json
import Services.Db_Manager as dbm
from streamlit_ace import st_ace
from Services import IchimokuDataRetriver as ichi, st_utils as utils
import streamlit_shadcn_ui as ui
from Models import Training_Model as tm, Reward_Function as rw, Process as pr
import pandas as pd
from st_aggrid import AgGrid as Ag, grid_options_builder
from CustomDQNModel import Layers as l, CustomDQNModel as model
import time

st.set_page_config(
    page_title='Home',
    page_icon=''
    )

st.title('Iteration_Overview')
#region Init_Method
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
                 name = pr.Process.build_process_from_record(resoult).name
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

if 'Training' in st.session_state:
    radi = st.checkbox('Usa il train nello stato')
    # TODO: QUESTA OPZIONE AL MOMENTO E INUTILE E CMQ TRAINING E UN PUSH ON DB
    if radi:
        st.write(radi)

#region tabella
t_iteration = dbm.retrive_all('training')

obj_converted = []
obj_converted_attr_dict = []

if radi == False or 'Training' not in st.session_state:
    for i in t_iteration:
        obj = tm.Training_Model.convert_db_response(i)
        obj_converted.append(obj)
        obj_converted_attr_dict.append(obj.attributi)
    
    #tento la sostituzione
    new_attr_dict = pulisci_e_filtra_dizionari(obj_converted_attr_dict, list(obj_converted_attr_dict[0].keys()))
    
    df = pd.DataFrame(new_attr_dict)
    gb = grid_options_builder.GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection('single', use_checkbox=True)
    grid_options = gb.build()
    grid_options['headerHeight'] = 50
    grid_options['groupHeaderHeight'] = 75
    
    response = Ag(df,height=(30+(50*len(df))), gridOptions=grid_options, theme='alpine',
                      enable_enterprise_modules=True, update_mode='SELECTION_CHANGED', fit_columns_on_grid_load=True)
    
    try:
        if response.selected_rows_id is not None:
            id = int(response.selected_rows_id[0])
           
    except :
        st.warning('No currently selected training')
#endregion

col1, col2, col3 = st.columns(3)

if radi == False or 'Training' not in st.session_state:
    with col3:
        if st.button('Try_Buuild'):

            st.session_state['Selected_Iteration'] = t_iteration[id-1]
            st.switch_page('pages/Ui_Training.py')

with col1:
    if st.button('Add_New_Iteration'):
        st.switch_page('pages/1Ui_function.py')
        #TODO: momentaneamente sospesa la pulizia degli stati
        st.session_state.clear()




    

