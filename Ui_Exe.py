import streamlit as st
import json
import Services.Db_Manager as dbm
from streamlit_ace import st_ace
from Services import IchimokuDataRetriver as ichi
import streamlit_shadcn_ui as ui
from Models import Training_Model as tm
import pandas as pd
from st_aggrid import AgGrid as Ag, grid_options_builder
from CustomDQNModel import Layers as l

st.set_page_config(
    page_title='Home',
    page_icon=''
    )

#region Init_Method
def build_df_from_objs(objs:list):

    dicts = []
    for i in objs:
        dict_ = {attr: getattr(i, attr) for attr in dir(i) if not attr.startswith('__')}
        dicts.append(dict_)

    df = pd.DataFrame(dicts)
    return df

def build_table(df:pd.DataFrame):
    # Configurazione delle opzioni della griglia per abilitare la selezione di righe
    gb = grid_options_builder.GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection('single', use_checkbox=True)  # Cambia 'multiple' in 'single' per la selezione singola
    grid_options = gb.build()
    grid_options['headerHeight'] = 50
    grid_options['groupHeaderHeight'] = 75
    
    # Visualizzazione del DataFrame con la griglia configurata
    return Ag(df,height=(30+(50*len(df))), gridOptions=grid_options, theme='alpine',
                  enable_enterprise_modules=True, update_mode='SELECTION_CHANGED', fit_columns_on_grid_load=True,)
#endregion

t_iteration = dbm.retrive_all('training')

obj_converted = []
obj_converted_attr_dict = []

for i in t_iteration:
    obj = tm.Training_Model.convert_db_response(i)
    obj_converted.append(obj)
    obj_converted_attr_dict.append(obj.attributi)

df = pd.DataFrame(obj_converted_attr_dict)
gb = grid_options_builder.GridOptionsBuilder.from_dataframe(df)
gb.configure_selection('single', use_checkbox=True)
grid_options = gb.build()
grid_options['headerHeight'] = 50
grid_options['groupHeaderHeight'] = 75

response = Ag(df,height=(30+(50*len(df))), gridOptions=grid_options, theme='alpine',
                  enable_enterprise_modules=True, update_mode='SELECTION_CHANGED', fit_columns_on_grid_load=True,)

if response.selected_rows_id[0] is not None:
    id = int(response.selected_rows_id[0])
    st.write(id)
    st.write(obj_converted_attr_dict[id])

    if st.button('Try_Buuild'):
        process = dbm.retive_a_list_of_recordos('id','processes',[int(obj_converted[id].process_id)])
        function_ =  dbm.retive_a_list_of_recordos('id','functions',[int(obj_converted[id].function_id)])
        layers_ = dbm.retive_a_list_of_recordos('training_id','training_layers',[int(obj_converted[id].id)])
        st.write(process)
        st.write(function_)
        st.write(layers_)



