from operator import indexOf
from sre_constants import LITERAL
from typing import Literal
import streamlit as st
import json
from Services import St_utils
from Services.St_utils import PageName
import Services.db_Manager as dbm
from streamlit_ace import st_ace
from Services import DataRetriver as ichi
from Services import St_utils as utils
import streamlit_shadcn_ui as ui
from Models import Training_Model as tm, Reward_Function as rw, process as pr
import pandas as pd
from st_aggrid import AgGrid as Ag, grid_options_builder
from Models.Model_Static import Layers as l, CustomDQNModel as model
import time
from Services.config import Config
from Services import Utils as logic_utils
import os
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

#TODO: Update Trading Resoult

st.set_page_config(
    page_title='Home',
    page_icon="random",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/library/api-reference/layout',
        'Report a bug': "https://https://shadcn.streamlit.app/DatePicker",
        'About': "# Here you will able to build Customs RL Functions"
    }
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
    
    col1, col2,_,_,_,_, col3 = st.columns(7, gap='large')
    
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
    
    _type = ['Data_Analysis','Run_Tensorboard'] 
    selected_type = st.selectbox('Select Your Report', _type)


    #TODO: forse e l ora di un vero one hot encoding
    if selected_type == 'Data_Analysis': 
        upload_file_actions = St_utils.build_patther('actions',selected_report_path)
        uploaded_file = St_utils.build_patther('resoult',selected_report_path)

        if uploaded_file is not None and upload_file_actions is not None:
            
            datframe_resoult = pd.read_csv(uploaded_file)
            datframe_action = pd.read_csv(upload_file_actions)
            
            df = pd.merge(datframe_resoult,datframe_action, right_index=True,left_index=True,suffixes=('Resoult','Actions'))
           
            
            control_pannel = st.expander('Controls')
            with control_pannel:
                pan1, pan2,pan3, pan4, pan5, pan6 = st.columns(6, gap='large')
                
                with pan5:
                    viewtype = st.multiselect('dataVIewType', ['Status','Reward'], label_visibility='collapsed')
                
                with pan6:
                    run_mode = st.radio('runmode',['Show Df','Hide Df'],label_visibility='collapsed')
                    
                    
                with pan1:
                    satrt = st.number_input('start from',min_value= 1,max_value= len(df),value=1)
                with pan2:
                    end = st.number_input('up to',min_value= satrt+10,max_value= len(df),value=len(df))
            
            if run_mode == 'Show Df':
                st.write(df.iloc[satrt:end])

            if 'Status' in viewtype:
                fig = St_utils.build_basic_resume_chart(df,[satrt,end])
                st.pyplot(fig)
                
            if 'Reward' in viewtype:
                figa = St_utils.build_action_profit_chart(df,[satrt,end])
                st.pyplot(figa)
                 

                
            # while run_mode == 'Run':
            #     fig = St_utils.build_basic_resume_chart(df,range_min_max)
                
            #     chart_plaicholder.pyplot(fig)
        
            #     # Simula aggiornamento periodico dei dati e range
            #     time.sleep(5)  # Aggiorna ogni 5 secondi
                
            #     # Aggiorna l'intervallo di righe (per esempio, cicla attraverso il DataFrame)
            #     range_min_max[0] = (range_min_max[0] + 10)
            #     range_min_max[1] = (range_min_max[1] + 10) 
            #     if range_min_max[1] == 0:
            #         range_min_max[1] = 1
            
            # Titolo dell applicazione
            st.title('Analisi dei Dati del Reinforcement Learning')
            
            # Distribuzione delle Azioni
            st.header('Distribuzione delle Azioni')
            fig, ax = plt.subplots()
            df['actionActions'].value_counts().plot(kind='bar', ax=ax)
            ax.set_title('Distribuzione delle Azioni')
            ax.set_xlabel('Azione')
            ax.set_ylabel('Frequenza')
            st.pyplot(fig)
            
            # Ricompensa Media per Azione
            st.header('Ricompensa Media per Azione')
            fig, ax = plt.subplots()
            df.groupby('actionActions')['reword'].mean().plot(kind='bar', ax=ax)
            ax.set_title('Ricompensa Media per Azione')
            ax.set_xlabel('Azione')
            ax.set_ylabel('Ricompensa Media')
            st.pyplot(fig)
            
            # Ricompensa nel Tempo
            st.header('Ricompensa nel Tempo')
            fig, ax = plt.subplots()
            df['balance'].plot(kind='line', ax=ax)
            ax.set_title('Ricompensa nel Tempo')
            ax.set_xlabel('Indice')
            ax.set_ylabel('Ricompensa')
            st.pyplot(fig)
            
            # Heatmap delle Transizioni di Stato
            st.header('Heatmap delle Transizioni di Stato')
            fig, ax = plt.subplots()
            transition_matrix = pd.crosstab(df['balance'], df['selection'])
            sns.heatmap(transition_matrix, cmap='viridis', ax=ax)
            ax.set_title('Heatmap delle Transizioni di Stato')
            ax.set_xlabel('Stato Successivo')
            ax.set_ylabel('Stato Attuale')
            st.pyplot(fig)
            
            # Ricompensa per Modalita di Selezione dell Azione
            st.header('Ricompensa per Modalita di Selezione dell Azione')
            fig, ax = plt.subplots()
            sns.boxplot(x='selection', y='reword', data=df, ax=ax)
            ax.set_title('Ricompensa per Modalita di Selezione dell Azione')
            ax.set_xlabel('Modalita di Selezione dell Azione')
            ax.set_ylabel('Ricompensa')
            st.pyplot(fig)
            
            # Matrice di Correlazione
            new_df = pd.DataFrame({'Price':df['actionActions'],'reword':df['reword']})
            st.header('Matrice di Correlazione')
            fig, ax = plt.subplots()
            correlation_matrix = new_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Matrice di Correlazione')
            st.pyplot(fig)
            
            # Analisi delle Sequenze di Azioni
            st.header('Sequenza delle Azioni nel Tempo')
            fig, ax = plt.subplots()
            df.pivot(index='step', columns='Price', values='actionActions').plot(ax=ax)
            ax.set_title('Sequenza delle Azioni nel Tempo')
            ax.set_xlabel('Passo Temporale')
            ax.set_ylabel('Azione')
            st.pyplot(fig)
            

            # for i in range(len(uploaded_file)):
            #     try:
            #         ex = st.expander(f'Data_Settings_{i}', False)
            #         df = pd.read_csv(uploaded_file[i])
            #         df_len = len(df.iloc[:, 0])
            #         with ex:
            #             mindatarange = st.slider(f'min data range_{i}', min_value=0, max_value=df_len, value=0)
            #             maxdatarange = st.slider(f'max data range_{i}', min_value=mindatarange, max_value=df_len, value=df_len)
            #             tables = st.checkbox(f'Show Tables_{i}')
            #             lines = [i for i in df.columns]
            #             linee = st.multiselect('Select_your_Lines', lines)
                    
            #         new_df = df.iloc[mindatarange:maxdatarange]
                    
            #         if tables:
            #             st.write("Loaded DataFrame:", df)
            #         if new_df is not None:
            #             # st.write(new_df.describe())
            #             # if len(linee) > 0:
            #             #     grapf = utils.display_stats(new_df,linee,facecolor='#3CB371FF',plot_color='#888888FF')
            #             # Comparative Analysis (Example)
            #             st.write("Comparative Analysis")
            #             compare_col = st.selectbox("Select column to compare", df.columns)
            #             compare_fig = px.histogram(df, x=compare_col, title='Histogram')
            #             st.plotly_chart(compare_fig)

            #             # Text Column Metrics
            #             st.write("Text Column Metrics")
            #             text_columns = df.select_dtypes(include=['object', 'string']).columns
            #             if text_columns.any():
            #                 text_column = st.selectbox("Select text column to analyze", text_columns)
            #                 if text_column:
            #                     metrics = logic_utils._text_column_metrics(df, text_column)
            #                     st.write(f"Metrics for column '{text_column}':")
            #                     for metric, value in metrics.items():
            #                         st.write(f"{metric}: {value}")
                                    
            #             st.write("Actions Visualization:")
            #             fig = px.scatter(df, x=df.index, y='action', color='selection', size_max=60, title='Agent Actions')
            #             st.plotly_chart(fig)
        
                # except Exception as e:
                #     st.write(f"Error loading CSV file: {e}")

    elif selected_type == 'Run_Tensorboard':
        last_path = os.path.join(selected_report_path, 'tensorboard')
        
        _,_,_,_,_,_,_,t = st.columns(8,gap='large')
        succes = None
        message = 'Not Loaded Yet'

        with t:
            if st.button('Run Tensorboard'):
                message, succes = logic_utils.run_tensorboard(last_path)
        if succes == True:
            st.success(message)
        elif succes == False:
            st.error(message)

            
    