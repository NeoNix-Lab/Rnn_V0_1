import streamlit as st
from Services.config import Config
import os
from Services import St_utils


St_utils.header('Data_Base Settings', St_utils.PageName.SETTINGS)

# Creazione di tabs
tab1, tab2 = st.tabs(["Streamlit Config", "Colab Config"])

with tab1:
    config =  Config()
    
    c1,c2 = st.columns(2,gap='small')
    
    with c1:
        st.write('Data Path:')
        st.write('Model Path:')
        st.write('Logs Path:')
    
    with c2:
        data_path = st.text_input("Percorso al Database dei Dati", config.data_path,label_visibility='collapsed')
        models_path = st.text_input("Percorso al Database dei Modelli", config.models_path,label_visibility='collapsed')
        logs_path = st.text_input("Percorso alla Directory dei Log", config.logs_path, label_visibility='collapsed')
    
        confirm = st.button("Save Steamlit Config")
        
    if confirm:
        config.data_path = data_path
        config.models_path = models_path
        config.logs_path = logs_path
        config.save_config()
        # TODO:Handle missing path or tables
        st.success("Saved Config!")
    
    # Utilizza i percorsi dei database nelle variabili d'ambiente
    os.environ['DATA_PATH'] = config.data_path
    os.environ['MODLES_PATH'] = config.models_path
    os.environ['LOGS_PATH'] = config.logs_path
   

with tab2:
    config = Config('colab')
    c_1,c_2 = st.columns(2,gap='small')
    
    with c_1:
        st.write('Data Path:')
        st.write('Model Path:')
        st.write('Logs Path:')
    
    with c_2:
        data_path = st.text_input("Percorso al Database dei Dati_colab", config.data_path,label_visibility='collapsed')
        models_path = st.text_input("Percorso al Database dei Modell_colabi", config.models_path,label_visibility='collapsed')
        logs_path = st.text_input("Percorso alla Directory dei Log_colab", config.logs_path, label_visibility='collapsed')
    
        confirm = st.button("Save Colab Config")
        
    if confirm:
        config.data_path = data_path
        config.models_path = models_path
        config.logs_path = logs_path
        config.save_config()
        # TODO:Handle missing path or tables
        st.success("Saved Config!")
    
    # Utilizza i percorsi dei database nelle variabili d'ambiente
    os.environ['DATA_PATH'] = config.data_path
    os.environ['MODLES_PATH'] = config.models_path
    os.environ['LOGS_PATH'] = config.logs_path

