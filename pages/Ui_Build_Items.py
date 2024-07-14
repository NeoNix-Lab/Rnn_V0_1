from operator import indexOf
import streamlit as st
import pandas as pd
from Services.DataRetriver import DataRetriver as dataret
from Models.dati import Dati
import os
from Services import Utils as logic_Utils
from Services import St_utils as utils
from streamlit_ace import st_ace
from Models.Reward_Function import Rewar_Function as Rw
from Models import Model_Static as  model
from Services import db_Manager as dbm
import ast
import json
from Models.Training_Model import Training_Model as training, Training_statu as status

      
#region functions
lstm1_dict = {
    "type": "LSTM",
    "params": {
        "units": 50,
        "return_sequences": True,
        "input_shape": (20,16),
        "name": "LSTM"
    }
}

def build_forms():
    ele = []
    for i in model.layers_type:
        ele.append(i.name)

    with st.form('insert_form_build'):
            # Create the input fields for the form
            layer = st.text_area("Layer", placeholder="Insert layer", value = str(lstm1_dict))
            name = st.text_input("Name", placeholder="Insert name")
            layers_type = st.selectbox("Type", options=ele, index=0)  # Example types
            schema = st.warning('Schemas will be inferted from Logic')#st.text_area("Schema", placeholder="Insert schema")
            notes = st.text_area("Notes", placeholder="Insert notes")
            
            # Form submission button
            submitted = st.form_submit_button("Submit")
        
    if submitted:
        enumeratore = getattr(model.layers_type,layers_type)

        try:
            layer = ast.literal_eval(layer)
        except ValueError as e:
            print(f'############################################################tipo di strato : {type(layer)}')
            print("Errore durante la conversione del dizionario:", e)
            st.warning(f"Errore durante la conversione di un layer nel metodo ui: {e}")

        # sovrascrivo il nome
        if 'layer_name' in layer:
            name = layer['layer_name']

        if 'Layers' not in st.session_state:
            st.session_state.Layers = []
            
            if enumeratore == model.layers_type.OUTPUT:
                schema = st.session_state.Obj_Function.action_schema
            else:
                schema = {**st.session_state.Obj_Function.data_schema, **st.session_state.Obj_Function.status_schema}

            if notes != '':
                new_layer = model.Layers(layer,name,enumeratore,schema,notes)
            else:
                new_layer = model.Layers(layer,name,enumeratore,schema)

            st.session_state.Layers.append(new_layer)
            st.success('Layer succesfuly added to streamlit session state')
            
        
        elif 'Layers'  in st.session_state:
            if notes != '':
                new_layer = model.Layers(layer,name,enumeratore,schema,notes)
            else:
                new_layer = model.Layers(layer,name,enumeratore,schema)

            st.session_state.Layers.append(new_layer)
            st.success('Layer succesfuly added to streamlit session state')
#endregion

# HINT: tento l utilizzo di un config globale
retriver = dataret()
conf = retriver.config

utils.header('Build Your Objects', utils.PageName.BUILD)

dataset, function, process, layers, model_ = st.tabs(['Dataset','Function', 'Process', 'Layers', 'Model'])

with dataset:
    st.subheader('Build your Data reference :')
    st.divider()
    col1,col2 = st.columns(2,gap='small')
    with col1:
        st.write('Data Load Mode :')
    with col2:
        mode_selector = st.select_slider('slider', ['Use Data Reference', 'Load CSV'], label_visibility='collapsed')

    if mode_selector == 'Load CSV':
        uploaded_file = st.file_uploader("Load a CSV", type="csv", label_visibility='collapsed')

        # Verifica se un file e stato caricato
        if uploaded_file is not None:
            # Leggi il CSV in un DataFrame Pandas
            df = pd.read_csv(uploaded_file)

            if 'Data' not in st.session_state:
                utils.st_sessions_states('Data',df)
                
            data, removed = utils.remove_columns(df)
            
            if removed:
                utils.st_sessions_states('Data',data)
            st.write(st.session_state.Data.head(5))
            
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
            col1b, col2b = st.columns(2,gap='large')
            with col1b:
                dati_name = st.text_input('label',label_visibility='collapsed',placeholder='Dati Table Name')
            with col2b:
                if st.button('Save New Data Table'):
                    retriver.create_A_Dedicated_Table(dati_name, st.session_state.Data)
                    dato = Dati(retriver.PATH, st.session_state.Data,conf,train_data,work_Data,test_Data, name=dati_name)
                    dato.push_on_db()
                    utils.st_sessions_states('Dati',dato)
                    
                    
    if mode_selector == 'Use Data Reference':
        
        tulpa = logic_Utils.retrive_generic_obj('dati', conf)
        
        with col1:
            st.write('')
            st.write('')
            st.write('')
            st.write('Select Your Data Reference:')

        with col2:
            st.write('')
            selection = st.selectbox('Names', tulpa[1],label_visibility='collapsed')
            procede = st.button('Proced')
            ind = indexOf(tulpa[1],selection)
            
        if procede:
            data = retriver.fetch_data(selection)
            utils.st_sessions_states('Data',data)
            utils.st_sessions_states('Dati',tulpa[0][ind])
            st.divider()
            st.write(data.head(5))
            
with function:
    
    if 'Data' not in st.session_state:
        tulpa = logic_Utils.retrive_generic_obj('dati', conf)
        st.write('Select Your Data Reference:')
        
        co1,_,co2 = st.columns(3,gap='small')
        
        with co1:
            selection = st.selectbox('Names_Function', tulpa[1],label_visibility='collapsed')

        with co2:
            procede = st.button('Proced with Function')
            ind = indexOf(tulpa[1],selection)
        
            
        data = retriver.fetch_data(selection)
        espander = st.expander('Show Data')
        with espander:
            st.write(data.head(5))
            
        if procede:
            utils.st_sessions_states('Data',data)
            utils.st_sessions_states('Dati',tulpa[0][ind])
            st.divider()
            st.write(data.head(5))
            st.experimental_rerun()
            
    else:
        st.subheader('Build Logic:')    
        e = st.expander('Guide')
        with e:
            st.text(logic_Utils.CODE_HINT)
          
        ar1, ar2 = st.columns(2,gap='large')

        content = st_ace(language="python", theme="dracula", keybinding="vscode", font_size=16, tab_size=4, value=logic_Utils.DEFOULT_CODE, key='new_one')
        
        with ar1:
           name = st.text_input('Insert your Logic Name', label_visibility='collapsed', placeholder='Insert your Logic Name')

        with ar2:
            f_build_btn = st.button('Build Function')

        var = exec(content)
        
        if f_build_btn:
            try:
                if 'Content' not in st.session_state:
                   st.session_state.Content = content
                else:
                    st.session_state.Content = content
        
                if 'Schemas' not in st.session_state:
                    st.session_state.Schemas = locals()['schema']
                else:
                    st.session_state.Schemas = locals()['schema']
        
                if 'Premia' not in st.session_state:
                    st.session_state.Premia = locals()['premia']
                else:
                    st.session_state.Premia = locals()['premia']
        
                pre_dikt = st.session_state.Data.head(1)
                dickt = pre_dikt.to_dict(orient='list')
                st.session_state.Schemas['Data_Schema'] = dickt
            
            except ValueError as e:
                raise ValueError(f'Mancato recupero delle funzioni error: {e}')
        
            
            obj = Rw(name, st.session_state.Content, st.session_state.Schemas['Data_Schema'], st.session_state.Schemas['Action_Schema'], st.session_state.Schemas['Status_Schema'],)
        
            if 'Obj_Function' not in st.session_state:
                st.session_state.Obj_Function = obj
            else:
                st.session_state.Obj_Function = obj
        
            st.experimental_rerun()
            
with process:
    processo = utils.show_process_form()

    if processo is not None:
        utils.st_sessions_states('Process', processo)

with layers:
    if 'Process' not in st.session_state:
         st.write('Select Your Process Reference:')
         
         tulpa = logic_Utils.retrive_generic_obj('processes', conf)
         
         co1,_,co2 = st.columns(3,gap='small')
         
         with co1:
             selection = st.selectbox('Names_Process', tulpa[1],label_visibility='collapsed')
         
         with co2:
             procede = st.button('Proced with Layers')
             ind = indexOf(tulpa[1],selection)
             
         espander = st.expander('Show Process Details')
         with espander:
             utils.show_process_details(tulpa[0][ind])
             
         if procede:
             utils.st_sessions_states('Process',tulpa[0][ind])
             st.experimental_rerun()
             
    if 'Obj_Function' not in st.session_state:
          st.write('Select Your Function Reference:')
          
          tulpa_func = logic_Utils.retrive_generic_obj('functions', conf)
          
          co1,_,co2 = st.columns(3,gap='small')
          
          with co1:
              selection_func = st.selectbox('Names_Function', tulpa_func[1],label_visibility='collapsed')

          with co2:
              procede = st.button('Proced with Layer')
              ind = indexOf(tulpa_func[1],selection_func)
          
              
          espander_func = st.expander('Show Function Details')
          with espander_func:
              utils.show_function_details(tulpa_func[0][ind])
              
          if procede:
              utils.st_sessions_states('Obj_Function',tulpa_func[0][ind])
              st.experimental_rerun()
              
    elif 'Process' in st.session_state and 'Obj_Function' in st.session_state:
        build_forms()
        if 'Layers'  in st.session_state:
            if len(st.session_state.Layers) > 0:
                st.divider()

                c1,_,c3 = st.columns(3,gap='large')
                
                with c3 :
                    if st.button('Push Layers'):
                        st.session_state.Layers[0].p()
                        
with model_:
    st.subheader('Load your layers')
    objs = dbm.retrive_all('layers')
    
    
    lis = []
    lis_name = []
    
    for obj in objs:
        var = model.Layers.convert_db_response(obj)
    
        lis.append(var)
        lis_name.append(f'{var.id}- {var.name}')
    
    box = st.multiselect('Select Your Layer', lis_name)
    w1,w2,w3 = st.columns(3,gap='large')
    
    with w3:
        if st.button('Save your selection'):
        
            indexes = [lis_name.index(sel)for sel in box]
        
            #selected = lis[indexes]
        
            if 'Layers' not in st.session_state:
                st.session_state.Layers = []
                for i in indexes:
                    st.session_state.Layers.append(lis[i])
                #[st.session_state.Layers.append(i)for i in selected]
            else:
                st.session_state.Layers = [] #HACK: resetto la lista ogni volta
                for i in indexes:
                    st.session_state.Layers.append(lis[i])
                
    st.divider()
                
    model_notes = st.text_input('modelnotes',label_visibility='collapsed', placeholder='Insert Model Notes')
    
    c11,__,c33 = st.columns(3,gap='large')
    with c11:
        model_name = st.text_input('modelname',label_visibility='collapsed', placeholder='Insert Model Name')
    with c33:
        bottone = st.button('Push Model on Db')

    if bottone:
        if model_name == '':
            st.warning('Please Provide a Model unique Name')
        else:
            modello = model.CustomDQNModel(st.session_state.Layers, st.session_state.Process.window_size,model_name)
            if model_notes == '':
                modello.build_layers()
            else:
                modello.build_layers(model_notes)
            st.success('Model Correctly Created')
            utils.st_sessions_states('Model', modello)
            st.rerun()
            
        
    
if 'Data' in st.session_state:
    exp = st.sidebar.expander('DataSet')
    with exp:
        st.write(st.session_state.Data)
        
if 'Dati' in st.session_state:
     utils.show_dati_details(st.session_state.Dati,True)

if 'Obj_Function' in st.session_state:
    espand = st.sidebar.expander('Function Details')   
    with espand:
       utils.show_function_details(st.session_state.Obj_Function)
       if st.button('Clear_Function'):
           st.session_state.pop('Obj_Function')
           if 'Layers' in st.session_state:
              st.session_state.pop('Layers')
           st.rerun()
       
if 'Process' in st.session_state:
    espand = st.sidebar.expander('Process Details')   
    with espand:
       utils.show_process_details(st.session_state.Process) 
       if st.button('Clear_Process'):
           st.session_state.pop('Process')
           if 'Layers' in st.session_state:
               st.session_state.pop('Layers')
           st.rerun()
           
if 'Model' in st.session_state:
    espand_model = st.sidebar.expander('Model Details')
    with espand_model:
        utils.show_model_details(st.session_state.Model)
        if st.button('Clear Model'):
            st.session_state.pop('Model')
           
       
if 'Layers' in st.session_state:
    espandi = st.sidebar.expander('Layers  details')
    with espandi:
        for i in st.session_state.Layers:
            st.json(i.layer)
            st.write(f'type: {type(i)}')
        
        if st.button('Clear Layers'):
            st.session_state.pop('Layers')
            st.rerun()
            
if 'Dati' in st.session_state and 'Layers' in st.session_state and 'Process' in st.session_state and 'Obj_Function' in st.session_state and 'Model' in st.session_state:
        espandi = st.expander('Create Training Elements')
        with espandi:
        
            co_1, co_2 = st.columns(2, gap='large')
            log_p = 'LOG_PATH_FITTIZZIA'

            with co_2:
                TRAINING_NAME = st.text_input('Training name', label_visibility='collapsed')
                #TODO: forse in ad training serve il refactor per utilizzare la log path standard
                    
            with co_1:
                st.write('Training Name:')

            st.divider()
            
            _,_,y3 = st.columns(3,gap='large')

            with y3:
                save_training = st.button('Save Your Training')

                    
            if save_training:
                train = training(TRAINING_NAME ,status.PLANNED,st.session_state.Obj_Function.id,st.session_state.Process.id, 
                                                st.session_state.Model.id, conf.logs_path)
                train.push_on_db('Pusch from Build Training')
                st.success('Training Set Up Saved')
                st.switch_page('Ui_Exe.py')

           
                    
            
    

