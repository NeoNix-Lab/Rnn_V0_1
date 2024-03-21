import streamlit as st
import streamlit_shadcn_ui as ui
from Services import IchimokuDataRetriver as ichi, Db_Manager as db
from streamlit_ace import st_ace
import CustomDQNModel as model
import ast
from Services import IchimokuDataRetriver as ichi
import json

st.set_page_config(
    page_title="Rnn_models_Builder",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/library/api-reference/layout',
        'Report a bug': "https://https://shadcn.streamlit.app/DatePicker",
        'About': "# Here you will able to build Customs RL Functions"
    }
)

st.title('Reward_Functions')
st.header('Set, Save or Load your RL Function')

#region METHODS / INIT
# TODO: definire lo schema di conversione db obj
# TODO: definire lo schema di creazione dell ambinte
# TODO: recuperare le forme dei layer dai dati o mantenre una lista fissa di layer?
# HACK : sarebbe bello aveere un form piu user frendly per la costruzione del dizionario

def build_forms(obj=None):
    ele = []
    for i in model.layers_type:
        ele.append(i.name)

    if obj is None:
        # Start a Streamlit form
        # TODO: verificare la compatibilita del dict layer e schema 
        with st.form('insert_form_build'):
            # Create the input fields for the form
            layer = st.text_area("Layer", placeholder="Insert layer")
            name = st.text_input("Name", placeholder="Insert name")
            layers_type = st.selectbox("Type", options=ele, index=0)  # Example types
            schema = st.warning('Schemas will be inferted from Logic')#st.text_area("Schema", placeholder="Insert schema")
            notes = st.text_area("Notes", placeholder="Insert notes")
            
            # Form submission button
            submitted = st.form_submit_button("Submit")

    else:
        i = ele.index(obj.type.name)

        with st.form("insert_form_load"):
            # Create the input fields for the form
            layer = st.text_area("Layer", placeholder="Insert layer", value=json.dumps(obj.layer, indent=4))
            name = st.text_input("Name", placeholder="Insert name", value=obj.name)
            layers_type = st.selectbox("Type", options=ele, index=i)
            schema = st.text_area("Schema", placeholder="Insert schema", value=obj.schema)
            notes = st.text_area("Notes", placeholder="Insert notes", value=obj.note)
         
            submitted = st.form_submit_button("Submit")
        
    if submitted:
        enumeratore = getattr(model.layers_type,layers_type)

        try:
            layer = ast.literal_eval(layer)
        except ValueError as e:
            print("Errore durante la conversione del dizionario:", e)
            st.warning(f"Errore durante la conversione di un layer nel metodo ui: {e}")

        # sovrascrivo il nome
        if 'layer_name' in layer:
            name = layer['layer_name']

        if 'Layers' not in st.session_state:
            st.session_state.Layers = []

            if obj == None:
                if enumeratore == model.layers_type.OUTPUT:
                    schema = st.session_state.Obj_Function.action_schema
                else:
                    schema = {**st.session_state.Obj_Function.data_schema, **st.session_state.Obj_Function.status_schema}
                

            if notes != '':
                new_layer = model.Layers(layer,name,enumeratore,schema,notes)
            else:
                new_layer = model.Layers(layer,name,enumeratore,schema)

            st.session_state.Layers.append(new_layer)
        
        if 'Layers'  in st.session_state:
            if notes != '':
                new_layer = model.Layers(layer,name,enumeratore,schema,notes)
            else:
                new_layer = model.Layers(layer,name,enumeratore,schema)

            st.session_state.Layers.append(new_layer)

#endregion

#region DEVELOP SESSION
lstm1_dict = {
    "type": "LSTM",
    "params": {
        "units": 50,
        "return_sequences": True,
        "input_shape": (20,16),
        "name": "LSTM"
    }
}

lstm2_dict = {
    "type": "LSTM",
    "params": {
        "units": 50,
        "name": "LSTM"
    }
}

dense1_dict = {
    "type": "Dense",
    "params": {
        "units": 50,
        "activation": "relu",
        "name": "Dense"
    }
}

dense2_dict = {
    "type": "Dense",
    "params": {
        "units": 3, #"n_outputs",
        "activation": "softmax",
        "name": "Dense"
    }
}

INPUT_LAYER_LIST = [lstm1_dict, lstm2_dict, dense1_dict, dense2_dict]
if 'Obj_Function' not in st.session_state: # obj sarebbe reward_Function
    #TODO: Progettare un sistema dui navigazione efficace UX
    st.warning('Please Visit Function page navigation system not implemented yet')

    if st.button('costruisci e carica il modello'):
        if 'Modello' not  in st.session_state:
            try:
                st.session_state.Modello = model.CustomDQNModel(INPUT_LAYER_LIST)
            except ValueError as e :
                print (e)
#endregion

else:
    modalita = st.radio('Modalita Di Recupero Strati', options=['Create', 'Load'])
    #region RECUPERARE STRATI
    if modalita == 'Load':
        st.subheader('Load your layers')
        objs = db.retrive_all('layers')

        lis = []
        lis_name = []
   
        for obj in objs:
            var = model.Layers.convert_db_response(obj)

            lis.append(var)
            lis_name.append(f'{var.id}- {var.name}')

        box = st.multiselect('Select Your Layer', lis_name)
        if st.button('Save your selection'):

            indexes = [lis_name.index(sel)for sel in box]

            #selected = lis[indexes]

            if 'Layers' not in st.session_state:
                st.session_state.Layers = []
                for i in indexes:
                    st.session_state.Layers.append(lis[i])
                #[st.session_state.Layers.append(i)for i in selected]
            else:
                st.session_state.Layers = []
                for i in indexes:
                    st.session_state.Layers.append(lis[i])



    #endregion

    #region COSTRUZIONE DEGLI STRATI
    if modalita == 'Create':
        st.subheader('Build your layers')

        build_forms()

    #region STRATI SALVATI
    if 'Layers' in st.session_state:
        # TODO: inutile perche non modifica niente, aggiunge e basta, il metodo pensato per generare ui diventa difficele da gestire , sopratutto a livello di
        # risposta nei diversi casi 
        if st.sidebar.checkbox('Modifi_existing layer'):
            lis_l = []
            for i in st.session_state.Layers:
                lis_l.append(i.name)
            #lis_l = [l.name for l in st.session_state.Layers]
            inn = st.sidebar.selectbox('List_Of_Layers', lis_l)
            #if inn:
            #    inde = lis_l.index(inn)
            #    ob = st.session_state.Layers[inde]

            #    # TODO forse mi creava doppioni rendendo impossibili i push!!!! probabilmente nella sezione db
            #    #build_forms(ob)

        if st.sidebar.button('Clear layers list'):
            st.session_state.pop('Layers')
            st.experimental_rerun()

        if st.button('Push Layers'):
            # TODO: gli errori di db non raggiungono streamlit , NEpusha uno alla volta altrimenti va in errore
            for lay in st.session_state.Layers:
                lay.push_layer()
    #endregion
    

    #endregion


    #region COSTRUZIONE DEL MODELLO
    #endregion
# TODO: passare tutto al modello tramite conversione e lasciare l ultima verifica al modello stesso
