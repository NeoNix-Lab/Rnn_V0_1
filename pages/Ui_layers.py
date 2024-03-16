import streamlit as st
import streamlit_shadcn_ui as ui
from Services import IchimokuDataRetriver as ichi, Db_Manager as db
from streamlit_ace import st_ace
import CustomDQNModel as model
import ast

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

INPUT_LAYER = [
    {
        "type": "InputLayer",  # Cambia "Input" in "InputLayer" 
        "input_shape": (4,),  # Usa "input_shape" invece di "shape" per l'InputLayer
    },
    {
        "type": "Dense",  
        "units": 32,  
        "activation": "relu",  
    },
    {
        "type": "Dense",  
        "units": 2, 
    },
]

# TODO: recuperare le forme dei layer dai dati o mantenre una lista fissa di layer?

lstm1_dict = {
    "type": "LSTM",
    "params": {
        "units": 50,
        "return_sequences": True,
        "input_shape": (20,16), #("n_timesteps", "n_features"),
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

# TODO: definire lo schema di conversione db obj
# TODO: definire lo schema di creazione dell ambinte


#TODO : logica momentaneamente invertita (if not)
if 'Obj' in st.session_state:
    #TODO: Progettare un sistema dui navigazione efficace UX
    st.warning('Please Visit Function page navigation system not implemented yet')

    if st.button('costruisci e carica il modello'):
        if 'Modello' not  in st.session_state:
            try:
                st.session_state.Modello = model.CustomDQNModel(INPUT_LAYER)
            except ValueError as e :
                print (e)

# TODO : sarebbe bello aveere un form piu user frendly per la costruzione del dizionario
else:
    # TDO: costruzione dei layer e recupero da db
    # costruzione
    st.subheader('Build your layers')

    # Start a Streamlit form
    # TODO: verificare la compatibilita del dict layer e schema 
    with st.form("insert_form"):
        # Create the input fields for the form
        layer = st.text_area("Layer", placeholder="Insert layer")
        name = st.text_input("Name", placeholder="Insert name")
        type = st.selectbox("Type", options=model.type, index=0)  # Example types
        schema = st.text_area("Schema", placeholder="Insert schema")
        notes = st.text_area("Notes", placeholder="Insert notes")
        
        # Form submission button
        submitted = st.form_submit_button("Submit")
    
    if submitted:
        # trasformo layer in un dizionario
        try:
            layer = ast.literal_eval(layer)
        except ValueError as e:
            print("Errore durante la conversione del dizionario:", e)
            st.warning(f"Errore durante la conversione: {e}")

        # sovrascrivo il nome
        if 'layer_name' in layer:
            name = layer['layer_name']

        if 'Layers' not in st.session_state:
            st.session_state.Layers = []
            if notes != '':
                new_layer = model.Layers(layer,name,str(type),schema,notes)
            else:
                new_layer = model.Layers(layer,name,str(type),schema)

            st.session_state.Layers.append(new_layer)
        
        # TODO: manca un sistema di verifica di ripetitivita? magari passando per il db
        if 'Layers'  in st.session_state:

            if notes != '':
                new_layer = model.Layers(layer,name,str(type),schema,notes)
            else:
                new_layer = model.Layers(layer,name,str(type),schema)

            st.session_state.Layers.append(new_layer)

    if 'Layers' in st.session_state:
        # TODO: manca un sistema di visualizzazione efficace 
        lis_l = [l.name for l in st.session_state.Layers]
        st.sidebar.multiselect('List_Of_Layers', lis_l)

        st.sidebar.write(f'conto : {len(st.session_state.Layers)}')

        if st.sidebar.button('Clear layers list'):
            st.session_state.pop('Layers')
            st.experimental_rerun()

# TODO: passare tutto al modello tramite conversione e lasciare l ultima verifica al modello stesso
