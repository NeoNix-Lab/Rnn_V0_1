import streamlit as st
from Models.Process import Process as pr, process_type as prt, ProcessLossFunction as prl, ProcessOptimizer as pro
from Services import Db_Manager as dbm
from CustomDQNModel import CustomDQNModel as mo

#region INIT
st.set_page_config(
    page_title="Rnn_Process_Builder",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/library/api-reference/layout',
        'Report a bug': "https://https://shadcn.streamlit.app/DatePicker",
        'About': "# Here you will able to build Customs RL Functions"
    }
)

st.title('Process')
st.subheader('Set, Save or Load your RL Process')
#endregion

#region UTILS
# Funzione per visualizzare il form e catturare l'input dell'utente
def show_process_form():
    with st.form("process_form"):
        name = st.text_input("Name", "")
        description = st.text_area("Description", "")
        epsilon_start = st.number_input("Epsilon Start", value=1.0)
        epsilon_end = st.number_input("Epsilon End", value=0.01)
        epsilon_reduce = st.number_input("Epsilon Reduce", value=0.995)
        gamma = st.number_input("Gamma", value=0.95)
        tau = st.number_input("Tau", value=0.125)
        learning_rate = st.number_input("Learning Rate", value=0.001)
        optimizer = st.selectbox("Optimizer", options=[e.value for e in pro])
        loss = st.selectbox("Loss Function", options=[e.value for e in prl])
        n_episode = st.number_input("Number of Episodes", value=1, step=1)
        epochs = st.number_input("Epochs", value=1000, step=1)
        type_ = st.selectbox("Type", options=[e.value for e in prt])
        window_size = st.number_input("Window Size", value=20.0)
        
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            _optimizer = pro(optimizer)
            _loos = prl(loss)
            _type = prt(type_)

            print(_optimizer)
            process = pr(name=name, notes=description, episodi=n_episode, epoche=epochs, 
                              epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_reduce=epsilon_reduce,
                              gamma=gamma, tau=tau, learning_rate=learning_rate, optimizer=_optimizer, 
                              loss_functions=_loos, type_=_type, window_size=window_size)
            if 'Process' not in st.session_state :
                st.session_state.Process = process
            else:
                st.session_state.Process = process

def show_summary_in_sidebar(process):
    st.sidebar.header("Riepilogo Processo")
    st.sidebar.text(f"Nome: {process.name}")
    st.sidebar.text(f"Descrizione: {process.description}")
    st.sidebar.text(f"Epsilon Start: {process.epsilo_start}")
    st.sidebar.text(f"Epsilon End: {process.epsilon_end}")
    st.sidebar.text(f"Epsilon Reduce: {process.epsilon_reduce}")
    st.sidebar.text(f"Gamma: {process.gamma}")
    st.sidebar.text(f"Tau: {process.tau}")
    st.sidebar.text(f"Learning Rate: {process.learning_rate}")
    st.sidebar.text(f"Ottimizzatore: {process.optimizer}")
    st.sidebar.text(f"Funzione di Perdita: {process.loss}")
    st.sidebar.text(f"Numero Episodi: {process.n_episode}")
    st.sidebar.text(f"Epoche: {process.epochs}")
    st.sidebar.text(f"Tipo: {process.type}")
    st.sidebar.text(f"Dimensione Finestra: {process.window_size}")


#endregion

radio = st.radio('Build Retrive your process', options=['Build', 'Retrive'])

if radio == 'Build':
    show_process_form()

#region Recupero dei processi
if radio == 'Retrive':
    processi = dbm.retrive_all('processes')
    #HACK: visualizzazione grezza ma efficace
    processo = st.selectbox('All processes', processi, help='id, name, description, epsilon_start, epsilon_end, epsilon_reduce, gamma, tau, learning_rate, optimizer, loss, n_episode, epochs, type, windows_size')
    
    if st.button('Save Selection'):
        obj = pr.build_process_from_record(processo)
    
        if 'Process' not in st.session_state :
            st.session_state.Process = obj
        else:
            st.session_state.Process = obj
#endregion

#region Processo in memoria
if 'Process' in st.session_state and radio == 'Build':
    show_summary_in_sidebar(st.session_state.Process)

    if st.sidebar.button('Clear_Process'):
        st.session_state.pop('Process')

    if st.button('Push_Process'):
        st.session_state.Process.push_process()
#endregion

if 'Process' in st.session_state:
    st.write(len(st.session_state.Layers))
    if st.button('Modello'):
        try:
            layer_S = []
            for i in st.session_state.Layers:
                layer_S.append(i.layer)
            modello = mo(layer_S,'Test_1')
            #TODO: momentaneamente evito la sovrascrittura dell input_shape
            modello.build_layers()#input_shape=st.session_state.Process.window_size)

            if 'Modello' not in st.session_state:
                st.session_state.Modello = modello
            else:
                st.session_state.Modello = modello

        except ValueError as e:
            st.warning(e)

       