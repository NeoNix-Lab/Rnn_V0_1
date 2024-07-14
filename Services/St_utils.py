from enum import Enum
from typing import Callable
from streamlit.elements.image import UseColumnWith
from Models.Training_Model import Training_Model as tr_mod, Training_statu
from Services import db_Manager as db
from Models import process as pr
from Models.process import process_type as prt, ProcessLossFunction as prl, ProcessOptimizer as pro

from Models import Reward_Function as rw
from Models import Flex_Envoirment as flex
from Models import Iteration as iter
from Models.dati import Dati
from CustomDQNModel import CustomDQNModel as model
from CustomDQNModel import Layers as ly
from Models.Mod_esecutor import Trainer
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import streamlit as st
from Services import DataRetriver as ichi
from streamlit_ace import st_ace
# from Services import St_utils as utils
# from Services.config import Config


class PageName(Enum):
    BUILD = 'Build'
    HOME = 'Home'
    FUNCTION = 'Function'
    LAYERS = 'Layers'
    PROCESS = 'Process'
    TRAINING = 'Training'
    COMPOSE = 'Compose'
    SETTINGS = 'settings'

def build_training_from_tr_record(record, config):
    # TODO: refactor per poter ricevere oggetti anziche records
    """
    Costruisce e ritorna gli oggetti Function, Process e Model da un record di training.

    Args:
        record (tuple): Un record del database rappresentante un training.

    Returns:
        tuple: Una tupla contenente tre oggetti in questo ordine specifico:
            - Rewar_Function: Loggetto Function costruito dal record del database.
            - Process: Loggetto Process costruito dal record del database.
            - Model: Loggetto Model costruito utilizzando l ID modello dal record del training e la dimensione della finestra dal Process.

    Raises:
        Exception: Solleva un eccezione se c e un errore nella costruzione degli oggetti dal record del training.
    """
    try:
        train = tr_mod.convert_db_response(record)

        process_ = db.retive_a_list_of_recordos('id', 'processes', train.process_id)
        _process = pr.Process.convert_db_response(process_[0], config)

        function_ = db.retive_a_list_of_recordos('id', 'functions', train.function_id)
        _function = rw.Rewar_Function.convert_db_response(function_[0], config)
        _model = build_static_model_from_id(train.model_id, _process.window_size)
       

        return _function, _process, _model, train
        
    except ValueError as e :
        raise(f'errore nella costruzione del processo completo dal record del training : ################################{e}')

def sort_list_of_layers_from_record(record):
    sorted_record = sorted(record, key=lambda x: x[2])
    sorted_indexes = [item[1] for item in sorted_record]

    return sorted_indexes

def build_static_model_from_id(id:int, input_shape:int):
    try:
        model_ = db.retive_a_list_of_recordos('id', 'models', id)
        _model = model_[0]

        list_layers = db.retive_a_list_of_recordos('id_model', 'model_layer_relation', id)

        list_of_indexses = sort_list_of_layers_from_record(list_layers)

        layers_ = db.retive_a_list_of_recordos('id', 'layers', list_of_indexses)

        lay = []
        for i in layers_:
            obj = ly.convert_db_response(i)
            lay.append(obj)

        if id == int(_model[0]):
            return model(lay,input_shape,_model[2],id=id,push=False)
        else:
            raise ValueError('retrived wrong model from db')

    except ValueError as e:
        raise(f'errore nella costruzione del modello statico dal id : ################################{e}')

def build_and_test_envoirment(data:pd.DataFrame, function:rw.Rewar_Function, process:pr.Process, test_action:int=1, test_function:Callable=None):
    action_space = list(pd.DataFrame([function.action_schema]).columns)
    position_space = list(pd.DataFrame([function.status_schema]).columns)

    exec(function.funaction)

    #HINT: per rendere accessibili a livello globale funzioni stringate e necessario recuperarle e reindirizzarle
    globals()["flex_buy_andSell"] = locals()['flex_buy_andSell']
    globals()["fillTab"] = locals()['fillTab']
    globals()['premia'] = locals()['premia']

    if test_function == None:
        env = flex.EnvFlex(data,globals()['premia'],[],action_space,position_space,int(process.window_size),process.fees,process.initial_balance)
    else:
        env = flex.EnvFlex(data,test_function,[],action_space,position_space,int(process.window_size),process.fees,process.initial_balance)

    s = env.step(test_action)
    print(f'[[[[[[[[[[[[[[[{s[0].head(30)}]]]]]]]]]]]]]]]')

    return env, env.Obseravtion_DataFrame

def display_stats(_data_frame:pd.DataFrame, printed_colum:list(), xmax_add=100, facecolor='lightgreen', plot_color='lightblue'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, len(_data_frame.iloc[:, 0]) + xmax_add)
    ax.set_ylim(_data_frame[printed_colum].values.min(), _data_frame[printed_colum].values.max())

    for col in printed_colum:
        ax.plot(_data_frame.index, _data_frame[col], label=col) 

    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(plot_color)
    texto = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize='medium')
    plt.title('Series Details')
    plt.xlabel('Step')
    plt.ylabel('Valore')
    plt.legend()

    st.pyplot(fig)
    
def display_Action_stats(_data_frame:pd.DataFrame, printed_colum:list(), xmax_add=100, facecolor='lightgreen', plot_color='lightblue'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, len(_data_frame['step']) + xmax_add)
    ax.set_ylim(_data_frame[printed_colum].values.min(), _data_frame[printed_colum].values.max())

    for col in printed_colum:
        ax.plot(_data_frame.index, _data_frame[col], label=col) 

    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(plot_color)
    texto = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize='medium')
    plt.title('Series Details')
    plt.xlabel('Step')
    plt.ylabel('Valore')
    plt.legend()

    st.pyplot(fig)

def build_vid(env, destination_path, frame_per_sec=6, printed_colum = ['Span_A_Fast','Span_B_Fast'], xmax_add=100,
          numero_barre = 100, pointer_=r'C:\Users\user\Downloads\ffmpeg-2023-12-23-git-f5f414d9c4-essentials_build\bin\ffmpeg.exe', BE_Type='Agg', data = 0):
    if isinstance (data, pd.DataFrame):
        _data_frame = data
    else:
        _data_frame = env.Obseravtion_DataFrame

    mpl.use(BE_Type)
    mpl.rcParams['animation.ffmpeg_path'] = pointer_

    min_val = _data_frame[printed_colum].min(axis=1)
    max_val = _data_frame[printed_colum].max(axis=1)

    min_val= min_val.tolist()
    max_val= max_val.tolist()

    # Impostazioni per il grafico
    fig, ax = plt.subplots(figsize=(10, 6))
    #ax.bar(x=grafico['Indice'], height=grafico['Close'], bottom=grafico['Bottom'], width=0.2, color='skyblue')
    ax.plot(_data_frame['step'], _data_frame[printed_colum])
    
    plt.title(f'Serie')
    plt.xlabel('Bars')
    plt.ylabel('Prezzo')
    
    
    # Linea per l'animazione
    linea, = ax.plot([], [], 'ro-', markersize=5)
    texto = ax.text(0.05, 0.05, 'Current_Balance', transform=ax.transAxes, fontsize='medium')
    
    # Funzione di inizializzazione per l'animazione
    def init():
        linea.set_data([], [])
        texto.set_text('Current_Balance')
        return linea, texto,
    
    #Funzione di animazione
    def animate(j):
        x = j
        y = _data_frame['Price'][j]
        linea.set_data(x, y)
    
    
        #Recupero la finestra 
        ax.set_ylim(ymin=min(min_val)-10, ymax=max(max_val)+10)

        val_curr_bal = _data_frame['balance'][j]
        val_pos_stat = _data_frame['position_status'][j]
        texto.set_text(f'Current_Balance: {val_curr_bal} Position_Status: {val_pos_stat}, x:{x} y:{y}')
    
        if j > 10:
            ax.set_xlim(xmin=j-9, xmax=j+xmax_add)
            xmax=j+100
            xmin= j-9
    
            if _data_frame['action'][j] == 'buy':
                linea.set_color('green')
                linea.set_data([xmin,xmax], [y,y])

            if _data_frame['action'][j] == 'sell':
                linea.set_color('red')
                linea.set_data([xmin,xmax], [y,y])

            if _data_frame['action'][j] == 'wait':
                linea.set_color('yellow')
                linea.set_data([xmin,xmax], [y,y])
    
        return linea, texto,

    #Crea l'animazione
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(_data_frame), interval=200, blit=True)
    
    # Devo cercare e aprire il file
    writer = animation.FFMpegWriter(fps=frame_per_sec)
    
     # Salva l'animazione come file video o GIF
    anim.save(f'{destination_path}.mp4', writer=writer)
    # oppure come GIF
    # anim.save(f'animazione_serie_{i}.gif', writer='pillow')
    
    plt.close(fig)

def show_list_of_dataset():
    set_dati = ichi.fetch_details()
    lis_dati = set_dati['Id']
    
    sel = st.selectbox('Select your set', lis_dati, index=2)
    return ichi.fetch_data_from_detailId(sel), sel

def remove_columns(data:pd.DataFrame):
     col1, col2 = st.columns(2,gap='large') 
     with col1:
        remover = st.multiselect('Remove Columns', data.columns, label_visibility='collapsed')
     with col2:
        remove = st.button('Remove Selected Columns')
     if remove:
         new_data = data.drop(columns=remover)
         return new_data, remove
     else:
         return data, remove

def Load_Data():
    if 'Data' not in st.session_state:
        data, selezione = show_list_of_dataset()

        if 'ichi_ref' not in st.session_state:
            st.session_state.ichi_ref = str(selezione)
        else:
            st.session_state.ichi_ref = str(selezione)
    
        st.write(data)
    
        if st.button('Select_Your_Data'):
            st.session_state['Data'] = data

    if 'Data' in st.session_state:
    
        if st.sidebar.checkbox('Display _data'):
            st.write(st.session_state.Data.head(5))
            remover = st.multiselect('Remove Columns', st.session_state.Data.columns)
            remove = st.button('Save new D_Frame')
            if remove:
                new_data = st.session_state.Data.drop(columns=remover)
                st.session_state.Data = new_data
    
        if st.sidebar.button('Clear_Data'):
            st.session_state.pop('Data')

def Try_Force_Corrispondenza(Obj_Function):
    if 'Data' in st.session_state:

         colonne_df = set(st.session_state.Data.columns)
         chiavi_dict = set(Obj_Function.data_schema.keys())
         
         corrispondenza = colonne_df == chiavi_dict
         
         if corrispondenza == False:
            st.warning('The dataset is not compatible with the function')
            if st.button('Try to force match'):
                new_dataframe = []
                for i in colonne_df:
                    if i in chiavi_dict:
                        new_dataframe.append(i)
                if chiavi_dict == set(new_dataframe):
                    new = st.session_state.Data[new_dataframe]
                    st.session_state.Data = new
         
                    colonne_df = set(st.session_state.Data.columns)
                    chiavi_dict = set(Obj_Function.data_schema.keys())
         
                    corrispondenza = colonne_df == chiavi_dict
                    st.rerun()

    else:
        st.warning('You need to add data in order to sync it')

def navigate():
    #TODO: implemente this navigation on pages
    if 'current_page' not in st.session_state:
        st.session_state.current_page = PageName.HOME.value

    cont = st.container(border=True)

    with cont:
        col1, col2 = st.columns(2)

        with col1:
            button = st.button('Home')
            if button:
                 st.session_state.current_page = PageName.HOME.value
        with col2:
            slider = st.select_slider('nav', label_visibility='collapsed',options=[pages.value for pages in PageName], value=st.session_state.current_page)
            st.session_state.current_page = slider
    
    if st.session_state.act_page != st.session_state.current_page:
        if st.session_state.current_page == PageName.HOME.value :
            st.switch_page('Ui_Exe.py')
        if st.session_state.current_page == PageName.FUNCTION.value :
            st.switch_page('pages/1Ui_function.py')
        elif st.session_state.current_page == PageName.LAYERS.value :
            st.switch_page('pages/1Ui_layers.py')
        elif st.session_state.current_page == PageName.PROCESS.value :
            st.switch_page('pages/Ui_Process.py')
        elif st.session_state.current_page == PageName.TRAINING.value :
            st.switch_page('pages/Ui_Env.py')
    
    return cont

def show_process_details(process:pr.Process):
    cont = st.container(border=True)

    with cont:
        st.header("Riepilogo Processo")
        st.write(f"Nome: {process.name}")
        st.write(f"Descrizione: {process.description}")
        st.write(f"Epsilon Start: {process.epsilo_start}")
        st.write(f"Epsilon End: {process.epsilon_end}")
        st.write(f"Epsilon Reduce: {process.epsilon_reduce}")
        st.write(f"Gamma: {process.gamma}")
        st.write(f"Tau: {process.tau}")
        st.write(f"Learning Rate: {process.learning_rate}")
        st.write(f"Ottimizzatore: {process.optimizer}")
        st.write(f"Funzione di Perdita: {process.loss}")
        st.write(f"Numero Episodi: {process.n_episode}")
        st.write(f"Epoche: {process.epochs}")
        st.write(f"Tipo: {process.type}")
        st.write(f"Dimensione Finestra: {process.window_size}")
        st.write(f"Fees: {process.fees}")
        st.write(f"Initial BBalance: {process.initial_balance}")

def show_function_details(function:rw.Rewar_Function):
    code = function.funaction
    
    content = st_ace(language="python", theme="dracula", keybinding="vscode", font_size=16, tab_size=4, value=code)

def show_model_details(model:model):
    c1,c2,_ = st.columns(3)
    with c1:
        st.write('Data Shape:')
    with c2:
        st.write(model.window_size)
        
    st.divider()
    for i in model.lay_obj:
        st.json(i.layer)
    
def show_dati_details(dati_obj:Dati, sidebar:bool=False):
    if not sidebar:
        ex = st.expander('Data Details')
    else:
        ex = st.sidebar.expander('Data Details')
        

        
    with ex:
        cont = st.container()

        with cont:
            st.write(f"ID: {dati_obj.id}")
            st.write(f"Name: {dati_obj.name}")
            st.write(f"Train Set: {dati_obj.train_data}")
            st.write(f"Work Set: {dati_obj.work_data}")
            st.write(f"Test Set: {dati_obj.test_data}")
            st.write(f"Decrese: {dati_obj.decrease_data}")
            st.write(f"Db Reference: {dati_obj.db_references}")
            

def show_train_details(train):
    cont = st.container()

    with cont:
        st.header("Dati Review")
        st.write(f"ID: {train.id}")
        st.write(f"Nome: {train.name}")
        st.write(f"Stato: {train.status}")
        st.write(f"ID Funzione: {train.function_id}")
        st.write(f"ID Processo: {train.process_id}")
        st.write(f"ID Modello: {train.model_id}")
        st.write(f"Percorso Log: {train.log_path}")
        st.write(f"Data di Creazione: {train.creation_date}")
        st.write(f"Miglior Risultato: {train.best_resoult}")

def find_not_trained_iters(tra_id):
    resoult = ''
    obj = []
    records = db.retive_a_list_of_recordos('training_id', 'iterazioni', tra_id)

    if records == None:
        resoult = 'empty'
    else:
        for i in records:
            ob = iter.Iterazione.convert_db_response(i)
            if ob.log_path == 'Not_posted_yet':
                obj.append(ob)

        if len(obj) == 0:
            resoult = 'trained'
        else:
            resoult = 'untrained'
    
    return obj, resoult

def find_not_trained_train():
    obj = []
    records = db.retive_a_list_of_recordos('status', 'training', Training_statu.PLANNED)
    for i in records:
        obj.append(tr_mod.convert_db_response(i))

    return obj

def SaveRecords(trainer:tr_mod,iter:iter):
    pass

def ShowFunctionForm(function_obj: rw.Rewar_Function):
    cont = st.container(border=True)
    data = pd.DataFrame.from_dict(function_obj.data_schema,orient='index')
    action = pd.DataFrame.from_dict(function_obj.action_schema,orient='index')
    status =pd.DataFrame.from_dict(function_obj.status_schema,orient='index')

    with cont:
        st.subheader("Function Details :")
        fc1, _, fc2 = st.columns(3, gap='small')
        with fc2:
            st.write(f"ID: {function_obj.id}")
        with fc1:
            st.write(f"Name: {function_obj.name}")
        st.write(f"Actions Schema:")
        st.dataframe(action,use_container_width=True)
        st.write(f"Data Schema:")
        st.dataframe(data,use_container_width=True)
        st.dataframe(status,use_container_width=True)
        
def st_sessions_states(statename, statevalue):
    if statename not in st.session_state:
        st.session_state[statename] = statevalue
    else:
        st.session_state[statename] = statevalue
        
def header(title, page_name:PageName):
    # st.divider()
    
    c1,c2,_,_,_,_,_,_,_,c3,c4,c5 = st.columns(12,gap='small')
    with c1:
        st.title(f'{title}')
        
    with c3:
        if page_name != PageName.SETTINGS:
            if st.button('DB Settings'):
                st.switch_page('pages/Ui_Tests.py')
        else:
            if st.button('Home__'):
                st.switch_page('Ui_Exe.py')
    with c4:
        if page_name != PageName.BUILD:
            if st.button('Build Items'):
                st.switch_page('pages/Ui_Build_Items.py')
        else:
            if st.button('Home__'):
                st.switch_page('Ui_Exe.py')
    with c5:
        if page_name != PageName.COMPOSE:
            if st.button('Compose'):
                 st.switch_page('pages/Ui_Add_Training.py')
        else:
           if st.button('Home__'):
                st.switch_page('Ui_Exe.py') 
    st.divider()
    
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
        fees = st.number_input("Fees", value=0.01)
        ini = st.number_input("Initial balance", value=100000)
        
        submitted = st.form_submit_button("Submit New Process")
        
        if submitted:
            _optimizer = pro(optimizer)
            _loos = prl(loss)
            _type = prt(type_)

            process = pr.Process(name=name, notes=description, episodi=n_episode, epoche=epochs, 
                              epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_reduce=epsilon_reduce,
                              gamma=gamma, tau=tau, learning_rate=learning_rate, optimizer=_optimizer, 
                              loss_functions=_loos, type_=_type, window_size=window_size, fees=fees, initial_balance=ini)
            
            process.push_on_db()

            return process
