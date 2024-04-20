from typing import Callable
from Models.Training_Model import Training_Model as tr_mod
from Services import Db_Manager as db
from Models import Process as pr, Reward_Function as rw, Flex_Envoirment as flex
from CustomDQNModel import CustomDQNModel as model, Layers as ly
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import streamlit as st
from Services import IchimokuDataRetriver as ichi

def build_training_from_tr_record(record):
    # TODO: refactor per poter ricevere oggetti anziche records
    """
    Costruisce e ritorna gli oggetti Function, Process e Model da un record di training.

    Args:
        record (tuple): Un record del database rappresentante un training.

    Returns:
        tuple: Una tupla contenente tre oggetti in questo ordine specifico:
            - Rewar_Function: L'oggetto Function costruito dal record del database.
            - Process: L'oggetto Process costruito dal record del database.
            - Model: L'oggetto Model costruito utilizzando l'ID modello dal record del training e la dimensione della finestra dal Process.

    Raises:
        Exception: Solleva un'eccezione se c'e un errore nella costruzione degli oggetti dal record del training.
    """
    try:
        train = tr_mod.convert_db_response(record)

        process_ = db.retive_a_list_of_recordos('id', 'processes', train.process_id)
        _process = pr.Process.build_process_from_record(process_[0])

        function_ = db.retive_a_list_of_recordos('id', 'functions', train.function_id)
        _function = rw.Rewar_Function.convert_db_response(function_[0])
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

#def concatena_liste_df(dataframes):
#    df = []

#    try:
#        for i in range(len(dataframes)):
#            #d = pd.concat(dataframes[i], ignore_index=True)
#            d = dataframes[0].concat(dataframes[1], ignore_index=True)
#            df.append(d)

#        return df
#    except ValueError as e:
#        raise e

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
            st.warning('Il set di dati non e compatibile con la funzione')
            if st.button('Tenta di forzare la corrispondenza'):
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

    else:
        st.warning('You need to add data in order to sync it')
