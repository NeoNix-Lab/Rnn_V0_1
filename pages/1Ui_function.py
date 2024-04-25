from ast import Str
import streamlit as st
import streamlit_shadcn_ui as ui
from Services import IchimokuDataRetriver as ichi, Db_Manager as db
from streamlit_ace import st_ace
from Models.Reward_Function import Rewar_Function as Rw
from Models.Flex_Envoirment import EnvFlex as env
import pandera as pdr
from Services import st_utils as utils


#region UTILS / BASELINE
# HINT: potrei caricare i dati , creare la funzione sui dati poi l ambiente prima di verificare tramite uno step fittizio l efficacia e solo dopo salvare 
# schema funzione e relativa relazione , l ambiente viene da se quando passo lo schema azioni e stati dalla funzione all ambiente
# viceversa quando carico una funzione, carico il relativo schema dati , aspetto nuovi dati filtrati, li inserisco 
# creo a parte un modello che aspetta in numero di barre dal processo, creo un processo relazionato alla funzione e al modello e relaziono
# una funzione a tanti modelli

# printa il tipo per ogni attributo di un oggetto x
def print_attributes(obj):
    for attr in vars(obj):
        valore = getattr(obj, attr)
        print(f"{attr}: {valore}, Tipo: {type(valore)}")
#endregion  

#region GLOBBAL VAR
DEFOULT_CODE = '''
def flex_buy_andSell(env, price_column_name: str, action: str):
    price = env.Obseravtion_DataFrame[price_column_name][env.current_step]
    _, action_array, action_name, _ = env.Endcode(env.action_space_tab, action)
    _, status_array, statuscode, _ = env.Endcode(env.position_tab, env.last_position_status)
    _fees = env.calculatefees()

    if action_name == 'buy':
        if statuscode == 'flat' or statuscode == 0:
            env.last_qty_both = env.current_balance / price
            env.last_Reward = 0
            env.last_position_status = 'long'

        elif statuscode == 'short' or statuscode == 2:
            gain = (env.last_qty_both * price) - env.current_balance
            env.last_Reward = gain
            env.current_balance += gain
            env.last_position_status = 'flat'
            env.last_qty_both = 0

        elif statuscode == 'long' or statuscode == 1:
            env.last_Reward = 0

    elif action_name == 'sell':
        if statuscode == 'flat' or statuscode == 0:
            env.last_Reward = 0
            env.last_qty_both = env.current_balance / price
            env.last_position_status = 'short'

        elif statuscode == 'long' or statuscode == 1:
            gain = (env.last_qty_both * price) - env.current_balance
            env.last_Reward = gain
            env.current_balance += gain
            env.last_position_status = 'flat'
            env.last_qty_both = 0

        elif statuscode == 'short' or statuscode == 2:
            env.last_Reward = 0

    if env.current_balance <= 0:
        env.done = True

def fillTab(env):
    step = env.current_step
    env.Obseravtion_DataFrame.loc[step, 'position_status'] = env.last_position_status
    env.Obseravtion_DataFrame.loc[step, 'step'] = env.current_step
    env.Obseravtion_DataFrame.loc[step, 'action'] = env.last_action
    env.Obseravtion_DataFrame.loc[step, 'balance'] = env.current_balance
    env.Obseravtion_DataFrame.loc[step, 'reward'] = env.last_Reward

# Definisco la funzione di premio
def premia(env, action):
    flex_buy_andSell(env, 'Price', action)
    fillTab(env)

schema = {
 'Action_Schema': {'wait': None, 'buy': None, 'sell': None}, 
 'Status_Schema': {'flat': None, 'long': None, 'short': None}
}

'''

# HACK: migliore il suggerimento
CODE_HINT = """
f_Premia: racchiude funzioni di ricompensa e di aggiornamento 
schema: lista di dizionari
    0:data 
    1:action 
    2:status
"""
#endregion

if 'act_page' not in st.session_state:
    st.session_state.act_page = utils.PageName.FUNCTION.value
else:
    st.session_state.act_page = utils.PageName.FUNCTION.value

#utils.navigate()

#st.set_page_config(
#    page_title="Rnn_Function_Builder",
#    page_icon="random",
#    layout="wide",
#    initial_sidebar_state="collapsed",
#    menu_items={
#        'Get Help': 'https://docs.streamlit.io/library/api-reference/layout',
#        'Report a bug': "https://https://shadcn.streamlit.app/DatePicker",
#        'About': "# Here you will able to build Customs RL Functions"
#    }
#)

st.title('Functions')

 #region DATA aggiungo una sezione per il recupero dei dati

#set_dati = ichi.fetch_details() 
#lis_dati = set_dati['Id']

#sel = st.selectbox('Select your set', lis_dati)
#data = ichi.fetch_data_from_detailId(sel)

#if 'Data' not in st.session_state:
#        st.write(data)

#if st.button('Select_Your_Data'):
#    st.session_state['Data'] = data

#if 'Data' in st.session_state:

#    if st.sidebar.checkbox('Display _data'):
#         st.write(st.session_state.Data.head(5))
#         remover = st.multiselect('Remove Columns', st.session_state.Data.columns)
#         remove = st.button('Save new D_Frame')
#         if remove:
#             new_data = st.session_state.Data.drop(columns=remover)

#             st.session_state.Data = new_data

#    if st.sidebar.button('Clear_Data'):
#        st.session_state.pop('Data')
utils.Load_Data()
#endregion 
if 'Data' in st.session_state:

    build_btn = st.radio('Select Mode', ['Load', 'Create'], key='Load_Create')
    
    #region LOAD
    # TODO : non si aggiorna schema se si utilizza la funzione load
    #HACK : trasformare in una funzione 
    #TODO : aggiungere una voto alle funzioni per agevolare la navigazione e le correzzioni
    if build_btn == 'Load':
        objs = db.retrive_all('functions')
    
        lis = []
        lis_name = []
       
        for obj in objs:
            var = Rw.convert_db_response(obj)
            lis.append(var)
            lis_name.append(f'{var.id}- {var.name}')
    
        box = st.selectbox('Select Your Function', lis_name)
        if box:
            index = lis_name.index(box)
    
            if 'Obj_Function' not in st.session_state:
                st.session_state.Obj_Function = lis[index]
                st.balloons()
            else:
                st.session_state.Obj_Function = lis[index]
                st.balloons()
    
            if st.button('Get_Functions'):
                
                 try:
                     var = exec(st.session_state.Obj_Function.funaction)
                     if 'Content' not in st.session_state:
                        st.session_state.Content = st.session_state.Obj_Function.funaction
                     else:
                         st.session_state.Content = st.session_state.Obj_Function.funaction
                 
                     if 'Schemas' not in st.session_state:
                         st.session_state.Schemas = locals()['schema']
                     else:
                         st.session_state.Schemas = locals()['schema']
                 
                     if 'Premia' not in st.session_state:
                         st.session_state.Premia = locals()['premia']
                     else:
                         st.session_state.Premia = locals()['premia']
                 
                 except ValueError as e:
                     raise ValueError(f'Mancato recupero delle funzioni error: {e}')
    
                 st.switch_page('pages/1Ui_layers.py')
    
    #endregion
    
    #region CREATE if not pushed
    # TODO: verificare la struttura della tabella di env per assicurare la difficile compatibilita
    # TODO: aggiungere funzione per visulizzare some data in order to optimize data_schema function
    if build_btn == 'Create' and 'Obj_Function' not in st.session_state and 'Pushed' not in st.session_state:
       
        name = st.text_input('Insert your Logic Name')
    
        st.text(CODE_HINT)
        content = st_ace(language="python", theme="dracula", keybinding="vscode", font_size=16, tab_size=4, value=DEFOULT_CODE)
        var = exec(content)
        
        f_build_btn = st.button('Build_Rw_Obj')
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
    
    if 'Obj_Function' in st.session_state :
        #region Verify SCHEMAS COERENCY 
        #TODO: dato il casino che sta saltando fuori con le verifiche le rimando a dopo
        #endregion
    
        options = st.session_state.Obj_Function.__dict__.keys()
        bx = st.sidebar.selectbox('Visualizza gli elementi della funzione memorizzata', options)
        if bx:
            if bx == 'funaction':
                st.sidebar.write(f'{bx} :')
                st.sidebar.text(st.session_state.Obj_Function.__dict__[bx])
            else:
               st.sidebar.write(f'{bx} :')
               st.sidebar.write(st.session_state.Obj_Function.__dict__[bx])
        
        if 'Pushed' not in st.session_state and build_btn == 'Create':
            notes = st.text_input('Insert additional notes')
    
            if st.button('Pusch'):
                st.session_state.Obj_Function.pusch_on_db(notes)
    
                st.switch_page('pages/1Ui_layers.py')
    
    
        if st.sidebar.button('Clear current logic'):
            st.session_state.pop('Obj_Function')
            st.session_state.pop('Pushed')
            st.experimental_rerun()
    #endregion 

#region ENVIROMENTR
# TODO : enviroment prende opzionalmente colonne aggiuntive ed una serie di dati che potrebbero esserre considerati parte del processo
# ed una prima ricompensa (probabilmente inutile)

#endregion

# HACK: testing custom components

# TODO: move to other page


