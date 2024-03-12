import streamlit as st
import streamlit_shadcn_ui as ui
from Models import Reward_Function
from Services import IchimokuDataRetriver as ichi, Db_Manager as db
from streamlit_ace import st_ace
from Models.Reward_Function import Rewar_Function as Rw

############### UTILS

# printa il tipo per ogni attributo di un oggetto x
def print_attributes(obj):
    for attr in vars(obj):
        valore = getattr(obj, attr)
        print(f"{attr}: {valore}, Tipo: {type(valore)}")


################ CODE
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

schema = [{
    'position_status' : None,
    'step' : None,
    'action' : None,
    'balance' : None,
    'reward' : None
},
{   
    'buy' : None,
    'sell' : None,
    'wait' : None
    },
    {
    'long' : None,
    'short' : None,
    'flat' : None}
]
'''

# HACK: migliore il suggerimento
CODE_HINT = """
f_Premia: racchiude funzioni di ricompensa e di aggiornamento 
schema: lista di dizionari
    0:data 
    1:action 
    2:status
"""

DEBBBU = 0
x=1

st.set_page_config(
    page_title="Rnn_Function_Builder",
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
st.subheader('Set, Save or Load your RL Function')

build_btn = st.radio('Select Mode', ['Load', 'Create'], key='Load_Create')

if build_btn == 'Load':
    st.text('start')
    objs = db.retrive_all('functions')

    lis = []
    lis_name = []
    #HACK : trasformare in una funzione 
    #TODO : aggiungere una voto alle funzioni per agevolare la navigazione e le correzzioni
    for obj in objs:
        var = Rw.convert_db_response(obj)
        lis.append(var)
        lis_name.append(f'{var.id}- {var.name}')

    box = st.selectbox('Select Your Function', lis_name)
    if box:
        index = lis_name.index(box)

        if 'Obj' not in st.session_state:
            st.session_state.Obj = lis[index]
        else:
            st.session_state.Obj = lis[index]


if build_btn == 'Create' and 'Obj' not in st.session_state and 'Pushed' not in st.session_state:
    # TODO: verificare la struttura della tabella di env per assicurare la difficile compatibilita
    # TODO: aggiungere funzione per visulizzare some data in order to optimize data_schema function
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
        
        except ValueError as e:
            raise ValueError(f'Mancato recupero delle funzioni error: {e}')

        
        obj = Rw(name, st.session_state.Content, str(st.session_state.Schemas[0]), str(st.session_state.Schemas[1]), str(st.session_state.Schemas[2]),)

        if 'Obj' not in st.session_state:
            st.session_state.Obj = obj
        else:
            st.session_state.Obj = obj

        st.experimental_rerun()

if 'Obj' in st.session_state :
    options = st.session_state.Obj.__dict__.keys()
    bx = st.sidebar.selectbox('Visualizza gli elementi della funzione memorizzata', options)
    if bx:
        if bx == 'funaction':
            st.sidebar.write(f'{bx} :')
            st.sidebar.text(st.session_state.Obj.__dict__[bx])
        else:
           st.sidebar.write(f'{bx} :')
           st.sidebar.write(st.session_state.Obj.__dict__[bx])
    
    if 'Pushed' not in st.session_state and build_btn == 'Create':
        notes = st.text_input('Insert additional notes')

        if st.button('Pusch'):
            
            print_attributes(st.session_state.Obj)
            st.session_state.Obj.pusch_on_db(notes)

            if 'Pushed' not in st.session_state:
                st.session_state.Pushed = True



    if st.sidebar.button('Clear current logic'):
        st.session_state.pop('Obj')
        st.session_state.pop('Pushed')
        st.experimental_rerun()



if 'Data' in st.session_state:pass
    #walker = pyg.walk(st.session_state.Data)


# HACK: testing custom components

# TODO: move to other page


