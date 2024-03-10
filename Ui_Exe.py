from email.policy import strict
import streamlit as st
import json
#import CustomDQNModel as model
from Services import IchimokuDataRetriver as model
import Services.Db_Manager as dbm
from streamlit_ace import st_ace
from Models.Flex_Envoirment import EnvFlex as env
from Services import IchimokuDataRetriver as ichi
import pandera as pdr # per la validazione degli schemi
from Models import Reward_Function as rw
from Models import Training_Model as tm

st.set_page_config(
    page_title='Home',
    page_icon=''
    )
st.sidebar.button('bottone')

defoult_code = '''
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

def dict_to_pandera_schema(dicy):
    pandera_schema_args = {
        column: pdr.Column(nullable=True)for column in dicy if dicy[column] is None
    }
    return pdr.DataFrameSchema(pandera_schema_args, strict = False)

st.title('Titolo')

if 'Data' not in st.session_state:
    st.session_state.Data = ichi.fetch_data_from_detailId('14220')

st.write(st.session_state.Data.head(5))

st.warning('Function takes self.env, action')
content = st_ace(language="python", theme="dracula", keybinding="vscode", font_size=16, tab_size=4, value=defoult_code)

user_input = st.text_area("Inserisci la configurazione del modello in JSON:", 
                          '''[
                                {"type": "Dense", "units": 64, "activation": "relu"},
                                {"type": "Dense", "units": 10, "activation": "softmax"}
                             ]''')

# Bottone per costruire il modello
if st.button('Costruisci Modello'):
    try:
        layer_config = json.loads(user_input)

        custom_model = model(layer_config)

        if 'Mod' not in st.session_state:
            st.session_state.Mod = custom_model
    except Exception as e:
        st.error(f'errore {e}')

if st.button('Push layers'):
    st.write(st.session_state.Mod.serialize_Layers_to_json())
    layers_dict = st.session_state.Mod.serialized_layers_dict()
    st.write(layers_dict)
    dbm.push_Layers(layers_dict)

if st.checkbox('show and push model'):
    data = st.session_state.Mod.serialize_to_json()
    st.write(data[0])
    if st.button('push'):
        dbm.push_Model(data[0],data[1])

if st.button('build env'):
    exec(content)
    if 'premia' in locals():
        fillTab_function = locals()['premia']
        if 'Schema' not in st.session_state:
            st.session_state.Schema = locals()['schema']
        else:
            st.session_state.Schema = locals()['schema']

    if 'Env' not in st.session_state:
        st.session_state.Env = env(st.session_state.Data,fillTab_function,[],['wait','buy','sell'],['flat','long','short'])

    if 'Env' in st.session_state:
       st.session_state.Env = env(st.session_state.Data,fillTab_function,[],['wait','buy','sell'],['flat','long','short'])

if st.button('try'):
    st.write(st.session_state.Env.step(1)[0])

    dict_pandera = []

    for i in st.session_state.Schema:
        dict_pandera.append(dict_to_pandera_schema(i))

    st.write(dict_pandera)

if st.button('test push function'):
    fun_model = rw('1','2','3','4')

    fun_model.push_to_db()

if st.button('test push test_item'):
    item = tm(1,1,1,'sss')
    item.push_to_db()
    

# TODO: lista del processo
st.sidebar.title('todo list:')
st.sidebar.text('1- cscriutta o recuperata funzione')
st.sidebar.text('2- recupero dei dati')
st.sidebar.text('3- manipolazione dei dati')
st.sidebar.text('4- verifica dei dati')
st.sidebar.text('5- creazione del modello e dell ambiente dagli schemi della funzione o recupero e validazione')
st.sidebar.text('6- selezione del processo')
st.sidebar.text('7- suddivisione dei dati per i test')
st.sidebar.text('8- addestramento , test e log plus report')





