import streamlit as st
from Services import st_utils as utils
from streamlit_ace import st_ace

st.title('this will be a test env page or an idea')


st.header('env_isin')

DEFOULT_CODE = '''
def flex_buy_andSell(env, price_column_name: str, action: str):
    price = env.Obseravtion_DataFrame[price_column_name][env.current_step]
    _, action_array, action_name, _ = env.Endcode(env.action_space_tab, action)
    _, status_array, statuscode, _ = env.Endcode(env.position_tab, env.last_position_status)
    _fees = env.calculatefees()

    if action_name == 'wait':
        env.last_Reward = 0

    elif action_name == 'buy':
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
    env.Obseravtion_DataFrame.loc[step, 'reword'] = env.last_Reward

def premia(env, action):
    flex_buy_andSell(env, 'Price', action)
    fillTab(env)

schema = {
 'Action_Schema': {'wait': None, 'buy': None, 'sell': None}, 
 'Status_Schema': {'flat': None, 'long': None, 'short': None}
}
'''

content = st_ace(language="python", theme="dracula", keybinding="vscode", font_size=12, tab_size=3, value=DEFOULT_CODE)
var = exec(content)


globals()["flex_buy_andSell"] = locals()['flex_buy_andSell']
globals()["fillTab"] = locals()['fillTab']
globals()['premia'] = locals()['premia']

iter = st.session_state['Selected_Iteration']
function, process, model_, _train = utils.build_training_from_tr_record(iter)
if 'e' not in st.session_state:
    st.session_state.e, _= utils.build_and_test_envoirment(st.session_state.Data, function, process, test_function=globals()['premia'])

#selected_columns = [i.name for i in st.session_state.e.Obseravtion_DataFrame.columns()]
colonne = st.multiselect('Visible columns', st.session_state.e.Obseravtion_DataFrame.columns,default=['Price','step','action','position_status','reword'])
area = st.slider('', label_visibility='collapsed', max_value=19, min_value=1, value=4)

st.write(st.session_state.e.Obseravtion_DataFrame.loc[st.session_state.e.current_step-area:st.session_state.e.current_step+area,colonne])

col1, col2,_,_,inp, col3 = st.columns(6)

with col1:
    if st.button('clear'):
        st.session_state.pop('e')



with col2:
    if st.button('reset'):
        st.session_state.e.reset()

with inp:
    act = st.text_input('action', label_visibility='collapsed', max_chars=5)
     
with col3:
    if st.button('step'):
        st.session_state.e.step(int(act))
       
        st.experimental_rerun()

if st.button('show_some details'):
    st.session_state.e.step(act)

    _, action_array, action_name, _ = st.session_state.e.Endcode(st.session_state.e.action_space_tab, act)
    _, status_array, statuscode, _ = st.session_state.e.Endcode(st.session_state.e.position_tab, st.session_state.e.last_position_status)

    st.write('action_array : ')
    st.write(action_array)
    st.write('action_name : ')
    st.write(action_name)
    st.write('status_array : ')
    st.write(status_array)
    st.write('statuscode : ')
    st.write(statuscode)

    if st.button('refreshtab'):
        st.experimental_rerun()




