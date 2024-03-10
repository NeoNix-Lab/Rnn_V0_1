import streamlit as st
import streamlit_shadcn_ui as ui
from Services import IchimokuDataRetriver as ichi

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

size = [0.1,0.1, 1]
col1, col2, tre= st.columns(size,gap='large')

col1.header('he')

col2.header('he')

trigger_btn = ui.button(text="Trigger Button", key="trigger_btn_1")
ui.alert_dialog(show=trigger_btn, title="Alert Dialog", description="This is an alert dialog", confirm_label="OK", cancel_label="Cancel", key="alert_dialog_1")

if 'Data' in st.session_state:pass
    #walker = pyg.walk(st.session_state.Data)