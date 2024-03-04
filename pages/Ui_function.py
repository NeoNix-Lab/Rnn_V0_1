import streamlit as st
import streamlit_shadcn_ui as ui
#import pygwalker as pyg
from Services import IchimokuDataRetriver as ichi


st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }

)

st.title('Set your Function')

size = [0.3,0.3, 0.1]
col1, col2, tre= st.columns(size,gap='large')

col1.header('he')

col2.header('he')

trigger_btn = ui.button(text="Trigger Button", key="trigger_btn_1")
ui.alert_dialog(show=trigger_btn, title="Alert Dialog", description="This is an alert dialog", confirm_label="OK", cancel_label="Cancel", key="alert_dialog_1")

if 'Data' in st.session_state:pass
    #walker = pyg.walk(st.session_state.Data)