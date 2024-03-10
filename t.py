from Services import IchimokuDataRetriver as ichi
from Models import Reward_Function as cu
import CustomDQNModel as model
import streamlit as st

data = ichi.fetch_details()

st.title('tit')
st.write(data)
