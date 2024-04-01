from Models.Process import Process as pr, process_type as prt, ProcessLossFunction as prl, ProcessOptimizer as pro
from Services.IchimokuDataRetriver import fetch_data_from_detailId as f, fetch_details as f
from Services import Db_Manager as db
from CustomDQNModel import Layers as lay, layers_type as ty
import CustomDQNModel as model
from Models.Training_Model import Training_Model as training, Training_statu as status
from Services import  st_utils as utils
import pandas as pd
import numpy as np

#pro = pr('Test____')

##bp_p = 'C:\\Users\\user\\OneDrive\\Desktop\\DB\\RNN_Tuning_V05.db'
##db.change_db_path(bp_p)

#pro.push_process()

mio_dizionario = {'chiave1': 'valore1', 'chiave2': 'valore2', 'chiave3': 'valore3'}
df = pd.DataFrame([mio_dizionario])

print(df)