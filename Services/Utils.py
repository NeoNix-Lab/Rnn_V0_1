from Models.Training_Model import Training_Model as tr_mod, Training_statu
from Services import db_Manager as db
from Models.process import Process
from Models.Reward_Function import Rewar_Function as rw
from Models.Iteration import Iterazione
from Models.Model_Static import CustomDQNModel as Models
from Models.Model_Static import Layers as Layers
from Models.dati import Dati
from Services.config import Config
import os
import subprocess
import webbrowser
from tensorboard import program

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

object_mapping = {
    'dati': Dati,
    'training': tr_mod,
    'models': Models,
    'layers': Layers,
    'iterazioni': Iterazione,
    'functions': rw,
    'processes': Process
}

def retrive_generic_obj(obj_type:str, db_config:Config):
    lis = []
    lis_name = []
    ids = []
    records = None
    error = ''
    try:
        records = db.retrive_all(obj_type)
            
        obj_class = object_mapping.get(obj_type.lower())
        
        for obj in records:
            var = obj_class.convert_db_response(obj,db_config)
            lis.append(var)
            lis_name.append(var.name)
            ids.append(var.id)
    except ValueError as e:
        error = e
    
    return lis, lis_name, ids, error

def run_tensorboard(log_dir):
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        message = f"Log directory {log_dir} does not exist."
        print(message)
        return message, False

    # Start TensorBoard
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()

    # Open TensorBoard in the default web browser
    webbrowser.open(url)
    message = f"TensorBoard started at {url}"
    print(message)
    return message, True

def compare_function_to_dati(function_obj:rw, dati_obj:Dati):
    chiavi = [i for i in function_obj.data_schema.keys()]
    columns = [i for i in dati_obj.data.columns]

    return chiavi == columns
    