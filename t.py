#from Models.Process import Process as pr, process_type as prt, ProcessLossFunction as prl, ProcessOptimizer as pro
from Services.IchimokuDataRetriver import fetch_data_from_detailId as fid, fetch_details as f
from Services import Db_Manager as db
#from CustomDQNModel import Layers as lay, layers_type as ty
#import CustomDQNModel as model
#from Models.Training_Model import Training_Model as training, Training_statu as status
from Services import  st_utils as utils
from Models import dati, Iteration as iteration
import sqlite3

DB_BASE_PATH = 'C:\\Users\\user\\OneDrive\\Desktop\\DB\\RNN_Tuning_V01.db'

d = dati.Dati('14217',['Id','Buffer'], 0.33,0.21,0.33, name='texst')
d.pusch_on_db()

#try:
#    data_tulpe = [(d.name, d.train_data, d.work_data, d.test_data, d.decrease_data, 'd.db_references', 'd.df_or_colonne'),]
#    db.push(data_tulpe, d.DB_SCHEMA, d.INSERT_QUERY, 'name', 0, 'dati')
#except ValueError as e:
#    print(e)


#try:
#    returned_objs = []
#    conn = sqlite3.connect(DB_BASE_PATH)
#    cursor = conn.cursor()
#    cursor.execute(d.DB_SCHEMA)

#    cursor.execute(d.INSERT_QUERY, data_tulpe[0])
        
#    conn.commit()
    

#except ValueError as e:
#    raise(f"Errore durante il push di : {e}")

#    if conn:
#        conn.rollback()  # Annulla le modifiche in caso di errore

#finally:
#    if conn:
#        conn.close()