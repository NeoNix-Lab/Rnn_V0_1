from Models.Process import Process as pr, process_type as prt, ProcessLossFunction as prl, ProcessOptimizer as pro
from Services.IchimokuDataRetriver import fetch_data_from_detailId as f, fetch_details as f
from Services import Db_Manager as db
from CustomDQNModel import Layers as lay, layers_type as ty
import CustomDQNModel as model
from Models.Training_Model import Training_Model as training, Training_statu as status


obj = db.retrive_all('layers')
objs=[]
for i in obj:
    objs.append(lay.convert_db_response(i))

ultimo_elemento = objs.pop()
objs.insert(0,ultimo_elemento)

print(f'@@@@@@@@@@@@@@@@@@@@@@{len(objs)}')
print(f'@@@@@@@@@@@@@@@@@@@@@@{type(objs[0])}')


_lay = []
for i in objs:
    _lay.append(i.layer)

modello = model.CustomDQNModel(_lay,'Test_1')

print(f'@@@@@@@@@@@@@@@@@@@@@@{modello.model_layers}')

modello.build_layers(33)
print(f'@@@@@@@@@@@@@@@@@@@@@@{modello.model_layers}')
print(f'@@@@@@@@@@@@@@@@@@@@@@{modello.lay_obj}')

#train = training('tra_name',status.PLANNED,0,0, 'dsfasd')
#l=[1,1,1,1,1]
#train.pusch_on_db(l,'Pusch from Ui_Env')

#resoult = db.retive_a_list_of_recordos('id','process',[1])

#print(resoult)