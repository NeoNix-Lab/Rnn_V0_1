from Models.Process import Process as pr, process_type as prt, ProcessLossFunction as prl, ProcessOptimizer as pro
from Services.IchimokuDataRetriver import fetch_data_from_detailId as f, fetch_details as f
from Services import Db_Manager as db
from CustomDQNModel import Layers as lay, layers_type as ty
import CustomDQNModel as model
from Models.Process import Process as pr

lstm1_dict = {
    "type": "LSTM",
    "params": {
        "units": 50,
        "return_sequences": True,
        "input_shape": (20,16),
        "name": "LSTM"
    }
}
obj = db.retrive_all('layers')
objs=[]
for i in obj:
    objs.append(lay.convert_db_response(i))

layers_ = [lstm1_dict, objs[1].layer, objs[2].layer, objs[3].layer]


modello = model.CustomDQNModel(layers_,'Test_1')

print(f'@@@@@@@@@@@@@@@@@@@@@@{modello.model_layers}')

modello.build_layers()
print(f'@@@@@@@@@@@@@@@@@@@@@@{modello.model_layers}')
