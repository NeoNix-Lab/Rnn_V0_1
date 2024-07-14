from operator import indexOf
import streamlit as st
import pandas as pd
from Services.DataRetriver import DataRetriver as dataret
from Models.dati import Dati
import os
from Services import Utils as logic_Utils
from Services import St_utils as utils
import streamlit_shadcn_ui as ui
from streamlit_ace import st_ace
from Models.Reward_Function import Rewar_Function as Rw
from Models import Model_Static as  model
import ast
import json

def build_forms_prog(enumeratore:model.layers_type, oj_function, layer, name):
    ele = []
    for i in model.layers_type:
        ele.append(i.name)

    try:
        layer = ast.literal_eval(layer)
    except ValueError as e:
        print("Errore durante la conversione del dizionario:", e)

    # sovrascrivo il nome
    if 'layer_name' in layer:
        name = layer['layer_name']

    if enumeratore == model.layers_type.OUTPUT:
        schema = oj_function.action_schema
    else:
        schema = {**oj_function.data_schema, **oj_function.status_schema}
            

       
    new_layer = model.Layers(layer,name,enumeratore,schema)
    print(type(new_layer.layer))
    print(new_layer.layer)
    

    return new_layer
  

retriver = dataret()
conf = retriver.config
layers = []
tulpa_func = logic_Utils.retrive_generic_obj('functions', conf)
tulp_lasyer = logic_Utils.retrive_generic_obj('layers', conf)
print(f'tulpa : {tulp_lasyer}')
Function = tulpa_func[0][0]
print('@@@@@@ layers type')
print(type(tulp_lasyer[0][0].layer))
print(type(tulp_lasyer[0][1].layer))
print(type(tulp_lasyer[0][3].layer))
print('@@@@@@ layers')
print(tulp_lasyer[0][0].layer)
print(tulp_lasyer[0][1].layer)
print(tulp_lasyer[0][3].layer)

# lay1 = json.loads(tulp_lasyer[0][0].layer)
# lay2 = json.loads(tulp_lasyer[0][1].layer)
# lay_input = json.loads(tulp_lasyer[0][3].layer)


# lay1 = '{"type": "LSTM", "params": {"units": 50, "name": "LSTM"}}'
# lay2 = '{"type": "Dense", "params": {"units": 50, "activation": "relu", "name": "Dense"}}'
# lay_input = '{"type": "LSTM", "params": {"units": 50, "return_sequences": true, "input_shape": [20, 16], "name": "LSTM"}}'

# l1 = build_forms_prog(model.layers_type.HIDDEN,Function,lay1,'uno')
# l2 = build_forms_prog(model.layers_type.HIDDEN,Function,lay2,'due')
# l3 = build_forms_prog(model.layers_type.INPUT,Function,lay_input,'uno')

layers.append(tulp_lasyer[0][0])
layers.append(tulp_lasyer[0][1])
layers.append(tulp_lasyer[0][3])

print(layers)

modello = model.CustomDQNModel(layers,20,'testprogrammatico',push=True)
print(modello.window_size)
print(type(modello.window_size))
modello.build_layers('testprogrammatico')