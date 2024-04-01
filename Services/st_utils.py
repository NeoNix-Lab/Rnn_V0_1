from Models.Training_Model import Training_Model as tr_mod
from Services import Db_Manager as db
from Models import Process as pr, Reward_Function as rw, Flex_Envoirment as flex
from CustomDQNModel import CustomDQNModel as model, Layers as ly
import pandas as pd

def build_training_from_tr_record(record):
    """
    Costruisce e ritorna gli oggetti Function, Process e Model da un record di training.

    Args:
        record (tuple): Un record del database rappresentante un training.

    Returns:
        tuple: Una tupla contenente tre oggetti in questo ordine specifico:
            - Rewar_Function: L'oggetto Function costruito dal record del database.
            - Process: L'oggetto Process costruito dal record del database.
            - Model: L'oggetto Model costruito utilizzando l'ID modello dal record del training e la dimensione della finestra dal Process.

    Raises:
        Exception: Solleva un'eccezione se c'e un errore nella costruzione degli oggetti dal record del training.
    """
    try:
        train = tr_mod.convert_db_response(record[0])

        process_ = db.retive_a_list_of_recordos('id', 'processes', train.process_id)
        _process = pr.Process.build_process_from_record(process_[0])

        function_ = db.retive_a_list_of_recordos('id', 'functions', train.function_id)
        _function = rw.Rewar_Function.convert_db_response(function_[0])
        _model = build_static_model_from_id(train.model_id, _process.window_size)

        return _function, _process, _model
        
    except ValueError as e :
        raise(f'errore nella costruzione del processo completo dal record del training : ################################{e}')

def sort_list_of_layers_from_record(record):
    sorted_record = sorted(record, key=lambda x: x[2])
    sorted_indexes = [item[1] for item in sorted_record]

    return sorted_indexes


def build_static_model_from_id(id:int, input_shape:int):
    try:
        model_ = db.retive_a_list_of_recordos('id', 'models', id)
        _model = model_[0]

        list_layers = db.retive_a_list_of_recordos('id_model', 'model_layer_relation', id)

        list_of_indexses = sort_list_of_layers_from_record(list_layers)

        layers_ = db.retive_a_list_of_recordos('id', 'layers', list_of_indexses)

        lay = []
        for i in layers_:
            obj = ly.convert_db_response(i)
            lay.append(obj)

        if id == int(_model[0]):
            return model(lay,input_shape,_model[2],id=id,push=False)
        else:
            raise ValueError('retrived wrong model from db')

    except ValueError as e:
        raise(f'errore nella costruzione del modello statico dal id : ################################{e}')

def build_and_test_envoirment(data:pd.DataFrame, function:rw.Rewar_Function, process:pr.Process, test_action:int=1):
    action_space = pd.DataFrame([function.action_schema])
    position_space = pd.DataFrame([function.status_schema])

    env = flex.EnvFlex(data,function.funaction,[],action_space,position_space,process.window_size,process.fees,process.initial_balance)

    env.step(test_action)

    return env.Obseravtion_DataFrame.head(process.window_size+10)
