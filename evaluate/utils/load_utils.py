from importlib import import_module


    
def load_models(config_dict, attribute_size):
    
    models = {}
    for variable in config_dict["causal_graph"].keys():
        if variable not in config_dict["mechanism_models"]:
            continue
        model_config = config_dict["mechanism_models"][variable]

        module = import_module(model_config["module"])
        model_class = getattr(module, model_config["model_class"])
        model = model_class(params=model_config["params"], attr_size=attribute_size)

        models[variable] = model
        if "finetune" in model_config["params"] and model_config["params"]["finetune"] == 1:
            model.name += '_finetuned'
            
    return models