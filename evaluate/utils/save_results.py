import os
import yaml
import json


def save_effectiveness_score(effectiveness_score, pa, result_dir):
    
    if not os.path.exists(os.path.join(result_dir, f"effectiveness_do_{pa}.csv")):
        with open(os.path.join(result_dir, f"effectiveness_do_{pa}.csv"), "w") as f:
            f.write("do_parent, attribute, mean, std\n")

    with open(os.path.join(result_dir, f"effectiveness_do_{pa}.csv"), "a") as f:
        for attr, score in effectiveness_score.items():
            f.write(f"{pa}, {attr}, {score[0]}, {score[1]}\n")
            
            
def save_config(config_path, result_dir):

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(result_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    
def save_params(params, result_dir):
    with open(os.path.join(result_dir, "params.json"), "w") as f:
        json.dump(params, f)
    
            

        
    
    
    