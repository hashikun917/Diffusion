import torch
import numpy as np
from typing import Dict, List
from json import load
# from model import SCM
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import argparse
import random
from typing import Union, List
import yaml
from pipeline.utils import dict2namespace
from datetime import datetime
from pathlib import Path
import sys
sys.path.append("../../")

from evaluate.anticausal.classifiers.classifier import Classifier
# from models.classifiers.celeba_classifier import CelebaClassifier
# from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier
# from models.classifiers.adni_classifier import ADNIClassifier
from dataset.morphomnist import MorphoMNISTLike2, load_morphomnist_like
# from datasets.celeba.dataset import Celeba
# from datasets.adni.dataset import ADNI
from evaluate.utils.transforms import get_attribute_ids, ReturnDictTransform
from evaluate.utils.load_utils import load_models

# from evaluation.metrics.composition import composition
# from evaluation.metrics.minimality import minimality
# from evaluation.embeddings.embeddings import get_embedding_model, get_embedding_fn
# from evaluation.metrics.fid import fid
from evaluate.metrics.effectiveness import effectiveness
# from evaluation.metrics.utils import save_selected_images, save_plots
from dataset.morphomnist import unnormalize as unnormalize_morphomnist
# from datasets.celeba.dataset import unnormalize as unnormalize_celeba
# from datasets.adni.dataset import unnormalize as unnormalize_adni

from pipeline.causal_pipeline import CausalPipeline
from diffusion.diffusion_model.models.model11 import UNet
from diffusion.utils.utils import load_json
from baselines.diffscm.diffscm import DiffusionSCM
from evaluate.utils.save_results import save_effectiveness_score, save_config, save_params
from evaluate.utils.parser_utils import str2bool


torch.multiprocessing.set_sharing_strategy('file_system')

rng = np.random.default_rng()

dataclass_mapping = {
    "morphomnist": (MorphoMNISTLike2, unnormalize_morphomnist),
    # "celeba": (Celeba, unnormalize_celeba),
    # "adni": (ADNI, unnormalize_adni)
}

# MIN_MAX = {
#     "thickness": [0.82152224, 6.384839],
#     "intensity": [66.48045, 254.93214],
#     "slant": [-43.692436, 66.94711],
#     "width": [10.000215, 24.999382],
#     "image": [0.0, 255.0]
# }

# QUARTILE_RANGES = {
#     'thickness': [2.04805145, 2.881358275],
#     'intensity': [117.097564, 197.4233375],
#     'slant': [-19.0626455, -2.3332007],
#     'width': [11.03513725, 19.5581495]
# }

def different_value(possible_values, value, bins, attribute):
    if bins is not None and attribute in bins:
        return np.digitize(possible_values, bins[attribute]) != np.searchsorted(bins[attribute], value)
    else:
        return possible_values != value

def produce_counterfactuals(factual_batch: Dict, scm: Union[nn.Module, CausalPipeline], do_parent: str, intervention_source: Dataset, force_change: bool = False, possible_values = None, device: str = 'cuda', bins = None):
    
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}

    #update with the counterfactual parent
    if force_change:
        possible_values = possible_values[do_parent]
        values = factual_batch[do_parent].cpu()
        if do_parent not in ["digit", "apoE", "slice"]:
            interventions = {do_parent: torch.cat([torch.tensor(np.random.choice(possible_values[different_value(possible_values, value, bins, do_parent)])).unsqueeze(0)
                                                for value in values]).view(-1).unsqueeze(1).to(device)}
        else:
            interventions = {do_parent: torch.cat([torch.tensor(rng.choice(possible_values[torch.where((different_value(possible_values, value, bins, do_parent)).any(dim=1))], axis=0)).unsqueeze(0)
                                                for value in values]).to(device)}
    else:
        batch_size, _ , _ , _ = factual_batch["image"].shape
        idxs = torch.randperm(len(intervention_source))[:batch_size] # select random indices from train set to perform interventions

        interventions = {do_parent: torch.cat([intervention_source[id][do_parent] for id in idxs]).view(-1).unsqueeze(1).to(device)
                        if do_parent not in ["digit", "apoE", "slice"] else torch.cat([intervention_source[id][do_parent].unsqueeze(0).to(device) for id in idxs])}
        

    if isinstance(scm, nn.Module):
        abducted_noise = scm.encode(**factual_batch)
        counterfactual_batch = scm.decode(interventions, **abducted_noise)
    elif isinstance(scm, CausalPipeline):
        diffused_noise = scm.diffuse(factual_batch)
        counterfactual_batch = scm.denoise(interventions, diffused_noise)
    elif isinstance(scm, DiffusionSCM):
        diffused_noise = scm.diffuse(factual_batch)
        counterfactual_batch = scm.denoise(factual_batch, interventions, diffused_noise)
        
    return counterfactual_batch
        
def evaluate_effectiveness(test_set: Dataset, unnormalize_fn, batch_size:int , scm: Union[nn.Module, CausalPipeline], attributes: List[str], do_parent:str,
                           intervention_source: Dataset, predictors: Dict[str, Classifier], dataset: str):

    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    effectiveness_scores = {attr_key: [] for attr_key in attributes}
    for factual_batch in tqdm(test_data_loader):
        
        # if isinstance(scm, nn.Module):
        counterfactuals = produce_counterfactuals(factual_batch, scm, do_parent, intervention_source,
                                                force_change=True, possible_values=test_set.possible_values, bins=test_set.bins)
        # elif isinstance(scm, CausalPipeline):
        #     ### いずれは元々の実装のintervention_source（possible_valuesからとってくるor訓練セットからとってくる）に変えるべき ###
        #     # intervention_source = {atr: np.random.uniform(MIN_MAX[atr][0], MIN_MAX[atr][1], size=factual_batch['image'].shape[0]).tolist() for atr in attribute_size.keys()}
        #     intervention_source = {atr: np.random.uniform(QUARTILE_RANGES[atr][0], QUARTILE_RANGES[atr][1], size=factual_batch['image'].shape[0]).tolist() for atr in attribute_size.keys()}
        #     counterfactuals = scm.produce_counterfactuals(factual_batch, do_parent, intervention_source, w=w)
            
        e_score = effectiveness(counterfactuals, unnormalize_fn, predictors, dataset)

        for attr in attributes:
            effectiveness_scores[attr].append(e_score[attr])

    effectiveness_score = {key  : (round(np.mean(score), 3), round(np.std(score), 3)) for key, score in effectiveness_scores.items()}

    print(f"Effectiveness score do({do_parent}): {effectiveness_score}")

    return effectiveness_score


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    ### configは自分の研究用のものを使用しているためいずれbenchmarkに合わせて修正したい ###
    parser.add_argument("--config", '-c', type=str, help="Config file for experiment.", default="config/config_all.yaml")
    parser.add_argument("--classifier-config", '-clf', type=str, help="Classifier config file.", default="evaluate/anticausal/configs/morphomnist/classifier.json")
    parser.add_argument("--metrics", '-m',
                        nargs="+", type=str,
                        help="Metrics to calculate. "
                        "Choose one or more of [composition, effectiveness, fid, minimality]. If not set, all metrics are calculated.",
                        choices=["composition", "effectiveness", "fid", "minimality"],
                        default=["composition", "effectiveness", "fid", "minimality"])
    # parser.add_argument("--cycles", '-cc', nargs="+", type=int, help="Composition cycles.", default=[1, 10])
    # parser.add_argument("--qualitative", '-qn', type=int, help="Number of qualitative results to produce", default=20)
    # parser.add_argument("--show-difference", '-sd', action='store_true', help="Show counterfactual-factual difference on qualitative results")
    # parser.add_argument("--embeddings", type=str, choices=["vgg", "clfs", "vae", "lpips", "clip"], help="What embeddings to use for composition metric. "
    #                     "Supported: [vgg, clfs, vae, lpips, clip]. If not set, will compute distance on image space")
    # parser.add_argument("--sampling-temperature", '-temp', type=float, default=0.1, help="Sampling temperature, used for VAE, HVAE.")
    
    ### 自分の研究用に追加したもの ###
    parser.add_argument("--data-dir", type=str, help="Dataset directory.", default="/home/hashikami/datadrive/morphomnist_all_model")
    parser.add_argument("--scm-type", type=str, help="The type of SCM (ours or diffscm or baseline)", default="ours")
    parser.add_argument("--result-dir", type=str, help="Result directory.", default="results/evaluate_counterfactual/effectiveness")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # torch.manual_seed(42)

    assert os.path.isfile(args.classifier_config), f"{args.classifier_config} is not a file"
    with open(args.classifier_config, 'r') as f:
        config_cls = load(f)

    # assert os.path.isfile(args.config), f"{args.config} is not a file"
    # with open(args.config, 'r') as f:
    #     config = load(f)
    
    with open(args.config, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config_dict)

    now = datetime.now().strftime('%Y-%m-%d_%H')
    result_dir = os.path.join(args.result_dir, now)
    os.makedirs(result_dir, exist_ok=True)

    # dataset = config["dataset"]
    # attribute_size = config["attribute_size"]
    
    dataset = config.image_data.name
    attribute_size = load_json(config.image_data.meta_data.attribute_size_path)

    batch_size = config.evaluate.batch_size

    if args.scm_type == "ours":
        scm = CausalPipeline(config)
        scm.prepare_models()
    
    elif args.scm_type == "diffscm":
        scm = DiffusionSCM(config)
        scm.prepare_models()
    
    elif args.scm_type == "baseline":
        
        causal_graph = load_json(config.image_data.meta_data.graph_path)
        models = load_models(config_dict, attribute_size)
        scm = SCM(checkpoint_dir=config.checkpoint_dir,
                  graph_structure=causal_graph,
                  temperature=config.sampling_temperature,
                  **models)

    data_class, unnormalize_fn = dataclass_mapping[dataset]

    transform = ReturnDictTransform(attribute_size)

    data_dir = args.data_dir
    train_set = data_class(attribute_size, split='train', transform=transform, data_dir=data_dir)
    test_set = data_class(attribute_size, split='test', transform=transform, data_dir=data_dir)
    
    
    # if args.qualitative > 0:
    #     produce_qualitative_samples(dataset=test_set, scm=scm, parents=list(attribute_size.keys()),
    #                                 intervention_source=train_set, unnormalize_fn=unnormalize_fn, num=args.qualitative,
    #                                 show_difference=args.show_difference)

    # embedding_model = get_embedding_model(args.embeddings, pretrained_vgg=True, classifier_config=args.classifier_config)
    # embedding_fn = get_embedding_fn(args.embeddings, unnormalize_fn, embedding_model)

    # if "composition" in args.metrics:
    #     evaluate_composition(test_set, unnormalize_fn, batch_size, cycles=args.cycles, scm=scm, embedding=args.embeddings, embedding_fn=embedding_fn)


    if "effectiveness" in args.metrics:
        if dataset == "morphomnist":
            predictors = {atr: Classifier(attr=atr, num_outputs=attribute_size[atr], context_dim=len(list(config_cls["anticausal_graph"][atr])))
                          for atr in attribute_size.keys()}
        # elif dataset == "celeba":
        #     if sum(attribute_size.values()) == 4:
        #         predictors = {atr: CelebaComplexClassifier(attr=atr, context_dim=len(list(config_cls["anticausal_graph"][atr])),
        #                                           num_outputs=config_cls[atr +"_num_out"],
        #                                           lr=config_cls["lr"], version=config_cls["version"]) for atr in attribute_size.keys()}
        #     else:
        #         predictors = {atr: CelebaClassifier(attr=atr, num_outputs=config_cls[atr +"_num_out"], lr=config_cls["lr"]) for atr in attribute_size.keys()}
        # else:
        #     attribute_ids = get_attribute_ids(attribute_size)
        #     predictors = {atr: ADNIClassifier(attr=atr, num_outputs=config_cls["attribute_size"][atr], children=config_cls["anticausal_graph"][atr],
        #                                       num_slices=config_cls["attribute_size"]['slice'], attribute_ids=attribute_ids, arch=config_cls['arch']) for atr in attribute_size.keys()}

        # load checkpoints of the predictors
        for key , cls in predictors.items():
            file_name = next((file for file in os.listdir(config_cls["ckpt_path"]) if file.startswith(key)), None)
            cls.load_state_dict(torch.load(config_cls["ckpt_path"] + file_name , map_location=torch.device('cuda'))["state_dict"])
            cls.to('cuda')

        for pa in attribute_size.keys():
            effectiveness_score = evaluate_effectiveness(test_set, unnormalize_fn, batch_size, scm, attributes=list(attribute_size.keys()), do_parent=pa,
                           intervention_source=train_set, predictors=predictors, dataset=dataset)

            save_effectiveness_score(effectiveness_score, pa, result_dir)
            
        params = {'metrics': args.metrics, 'dataset': dataset, 'scm_type': args.scm_type}
        save_params(params, result_dir)
        save_config(args.config, result_dir)
            

    # if "fid" in args.metrics:
    #     feat_dict = evaluate_fid(real_set=train_set, test_set=test_set, batch_size=batch_size, scm=scm, attributes=list(attribute_size.keys()))

    # if "minimality" in args.metrics:
    #     evaluate_minimality(real_set=train_set, test_set=test_set, batch_size=batch_size, scm=scm, attributes=list(attribute_size.keys()),
    #                         embedding=args.embeddings, embedding_fn=embedding_fn)