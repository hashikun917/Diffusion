import numpy as np

import yaml
import argparse

    
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def normalize(array: np.ndarray) -> np.ndarray:
    
    arrays = []
    for i in range(array.shape[1]):
        min_val = np.min(array[:, i])
        max_val = np.max(array[:, i])
        temp = (array[:, i] - min_val) / (max_val - min_val)
        scaled_array = 2 * temp - 1
        arrays.append(scaled_array)
    return np.array(arrays).T