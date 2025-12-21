import yaml
import os
import copy
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_up_experiment(config):
    base_name = config["experiment"]["name"]
    exp_id = str(hash_config(normalize_config(config)))
    folder_name = config["experiment"]["name"] + "_" + exp_id
    exp_dir = os.path.join("experiments", folder_name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    return exp_dir

def load_data(data_dir):
    gene = pd.read_csv(data_dir, delimiter='\t', header=0)
    X = gene.iloc[:,1:-1].to_numpy()
    y = gene.iloc[:,-1].to_numpy()
    
    return X, y

def compute_metrics(y_test, y_pred):
    RMSE = mean_squared_error(y_test, y_pred)
    pearson_corr = pearsonr(y_test, y_pred)

    return {"RMSE": RMSE, "pearsonr": pearson_corr[0]}

def normalize_config(config):
    cfg = copy.deepcopy(config)
    cfg.pop("experiment", None)
    cfg.pop("data", None)
    return cfg

def hash_config(config, length=8):
    canonical = json.dumps(config, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]
