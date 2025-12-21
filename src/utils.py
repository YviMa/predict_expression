import yaml
import os
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_up_experiment(config):
    base_name = config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{base_name}_{timestamp}"

    exp_dir = os.path.join("experiments", exp_id)
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

    return {"RMSE": RMSE, "pearson_corr": pearson_corr}