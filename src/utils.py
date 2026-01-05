import yaml
import os
import copy
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_up_experiment(config):
    base_name = config["experiment"]["name"]
    exp_id = str(hash_config(normalize_config(config)))
    folder_name = base_name + "_" + exp_id
    exp_dir = os.path.join("experiments", folder_name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    return exp_dir

def load_data(data_dir):
    gene = pd.read_csv(data_dir, delimiter='\t', header=0)  
    X = gene.iloc[:,1:-1]
    y = gene.iloc[:,-1]
    
    return X, pd.DataFrame(y)

def compute_metrics(y_test, y_pred):
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

    if len(np.unique(y_test))==1:
        print("pearsonr undefined: all test samples are equal")
        pearson_corr=float('NaN')
    elif len(np.unique(y_pred))==1:
        print("pearsonr undefined, all predicted samples are equal, value: "+str(y_pred[0]))
        pearson_corr=float('NaN')
    else:
        pearson_corr = pearsonr(y_test, y_pred)[0]

    return {"RMSE": RMSE, "pearsonr": pearson_corr}

def normalize_config(config):
    cfg = copy.deepcopy(config)
    cfg.pop("experiment", None)
    return cfg

def hash_config(config, length=8):
    canonical = json.dumps(config, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]

def plot_training_results(y_test, y_pred, exp_dir):
    fig, axs = plt.subplots(1,2, figsize=(12,4))

    axs[0].scatter(y_test, y_pred)
    axs[0].set_xlabel("y_test")
    axs[0].set_ylabel("y_pred")

    axs[1].scatter(np.arange(len(y_test)), y_test)
    axs[1].scatter(np.arange(len(y_pred)), y_pred)
    axs[1].set_xlabel("sample index")
    axs[1].set_ylabel("expression value")
    #axs[1].set_yscale("log")

    fig.savefig(os.path.join(exp_dir,"training_plots.png"), format='PNG')
    plt.clf

def plot_feature_importances(model, exp_dir):
    
    if hasattr(model, "coef_"):
        importances = model.coef_
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return None
    
    plt.plot(np.sort(importances))
    plt.xlabel('featues index')
    plt.ylabel('importance')

    plt.savefig(os.path.join(exp_dir,"feature_importance.png"), format='PNG')
    plt.clf

def custom_metric(y_test, y_pred):
    metrics = compute_metrics(y_test, y_pred)
    return -metrics["RMSE"]+metrics["pearsonr"]

def log2p1(X):
    return np.log2(np.asarray(X)+1)

def exp2m1(X):
    return 2**(np.asarray(X))-1