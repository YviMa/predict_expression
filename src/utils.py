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
from scipy.stats import pearsonr, randint, uniform, loguniform

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
    if isinstance(X, pd.Series):
        X_ = np.asarray(X)
        X_ = np.log2(X_+1)
        logged = pd.Series(X_, name=X.name, index=X.index)
    elif isinstance(X, np.ndarray):
        X_ = X
        logged = np.log2(X_+1)
    else:
        raise TypeError("Wrong data type, must be pd.Series or numpy array.")
    return logged

def exp2m1(X):
    if isinstance(X, pd.Series):
        X_ = np.asarray(X)
        X_ = np.exp2(X_)-1
        expd = pd.Series(X_, name=X.name, index=X.index)
    elif isinstance(X, np.ndarray):
        X_ = X
        expd = np.exp2(X_)-1
    else:
        raise TypeError("Wrong data type, must be pd.Series or numpy array.")
    return expd

def split_params(params_grid):
    model_params = {}
    trainer_params = {}
    optimizer_params = {}

    for key in params_grid.keys():
        subkeys = key.split("__")
        if subkeys[0] == "model_params":
            model_params.update({subkeys[1]: params_grid[key]})
        if subkeys[0] == "trainer_params":
            trainer_params.update({subkeys[1]: params_grid[key]})
        if subkeys[0] == "optimizer_params":
            optimizer_params.update({subkeys[1]: params_grid[key]})
    
    return model_params, trainer_params, optimizer_params

def split_sk_params(params_grid):
    model_params = {}
    selector_params = {}
    selector_estimator_params = {}

    for key in params_grid.keys():
        subkeys = key.split("__", 1)
        #print(subkeys)
        if subkeys[0] == "model_config":
            model_params.update({subkeys[1]: params_grid[key]})
        if subkeys[0] == "selector_config":
            selector_params.update({subkeys[1]: params_grid[key]})
        if subkeys[0] == "selector_estimator_config":
            selector_estimator_params.update({subkeys[1]: params_grid[key]})
    
    return model_params, selector_params, selector_estimator_params

def setup_param_dist(param_grid):
    param_dist = {}

    for key in param_grid.keys():
        param = param_grid[key]
        if param["type"] == "uniform":
            dist = uniform(param["min"], param["max"])
        elif param["type"]== "loguniform":
            dist = loguniform(param["min"], param["max"])
        elif param["type"]== "randint":
            dist = randint(param["min"], param["max"]+1)
        else:
            raise ValueError("Not a valid distribution")
        param_dist.update({key: dist})
    
    return param_dist
