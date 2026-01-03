import argparse
import model_registry
import tuning
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join
from utils import load_config, set_up_experiment, load_data, compute_metrics, plot_training_results
from scaling import apply_scaling
from feature_selection import get_feature_selector

# parse the yaml file 
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to YAML config")
args = parser.parse_args()

# load yaml config file with all the options you provided
config = load_config(args.config)

# creates separate directory for each experiment
exp_dir = set_up_experiment(config)

X, y = load_data(os.path.join(config['data']['data_dir'], config['data']['file_name']))

x_scalers = config['preprocessing']['x_scaling']
y_scalers = config['preprocessing']['y_scaling']

X, _ = apply_scaling(X, X, x_scalers)
y, _ = apply_scaling(y.reshape(-1,1), y.reshape(-1,1), y_scalers)
y = y.ravel()
y = y.ravel()

if config["preprocessing"]["feature_selection"]["apply"]==True:
    feature_selector = get_feature_selector(X,y,config["preprocessing"]["feature_selection"])
    X = feature_selector.fit_transform(X, y)

if config["training"]["tune"]==True:

    estimator = model_registry.create_model(config["training"]["estimator"])

    Tuner=tuning.get_tuner(config["training"]["tuner"], {"estimator": estimator, "config":config["training"]})
    model, best_params = Tuner.tune(X,y)
    
    with open(os.path.join(exp_dir,"tuned_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

else:
    try:
        params = config["training"]["parameters"]
    except Exception:
        raise ValueError("No valid parameters provided.")
    
    model = model_registry.create_model(config["training"]["estimator"], **params)
        
    model.fit(X, y)



