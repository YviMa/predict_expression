import argparse
import model_registry
import tuning
import pickle
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join
from utils import load_config, set_up_experiment, load_data, compute_metrics
from scaling import apply_scaling

# parse the yaml file 
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to YAML config")
args = parser.parse_args()

# load yaml config file with all the options you provided
config = load_config(args.config)

# creates separate directory for each experiment
exp_dir = set_up_experiment(config)

X, y = load_data(os.path.join(config['data']['data_dir'], config['data']['file_name']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

x_scalers = config['preprocessing']['x_scaling']
y_scalers = config['preprocessing']['y_scaling']

X_train, X_test = apply_scaling(X_train, X_test, x_scalers)
y_train, y_test = apply_scaling(y_train, y_test, y_scalers)
     

if config["training"]["tune"]=="true":

    estimator = model_registry.create_model(config["training"]["estimator"])

    Tuner=tuning.get_tuner(estimator, config["training"])
    model, best_params = Tuner.tune(X_train,y_train)

    with open(os.join(exp_dir,"tuned_params.pkl"), "wb") as f:
        pickle.dump(best_params, f)

else:
    try:
        params = config["training"]["parameters"]
    except Exception:
        raise ValueError("No valid parameters provided.")
    
    model = model_registry.create_model(config["training"]["estimator"], **params)
        
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# compute RMSE and pearson correlation
metrics_dict = compute_metrics(y_test, y_pred)

with open(os.join(exp_dir,"metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)



