import argparse
import tuning
import matplotlib.pyplot as plt
import json
import os
from scaling import ScalerWrapper
from os.path import join
from utils import load_config, set_up_experiment, load_data
from feature_selection import get_feature_selector

# parse the yaml file 
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to YAML config")
args = parser.parse_args()

# load yaml config file with all the options you provided
config = load_config(args.config)

# creates separate directory for each experiment
exp_dir = set_up_experiment(config)

data_path = join(config['data']['data_dir'])
X, y = load_data(data_path)

x_scalers = config['preprocessing']['x_scaling']
y_scalers = config['preprocessing']['y_scaling']

X_scaler = ScalerWrapper(x_scalers)
Y_scaler = ScalerWrapper(y_scalers)

X = X_scaler.fit_transform(X)
y = Y_scaler.fit_transform(y)

if config["preprocessing"]["feature_selection"]["apply"]==True:
    feature_selector = get_feature_selector(X,y,config["preprocessing"]["feature_selection"])
    X = feature_selector.fit_transform(X, y)

tuner_name = config["tuning"]["tuner"]
tuning_config = config["tuning"]["tuning_config"]
Tuner=tuning.get_tuner(tuner_name, tuning_config)
model, best_params = Tuner.tune(X,y)

with open(os.path.join(exp_dir,"tuned_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)


