import argparse
import pandas as pd
import pickle
import json
import yaml
import os
from sklearn.model_selection import train_test_split
from utils import load_config, load_data, compute_metrics
from scaling import apply_scaling
from model_registry import create_model

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Path to YAML config")
parser.add_argument("--exp_dir", required=True, help="Path to experiment folder")
args = parser.parse_args()

exp_dir = args.exp_dir

if args.config is not None:
    config = load_config(args.config)
else:
    config = load_config(os.path.join(exp_dir, "config.yaml"))

X, y = load_data(os.path.join(config['data']['data_dir'], config['data']['file_name']))

random_states=config["evaluation"]["random_states"]

if config["training"]["tune"]==True:
    
    with open(os.path.join(exp_dir, "tuned_params.json"), "r") as f:
        params = json.load(f)
else:
    params = config["training"]["parameters"]

metrics_df = pd.DataFrame({})
for n in range(config["evaluation"]["n_splits"]):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["evaluation"]["test_size"], random_state=random_states[n])

    x_scalers = config['preprocessing']['x_scaling']
    y_scalers = config['preprocessing']['y_scaling']

    X_train, X_test = apply_scaling(X_train, X_test, x_scalers)
    y_train, y_test = apply_scaling(y_train.reshape(-1,1), y_test.reshape(-1,1), y_scalers)
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    model = create_model(config["training"]["estimator"], params=params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics_dict = compute_metrics(y_test, y_pred)

    col_name = "split " + str(n)
    metrics_df[col_name] = [metrics_dict["RMSE"], metrics_dict["pearsonr"]]

metrics_df.index = ["RMSE", "pearsonr"]
metrics_df["mean"]=metrics_df.mean(axis=1)

metrics_df.to_csv(os.path.join(exp_dir, "evaluation.csv"), sep='\t')

    
