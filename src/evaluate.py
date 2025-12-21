import argparse
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from utils import load_config, load_data, compute_metrics
from model_registry import create_model

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to YAML config")
parser.add_argument("--exp_dir", required=True, help="Path to experiment folder")
args = parser.parse_args()

config = load_config(args.config)
exp_dir = args.exp_dir

X, y = load_data(config['data_dir'])

random_states=config["evaluation"]["random_states"]

if config["training"]["tune"]==True:
    with open(os.join(exp_dir, "tuned_params.pkl"), "r") as f:
        params = pickle.load(f)
else:
    params = config["training"]["parameters"]

metrics_df = pd.DataFrame({})
for n in config["evaluation"]["n_splits"]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["evaluation"]["test_size"], random_state=random_states[n])

    model = create_model(config["training"]["estimator"], **params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics_dict = compute_metrics(y_test, y_pred)

    col_name = "split " + str(n)
    metrics_df[col_name] = [metrics_dict["RMSE"], metrics_dict["pearsonr"]]

metrics_df.index = ["RMSE", "pearsonr"]
metrics_df["mean"]=metrics_df.mean(axis=0)

    
