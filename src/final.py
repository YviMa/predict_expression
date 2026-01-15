"""
Script to load trained models per gene and make predictions on test data.
Output is saved as TSV files: pred_gene1.tsv, pred_gene2.tsv, pred_gene3.tsv
DISCLAIMER: Please change the paths to filenames and/or filenames if necessary!
"""

import os
import types
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.feature_selection import SelectFromModel, VarianceThreshold


#---------------------------------------------------------
# Utility
#---------------------------------------------------------

def log2p1(X):
    return np.log2(X + 1)

def exp2m1(X):
    return np.exp2(X) - 1

utils = types.ModuleType("utils")

utils.log2p1 = log2p1
utils.exp2m1 = exp2m1

sys.modules["utils"] = utils

class Log2Scaler(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return log2p1(X)

    def inverse_transform(self, X):
        return exp2m1(X)

# Classes 

class ClassifierGuidedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, classifier=None, regressor=None):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y):
        self.classifier.fit(X, (y > 0).astype(int))
        mask = y > 0
        if np.any(mask):
            self.regressor.fit(X[mask], y[mask])
        return self

    def predict(self, X):
        labels = self.classifier.predict(X)
        preds = np.zeros(len(X))
        mask = labels == 1
        if np.any(mask):
            preds[mask] = self.regressor.predict(X[mask])
        return preds

class SelectFromTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        return self.estimator.fit(X, y)

    def transform(self, X):
        return self.estimator.transform(X)


#---------------------------------------------------------
# Main script
#---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to test data (TSV)")
    parser.add_argument("--gene", type=int, required=True, choices=[1,2,3], help="Gene number (1,2,3)")
    parser.add_argument("--output", required=True, help="Output TSV file for predictions")
    args = parser.parse_args()

    #Load test data
    test_df = pd.read_csv(args.input, sep='\s+')
    
    #Extract IDs and features
    ids = test_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]

    # need to be changed for 2 and 3
    model_files = {
        1: "experiments/lassolars/lassolars_agg_gene_1_52bbfb3d/model.joblib",
        2: "models/model_gene2.joblib",
        3: "models/model_gene3.joblib"
    }

    model_file = model_files[args.gene]
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    print(f"Loading model: {model_file}")
    model = joblib.load(model_file)

    #Predict
    y_pred = model.predict(X_test)
    
    out_df = pd.DataFrame({"ID": ids, "Expression": y_pred})
    out_df.to_csv(args.output, sep='\t', index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
