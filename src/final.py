"""
Script to load trained models per gene and make predictions on test data.
Output is saved as TSV files: pred_gene1.tsv, pred_gene2.tsv, pred_gene3.tsv
DISCLAIMER: Please change the paths to filenames and/or filenames if necessary!
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.base import TransformerMixin

#----------------------------
# Scaling utilities
#----------------------------
def log2p1(X):
    return np.log2(X + 1)

def exp2m1(X):
    return np.exp2(X) - 1

class Log2Scaler(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return log2p1(X)
    def inverse_transform(self, X):
        return exp2m1(X)

#----------------------------
#Dummy model for testing
#----------------------------
class ConstantDummyModel(BaseEstimator, RegressorMixin):
    """Dummy model that always predicts the same value"""
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self
    def predict(self, X):
        return np.full(len(X), self.mean_)

#----------------------------
# Feature selection utilities
#----------------------------
FEATURE_SELECTION_REGISTRY = {
    "variance_threshold": VarianceThreshold,
    "scikitlearn_estimator": SelectFromModel
}

def get_feature_selector(name, estimator=None, params=None):
    if name not in FEATURE_SELECTION_REGISTRY:
        raise ValueError(f"Unknown feature selector: {name}")
    if params is None:
        params = {}
    return FEATURE_SELECTION_REGISTRY[name](estimator=estimator, **params)

#----------------------------
# Pipeline builder
#----------------------------
def build_pipeline(model_file, selector_name=None, selector_params=None, estimator=None):
    steps = []
    steps.append(('scaler', Log2Scaler()))

    if selector_name is not None:
        selector = get_feature_selector(name=selector_name,
                                        estimator=estimator,
                                        params=selector_params)
        steps.append(('selector', selector))

    steps.append(('regressor', estimator))
    pipe = Pipeline(steps)
    return pipe

#----------------------------
# Main
#----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to test data (TSV)")
    parser.add_argument("--gene", type=int, required=True, choices=[1,2,3], help="Gene number (1,2,3)")
    parser.add_argument("--output", required=True, help="Output TSV file for predictions")
    args = parser.parse_args()

    # Load test data
    test_df = pd.read_csv(args.input, sep='\t')
    ids = test_df.iloc[:, 0] 
    X_test = test_df.iloc[:, 1:]

    # Map gene number to model file -> change to real models
    model_files = {
        1: "models/model_gene1.joblib",
        2: "models/model_gene2.joblib",
        3: "models/model_gene3.joblib"
    }
    model_file = model_files[args.gene]

    estimator = joblib.load(model_file)

    # Feature selection -> change
    selector_name = None
    selector_params = None

    pipe = build_pipeline(model_file=model_file,
                          selector_name=selector_name,
                          selector_params=selector_params,
                          estimator=estimator)

    # Predict
    y_pred = pipe.predict(X_test)
    y_pred = exp2m1(y_pred) 

    out_df = pd.DataFrame({"ID": ids, "Expression": y_pred})
    out_df.to_csv(args.output, sep='\t', index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()

