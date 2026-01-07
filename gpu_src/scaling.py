from utils import log2p1, exp2m1
from cuml.preprocessing import StandardScaler
from cuml.pipeline import Pipeline
from cuml.preprocessing import StandardScaler, FunctionTransformer

SCALER_REGISTRY = {
    "standard": StandardScaler
}

def get_scaler(name, params={}):
    cls = SCALER_REGISTRY[name]
    return cls(**params)

def build_scaler(scaler_list, params):
    steps=[]
    for name in scaler_list:
        if name in params:
            scaler_params = params[name]
        else:
            scaler_params = {}
        steps.append((name, get_scaler(name, scaler_params)))
    return Pipeline(steps)


def get_log2_transformer(**kwargs):
    return FunctionTransformer(func=log2p1, inverse_func=exp2m1, **kwargs)

SCALER_REGISTRY["log2"] = get_log2_transformer
