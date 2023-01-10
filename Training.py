from Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from huggingface_hub import login, notebook_login

login(token="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")


from datasets import load_dataset, DatasetDict
traffic = DatasetDict()
traffic = load_dataset("tilos/?") #read dataset from huggingface

X, y = shuffle(traffic.data, traffic.target, random_state=13)
X = X.astype(np.float32)

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)

kwargs = {
    "dataset_tags": "Traffic data",
    "dataset": "Traffic and weather",  # a 'pretty' name for the training dataset
    "model_name": "Traffic Prediction",  # a 'pretty' name for our model
    "tasks": "Traffic Regression",
    "tags": "regression",
    }
models.push_to_hub(**kwargs)