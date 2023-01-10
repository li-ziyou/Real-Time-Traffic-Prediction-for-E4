from Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import login, notebook_login

login(token="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")


from datasets import load_dataset, DatasetDict
traffic_dataset = DatasetDict()
traffic_dataset = load_dataset("tilos/IL2223_project") #read dataset from huggingface
#print(traffic_dataset)
traffic = traffic_dataset['train'].train_test_split(test_size=0.2, shuffle=True) #splite train and test

target = traffic.remove_columns(['referenceTime', 't', 'ws', 'prec1h', 'fesn1h', 'vis', 'confidence', 'congestionLevel']) #lable
features = traffic.remove_columns(["congestionLevel"]) # features
print(features, "\n",target)


#X, y = shuffle(features.data, target.data, random_state=13)
#X, y = features.data, target.data
#X = X.astype(np.float32)
#offset = int(X.shape[0] * 0.9)

X_train, y_train = features["train"].data, target["train"].data
X_test, y_test = features["test"].data, target["test"].data

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)
print("done")

kwargs = {
    "dataset_tags": "Traffic data",
    "dataset": "Traffic and weather",  # a 'pretty' name for the training dataset
    "model_name": "Traffic Prediction",  # a 'pretty' name for our model
    "tasks": "Traffic Regression",
    "tags": "regression",
    }
models.push_to_hub(**kwargs)