from Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from huggingface_hub import login, notebook_login

login(token="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")

from datasets import load_dataset, DatasetDict
traffic_dataset = DatasetDict()
traffic_dataset = load_dataset("tilos/IL2223_project") #read dataset from huggingface
#print(traffic_dataset)
traffic = traffic_dataset['train'].train_test_split(test_size=0.2, shuffle=True) #splite train and test

features = traffic.remove_columns(["congestionLevel"]) # features
target = traffic.remove_columns(['referenceTime', 't', 'ws', 'prec1h', 'fesn1h', 'vis', 'confidence']) #lable
print(features, "\n" ,target)


#X, y = shuffle(features.data, target.data, random_state=13)
#X, y = features.data, target.data
#X = X.astype(np.float32)
#offset = int(X.shape[0] * 0.9)

X_train, y_train = features["train"][['referenceTime', 't', 'ws', 'prec1h', 'fesn1h', 'vis', 'confidence']], target["train"]['congestionLevel']
X_test, y_test = features["test"].data, target["test"].data
#print(y_train)

#reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
#models, predictions = reg.fit(X_train, X_test, y_train, y_test)
#print(models)

print(X_train,"\n y train:\n",y_train)
model = LinearRegression()#copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model.fit(X_train,y_train)

predictions = model.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


print("done")

kwargs = {
    "dataset_tags": "Traffic data",
    "dataset": "Traffic and weather",  # a 'pretty' name for the training dataset
    "model_name": "Traffic Prediction",  # a 'pretty' name for our model
    "tasks": "Traffic Regression",
    "tags": "regression",
    }
model.push_to_hub(**kwargs)