import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import os
import modal

LOCAL=True

if LOCAL == False:

   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["requests", "huggingface_hub", "datetime", "datasets", "scikit-learn"]).apt_install(["libsndfile1"])
   @stub.function(image=image, schedule=modal.Period(hours=1), secret=modal.Secret.from_name("ScalableML_lab1"))
   def f():
       g()

def timeconvert(time):
    import dateutil.parser as dp
    timestamp = []
    for i in time:
        # print(i)
        i = dp.parse(i).timestamp()
        timestamp.append(i)
    # traffic_dataset.update({"timestamp": timestamp})
    return timestamp

def g():

      from huggingface_hub import login, notebook_login
      # Login to huggingface
      login(token="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")

      from datasets import load_dataset, DatasetDict
      #traffic_dataset = DatasetDict()
      traffic_dataset = load_dataset("tilos/IL2223_project") #read dataset from huggingface
      #print(traffic_dataset)


      traffic = traffic_dataset['train'].train_test_split(test_size=0.2, shuffle=True) #splite train and test

      features = traffic.remove_columns(["congestionLevel"]) # features
      target = traffic.remove_columns(['referenceTime', 't', 'ws', 'prec1h', 'fesn1h', 'vis', 'confidence']) #lable
      print(features, "\n" ,target)

      X_train_df = pd.DataFrame.from_dict(features["train"])
      y_train_df = pd.DataFrame.from_dict(target["train"])
      X_test_df  = pd.DataFrame.from_dict(features["test"])
      y_test_df  = pd.DataFrame.from_dict(target["test"])

      #datetime convert to timestamp
      timestamp_train = timeconvert(X_train_df['referenceTime'])
      X_train_df = X_train_df.drop(columns='referenceTime')
      X_train_df['referenceTime'] = timestamp_train

      timestamp_test = timeconvert(X_test_df['referenceTime'])
      X_test_df = X_test_df.drop(columns='referenceTime')
      X_test_df['referenceTime'] = timestamp_test

      #set train and test set for training
      X_train, y_train = X_train_df[[ 'referenceTime','t', 'ws', 'prec1h', 'fesn1h', 'vis', 'confidence']], y_train_df['congestionLevel'] #ref time
      X_test, y_test = X_test_df[[ 'referenceTime','t', 'ws', 'prec1h', 'fesn1h', 'vis', 'confidence']], y_test_df['congestionLevel']

      print(X_train,"\n",y_train)

      # Multi regression
      from Supervised import LazyRegressor
      reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
      models, predictions = reg.fit(X_train, X_test, y_train, y_test)
      print(models)

      # Single reg
      model = LinearRegression()
      model.fit(X_train,y_train)

      predictions = model.predict(X_test)

      #plt.scatter(y_test,predictions)
      #plt.xlabel('Y Test')
      #plt.ylabel('Predicted Y')

      from sklearn import metrics
      print('MAE:', metrics.mean_absolute_error(y_test, predictions))
      print('MSE:', metrics.mean_squared_error(y_test, predictions))
      print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


      # Save the model to huggingface
      import pickle
      filename = 'traffic_model.pkl'
      pickle.dump(model, open(filename, 'wb'))

      #from huggingface_hub import create_repo
      #create_repo(repo_id="Traffic_Prediction")

      from huggingface_hub import upload_file
      upload_file(
         path_or_fileobj="traffic_model.pkl",
         path_in_repo="traffic_model.pkl",
         repo_id="tilos/Traffic_Prediction"
      )

      print("done")


if __name__ == "__main__":
   if LOCAL == True:
      g()
   else:
      with stub.run():
         f()