import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt

import os
import modal

LOCAL=False

if LOCAL == False:

   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["requests", "huggingface_hub", "datetime", "datasets", "scikit-learn", "lazypredict", "matplotlib", "seaborn"]).apt_install(["libsndfile1"])
   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("ScalableML_lab1"))
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

def corr_analysis(traffic_dataset):
    # correlation analysis
    import seaborn as sns
    datacorr = pd.DataFrame.from_dict(traffic_dataset['train'])
    timestamp_corr = timeconvert(datacorr['referenceTime'])
    datacorr = datacorr.drop(columns='referenceTime')
    datacorr['referenceTime'] = timestamp_corr
    #print(datacorr)

    plt.subplots(figsize=(12, 8))
    sns.heatmap(datacorr.corr(), cmap='RdGy')
    plt.show()

def g():

      from huggingface_hub import login, notebook_login
      # Login to huggingface
      login(token="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")

      from datasets import load_dataset, DatasetDict
      traffic_dataset = DatasetDict()
      traffic_dataset = load_dataset("tilos/IL2223_project") #read dataset from huggingface
      #print(traffic_dataset)


      traffic = traffic_dataset['train'].train_test_split(test_size=0.2, shuffle=True) #splite train and test

      corr_analysis(traffic_dataset)

      # Time-Congestion (TC) plot overview
      TC_time_df = pd.DataFrame.from_dict(traffic_dataset['train'])
      timestamp_TC = timeconvert(TC_time_df['referenceTime'])
      TC_time_df = TC_time_df.drop(columns='referenceTime')
      TC_time_df['referenceTime'] = timestamp_TC
      x_TC = TC_time_df['referenceTime']
      y_TC = pd.DataFrame.from_dict(traffic_dataset['train'])['congestionLevel']
      plt.plot(x_TC,y_TC)
      plt.show()



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
      X_train, y_train = X_train_df[[ 'referenceTime',"t",'ws', 'prec1h', 'fesn1h', 'vis', 'confidence']], y_train_df['congestionLevel'] #ref time
      X_test, y_test = X_test_df[[ 'referenceTime',"t",'ws', 'prec1h', 'fesn1h', 'vis', 'confidence']], y_test_df['congestionLevel']

      print(X_train,"\n",y_train)



      # Multi regression
      from lazypredict.Supervised import LazyRegressor

      reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
      models, predictions = reg.fit(X_train, X_test, y_train, y_test)
      print(models)


      # Single reg
      model = AdaBoostRegressor()
      model.fit(X_train,y_train)

      y_predictions = model.predict(X_test)
      print("True:\n ",y_test,"\n","Predict:\n ", y_predictions)

      # Plot scatter

      plt.scatter(y_test,y_predictions)
      plt.xlabel('Y Test')
      plt.ylabel('Predicted Y')
      plt.show()

      from sklearn import metrics
      print('MAE:', metrics.mean_absolute_error(y_test, y_predictions))
      print('MSE:', metrics.mean_squared_error(y_test, y_predictions))
      print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predictions)))
      print('R2 score:', model.score(X_test, y_test))
      print('R2 score:', metrics.r2_score(y_test, y_predictions))

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