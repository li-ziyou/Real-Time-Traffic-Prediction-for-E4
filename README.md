
# Real Time Traffic Prediction for E4 outside Kista

This service serves as project of group 46 of ID2223 Scalable Machine Learning
and Deep Learning course at KTH Royal Institute of Technology.

This service is a automatically-updated machine learning-based traffic prediction service
for [the motorway E4 entering Kista](https://goo.gl/maps/EXtaAHKBymb5w7P18). The service predicts
the clearness of E4 traffic based on the time of day and the weather condition on the road. The metric shown
is free freeflow level, which equals to the quotient of the actual speed and the freeflow speed, with
0 showing a total congestion and 1 showing a complete freeflow.

The service UI is built as a [Huggingface Space](https://huggingface.co/spaces/Chenzhou/Real_Time_Traffic_Prediction) with the inference pipeline built in.

## Feature Pipeline

The service captures real-time traffic data from the [traffic 
API from Tomtom](https://developer.tomtom.com/products/traffic-api) 
and the hourly updated weather data from the nearest weather station from the
[API from SMHI](https://opendata.smhi.se/apidocs/metanalys/index.html)
as features and stores them as 
[Huggingface Dataset](https://huggingface.co/datasets/tilos/IL2223_project). We selected the timestamp of retrieval, air temperature, wind speed, precipitation, visibility, confidence of traffic data and the conputed freeflow level as features.
To run the feature pipeline automatically, it is deployed on Modal and the features
are captured and stored every 10 minutes.

### Features Correlation Analysis
According to the correlation analysis of the current data, temperature and time have the largest correlation with the degree of congestion level. Traffic congestion often occurs in the afternoon of a day, which may be caused by people commuting. In addition, the current data show a negative correlation between traffic fluency and temperature. In other words, the higher the temperature, the less smooth the traffic. This is because people tend to drive less in cold temperatures (roads can be icy), avoiding congestion.

We believe that confidence of data is an important feature. Since the confidence of the data obtained so far is all 1 (perfect confidence), the result shown in the model is that the data confidence has nothing to do with traffic. We believe that more data in the future will change this relationship and therefore retain it.

The feature pipeline can be found [here](https://github.com/Tilosmsh/IL2223_project/blob/main/feature_pipeline.py).
## Training Pipeline

### Model Selection

Several machine learning models have been compared and AdaBoostRegression
is chosen as it showed best performance in terms of RMSE and R2 score. 

### Training

The training pipeline retrieves features stored on huggingface and
the model is retrained on the whole updated dataset, because incremental training 
(partial_fit function) of scikit-learn does not support AdaBoostRegression.

[The training pipeline](https://github.com/Tilosmsh/IL2223_project/blob/main/Training.py) is deployed on Modal and automatically retrains every day.
The trained model is then uploaded and stored on [huggingface](https://huggingface.co/tilos/Traffic_Prediction), which can finally be inferenced by the huggingface space directly.



## Problems Encountered

Several problems have been encountered during the completion of this project, they are mostly
because of trying to collect in a short time a large amount of real-life data that generally varies 
at the period of a year.

### Severely Imbalanced dataset

The dataset is severely imbalanced. For a vast majority of the time,
the traffic of E4 is very unsaturated and the free flow level stays 1. This may be 
due to that the project was done during a rather short and special period time at the end of the year
where people are on holiday, so the traffic is also few. We believe that, through a longer time period of self-updating, the model
can predict the traffic more accurately.

### Unchanged Weather

The weather data also mostly stayed the same at the beginning of the year. Our
analysis of correlation therefore showed that the weather contributed little to the variation of
the traffic, which does not seem logical. We also believe that this situation can be changed with
a larger amount of features.


## Authors

- [@Ziyou Li](https://www.github.com/Tilosmsh)
- [@Chenzhou Huang](https://github.com/Chenzhou98)



## Acknowledgements

 - [ID2223 @ KTH](https://id2223kth.github.io/)    
 - [Modal](modal.com)
 - [Hugging Face](huggingface.co)
 - [SMHI](https://www.smhi.se/q/Stockholm/2673730)
 

