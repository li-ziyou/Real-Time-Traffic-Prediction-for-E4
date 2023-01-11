import sklearn
import gradio as gr
import joblib
import pandas as pd
import datasets

title = "Stoclholm Highway E4 Real Time Traffic Prediction"
description = "Stockholm E4 (59°23'44.7"" N 17°59'00.4""E) highway real time traffic prediction, updated in every hour"

inputs = [gr.Dataframe(row_count = (1, "fixed"), col_count=(1,"fixed"), label="Input Data", interactive=1)]

outputs = [gr.Dataframe(row_count = (1, "fixed"), col_count=(1, "fixed"), label="Predictions", headers=["Congestion Level"])]

model = joblib.load("tilos/Traffic_Prediction/tree/main/traffic_model.pkl")

# we will give our dataframe as example
#df = datasets.load_dataset("merve/supersoaker-failures")
#df = df["train"].to_pandas()

def infer(input_dataframe):
  return pd.DataFrame(model.predict(input_dataframe))

gr.Interface(fn = infer, inputs = inputs, outputs = outputs).launch()