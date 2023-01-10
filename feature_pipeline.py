import os
import modal

LOCAL=True

if LOCAL == False:

   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["requests","csv", "huggingface_hub", "json", "datetime", "datasets"]).apt_install(["libsndfile1"])
   @stub.function(image=image, schedule=modal.Period(hours=1), secret=modal.Secret.from_name("ScalableML_lab1"), timeout=5000)
   def f():
       g()

def g():

    from huggingface_hub import login, notebook_login
    from datasets import load_dataset, Dataset
    import requests
    import os.path
    import csv
    import json
    import time
    from datetime import datetime

    # Login to huggingface
    login(token="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")
    notebook_login()

    # Get traffic data of E4 entering KTH Kista from tomtom API, updated in real time

    response_tomtom = requests.get(
                'https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key=azGiX8jKKGxCxdsF1OzvbbWGPDuInWez&point=59.39575,17.98343')
    json_response_tomtom = json.loads(response_tomtom.text)  # get json response

    currentSpeed = json_response_tomtom["flowSegmentData"]["currentSpeed"]
    freeFlowSpeed = json_response_tomtom["flowSegmentData"]["freeFlowSpeed"]
    congestionLevel = currentSpeed/freeFlowSpeed

    confidence = json_response_tomtom["flowSegmentData"]["confidence"] # Reliability of the traffic data, by percentage


    # Get weather data from SMHI, updated hourly

    response_smhi = requests.get(
                'https://opendata-download-metanalys.smhi.se/api/category/mesan1g/version/2/geotype/point/lon/17.983/lat/59.3957/data.json')
    json_response_smhi = json.loads(response_smhi.text) 

    # weather data manual https://opendata.smhi.se/apidocs/metanalys/parameters.html#parameter-wsymb
    referenceTime = json_response_smhi["referenceTime"]

    t             = json_response_smhi["timeSeries"][0]["parameters"][0]["values"][0] # Temperature
    ws            = json_response_smhi["timeSeries"][0]["parameters"][4]["values"][0] # Wind Speed
    prec1h        = json_response_smhi["timeSeries"][0]["parameters"][6]["values"][0] # Precipation last hour
    fesn1h        = json_response_smhi["timeSeries"][0]["parameters"][8]["values"][0] # Snow precipation last hour
    vis           = json_response_smhi["timeSeries"][0]["parameters"][9]["values"][0] # Visibility

    
    row           ={"referenceTime": referenceTime, 
                    "t": t, 
                    "ws": ws, 
                    "prec1h": prec1h, 
                    "fesn1h": fesn1h, 
                    "vis": vis, 
                    "confidence": confidence, 
                    "congestionLevel": congestionLevel}


    ds = load_dataset("tilos/IL2223_project", split='train')
    ds = ds.add_item(row)
    ds.push_to_hub("tilos/IL2223_project")


    
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()