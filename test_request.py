import requests
import csv
import json
import requests
import time
from datetime import datetime

# url = 'https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/0..22/json'
# params = {
#     'key': 'azGiX8jKKGxCxdsF1OzvbbWGPDuInWez',
#     'point': '59.39575,17.98343'
# }
# response = requests.get(url, params=params)



#response = requests.get('https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key=azGiX8jKKGxCxdsF1OzvbbWGPDuInWez&point=59.39575,17.98343')
my_file = open("test.csv", "w")

with my_file as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ["FunctionalRoadClass","CurrentSpeed"])

    response = requests.get(
                'https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key=azGiX8jKKGxCxdsF1OzvbbWGPDuInWez&point=59.39575,17.98343')
    json_response = json.loads(response.text)  # get json response

    #data structure see "RESPONSE BODY - JSON" part : https://developer.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data
    frc = json_response["flowSegmentData"]["frc"]  #Functional Road Class. This indicates the road type: FRC0 : Motorway, freeway or other major road
                                                                                                        #FRC1 : Major road, less important than a motorway
                                                                                                        #FRC2 : Other major road
                                                                                                        #FRC3 : Secondary road
                                                                                                        #FRC4 : Local connecting road
                                                                                                        #FRC5 : Local road of high importance
                                                                                                        #FRC6 : Local road
    currentSpeed = json_response["flowSegmentData"]["currentSpeed"]


    csv_writer.writerow([frc,currentSpeed])  # write to csv
    my_file.flush()

print("done")