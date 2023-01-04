import requests
import csv


# url = 'https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/0..22/json'
# params = {
#     'key': 'azGiX8jKKGxCxdsF1OzvbbWGPDuInWez',
#     'point': '59.39575,17.98343'
# }
# response = requests.get(url, params=params)

response = requests.get('https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key=azGiX8jKKGxCxdsF1OzvbbWGPDuInWez&point=59.39575,17.98343')
print(response.text)

with open('test.csv', 'w') as f:
    writer = csv.writer(f)

    writer.writerow(response)

print("done")