import requests
import numpy as np
import json
import csv
import time

#url = ('https://api.opendota.com/api/matches/271145478?api_key=3ea68b18-147a-4d37-9d13-68e24a4fd033')
#r = requests.get(url)

def get_data():
    #match_data = requests.get('https://api.opendota.com/api/matches/3919387348')
    parameters = {'mmr_descending':'mmr'}
    response = requests.get('https://api.opendota.com/api/publicMatches',params=parameters)
    if response.status_code ==200:
        return response
    else:
        return None
    #print(match_data)

#get_data()

########
#json.dumps() transfer python data structure to json
#json.loads() transfer json data to python data structure
def save_data(data):
    data = json.loads(data.content.decode('utf-8'))
    f = csv.writer(open('data.csv', 'w'))  #
    f.writerow(data[0].keys()) #header
    for row in data:
        f.writerow(row.values()) #write value

if __name__ == '__main__':
    data = []
    for step in range(1):
        raw_data = get_data()
        save_data(raw_data)















