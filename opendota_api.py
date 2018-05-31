## get raw dota match data from opendota api
import requests
import numpy as np
import json
import csv
import time
import urllib3


#url = ('https://api.opendota.com/api/matches/271145478?api_key=3ea68b18-147a-4d37-9d13-68e24a4fd033')
#r = requests.get(url)

class data_process(object):
    def get_data(self):
        #match_data = requests.get('https://api.opendota.com/api/matches/3919387348')
        parameters = {'mmr_descending':'mmr_descending'}
        response = requests.get('https://api.opendota.com/api/publicMatches',params=parameters)
        #print(response.content.decode('utf-8'))
        if response.status_code ==200:
            return response
        else:
            return None
        #print(match_data)

    #json.dumps() transfer python data structure to json
    #json.loads() transfer json data to python data structure
    def save_data(self, data):
        #data = json.loads(data.content.decode('utf-8'))
        f = csv.writer(open('test.csv', 'w'))  #
        f.writerow(data[0].keys()) #header
        for row in data:
            f.writerow(row.values()) #write value

    def get_hero_data(self, hero_id):
        # http = urllib3.PoolManager()
        heroes = requests.get('https://api.opendota.com/api/heroes')
        # print(heroes)
        # print(heroes.content)
        # print(heroes.content.decode('UTF-8'))
        heroes_dict = json.loads(heroes.content.decode('utf-8'))
        print(heroes_dict)
        print(heroes_dict[hero_id-1]['localized_name'])

    ##process csv match data for training
    def process_data(self,filename):
        with open(filename) as csv_file:
            f = csv.reader(csv_file)
            f = list(f)
            print(len(f))
            data_list = []
            for i in range(2,60001,2):
                data_list.append(f[i][2])
            # print(data_list)
            print(len(data_list))
        return (data_list)












if __name__ == '__main__':
    data_process_1 = data_process()
    # #  collect data
    # match_data = []
    # for step in range(3):
    #     print(step)
    #     raw_data = data_process_1.get_data()
    #     raw_data = json.loads(raw_data.content.decode('utf-8'))
    #     for item in raw_data:
    #         match_data.append(item)
    #     time.sleep(1)
    # data_process_1.save_data(match_data)


    # data_process_1.get_hero_data(1)

    data_list = data_process_1.process_data('D:/Noobdota2predictor/data.csv')
    print(data_list)







