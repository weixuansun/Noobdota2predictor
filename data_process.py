import requests
import numpy as np
import json
import csv
import time
import urllib3
import  tensorflow as tf

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
        f = csv.writer(open('data_2.csv', 'w'))  #
        f.writerow(data[0].keys()) #header
        for row in data:
            f.writerow(row.values()) #write value

    def get_hero_data(self, hero_id):
        heroes = requests.get('https://api.opendota.com/api/heroes')
        # print(heroes)
        # print(heroes.content)
        # print(heroes.content.decode('UTF-8'))
        heroes_dict = json.loads(heroes.content.decode('utf-8'))
        # print(heroes_dict)
        print(heroes_dict[hero_id-1]['localized_name'])
        # for i in range(115):
        #     print(heroes_dict[i])

    ##process csv match data for training
    def process_data(self,filename):
        with open(filename) as csv_file:
            f = csv.reader(csv_file)
            f = list(f)
            # print(len(f))
            data_matrix = np.zeros([int((len(f)-1)/2),11])
            for i in range(2,len(f),2):
                data_matrix[int((i / 2) - 1), 0] = 1 if f[i][2] == 'TRUE' else 0
                # if f[i][2] == 'TRUE':
                #     data_matrix[int((i/2)-1),0] = 1
                # else:
                #     data_matrix[(int(i/2)-1),0] = 0
                radiant_team = f[i][12]
                dire_team = f[i][13]
                radiant_team = radiant_team.split(',',4)
                dire_team = dire_team.split(',',4)
                # print(radiant_team)
                # print(dire_team)
                data_matrix[(int(i / 2) - 1), 1:6] = radiant_team[0:5]
                data_matrix[(int(i / 2) - 1), 6:11] = dire_team[0:5]
        heros_data = data_matrix[0:data_matrix.shape[0],1:11]
        results_data = data_matrix[0:data_matrix.shape[0],0]
        # results_data = np.transpose(results_data)
        # print(heros_data.shape)
        return heros_data, results_data
    # transfer match data into one shot data
    def vec_bin_array(self, arr, m):
        """
        Arguments:
        arr: Numpy array of positive integers
        m: Number of bits of each integer to retain

        Returns a copy of arr with every element replaced with a bit vector.
        Bits encoded as int8's.
        """
        to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
        strs = to_str_func(arr)
        ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
        for bit_ix in range(0, m):
            fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
            ret[..., bit_ix] = fetch_bit_func(strs).astype("int8")
        return ret

    def map_heros_data_matrix(self, heros_data, heros_dict):
        heros_features = np.zeros([heros_data.shape[0],70])
        for i in range(heros_data.shape[0]):
            for j in range(10):
                heros_features[i,(7*j):(7*j+7)] = heros_dict[heros_data[i,j]]
        return heros_features

if __name__ == '__main__':
    data_process_1 = data_process()

    #  collect data from opendota
    ###################
    # match_data = []
    # for step in range(300):
    #     print(step)
    #     raw_data = data_process_1.get_data()
    #     raw_data = json.loads(raw_data.content.decode('utf-8'))
    #     for item in raw_data:
    #         match_data.append(item)
    #     time.sleep(60)
    # data_process_1.save_data(match_data)
    ##################

    # data_process_1.get_hero_data(1)
    heros_id = np.arange(1, 121)
    heros_id_matrix = data_process_1.vec_bin_array(heros_id,7)
    heros_dict = dict(zip(heros_id,heros_id_matrix))
    # print(heros_dict)
    heros_data, results_data = data_process_1.process_data('D:/Noobdota2predictor/data.csv')
    test_heros_data, test_results_data = data_process_1.process_data('D:/Noobdota2predictor/data_2.csv')
    # results_data = np.transpose(results_data)
    # print(results_data.shape)
    test_results_data = np.reshape(test_results_data,[30000,1])
    results_data = np.reshape(results_data,[30000,1])
    # print(results_data)
    print(heros_data.shape)
    #  todo: sort heros of both team by positions: carry, mid, initiate, support.
    heros_features = data_process_1.map_heros_data_matrix(heros_data,heros_dict)
    test_heros_features = data_process_1.map_heros_data_matrix(test_heros_data,heros_dict)
    print(heros_features.shape)
    train(heros_features,results_data,test_heros_features,test_results_data)














