import requests
import numpy as np
import json
import csv
import time
import urllib3
import  tensorflow as tf
import boxx
import pickle

#url = ('https://api.opendota.com/api/matches/271145478?api_key=3ea68b18-147a-4d37-9d13-68e24a4fd033')
#r = requests.get(url)

class data_process(object):
    def fetch_data(self,amount):
        '''
        get data from openDota api
        :return: reponse from api
        '''
        #match_data = requests.get('https://api.opendota.com/api/matches/3919387348')

        match_data = []
        for step in range(amount):
            print(step)
            try:
                parameters = {'mmr_descending': 'mmr_descending'}
                response = requests.get('https://api.opendota.com/api/publicMatches', params=parameters)
                # print(response.content.decode('utf-8'))
            except:
                raise Exception('Error! cannot get data！')
            raw_data = json.loads(response.content.decode('utf-8'))
            for item in raw_data:
                match_data.append(item)
            time.sleep(1)
        data_process_1.save_data(match_data)

        #print(match_data)m

    #json.dumps() transfer python data structure to json
    #json.loads() transfer json data to python data structure
    def save_data(self, data):
        #data = json.loads(data.content.decode('utf-8'))
        f = csv.writer(open('data_5.csv', 'w'))  #
        f.writerow(data[0].keys()) #header
        for row in data:
            f.writerow(row.values()) #write value

    def get_hero_data(self):
        heroes = requests.get('https://api.opendota.com/api/heroes')
        heroes_dict = json.loads(heroes.content.decode('utf-8'))
        # print(heroes_dict)
        heroes_id = np.arange(115)
        heroes_info_dict = dict(zip(heroes_id,heroes_dict))
        # dict.keys()[dict.values().index()]
        # print(len(heroes_dict))
        id_list = []
        output = open('heroes_info_dict.pkl','wb')
        pickle.dump(heroes_info_dict,output)
        output.close()
        # return heroes_info_dict
        # for i in range(115):
        #     print(heroes_dict[i])
    def save_id_dict(self):
        pkl_file = open('heroes_info_dict.pkl', 'rb')
        heroes_info_dict = pickle.load(pkl_file)
        pkl_file.close()

        heros_id = np.arange(115)
        id_list = []
        for i in range(115):
            id_list.append(int(heroes_info_dict[i]['id']))
        id_dict_1 = dict(zip(id_list, heros_id))
        id_dict_2 = dict(zip(heros_id, id_list))
        output_1 = open('id_dict_1.pkl','wb')
        output_2 = open('id_dict_2.pkl', 'wb')
        pickle.dump(id_dict_1,output_1)
        pickle.dump(id_dict_2,output_2)
        output_1.close()
        output_2.close()


    ##process csv match data for training
    def process_data(self,filename):
        '''
        read match csv data into array
        :param filename:
        :return: heros_data, results_data
        '''
        with open(filename) as csv_file:
            f = csv.reader(csv_file)
            f = list(f)
            data_matrix = np.zeros([int((len(f)-1)/2),12])
            for i in range(2,len(f),2):
                # print(f[i][2])
                data_matrix[int((i / 2) - 1), 0:2] = [1,0] if f[i][2] == 'True' or f[i][2] =='TRUE' else [0,1]
                # print(data_matrix[int((i / 2) - 1), 0:2])
                radiant_team = f[i][12]
                dire_team = f[i][13]
                radiant_team = radiant_team.split(',',4)
                dire_team = dire_team.split(',',4)
                # print(radiant_team)
                # print(dire_team)
                data_matrix[(int(i / 2) - 1), 2:7] = radiant_team[0:5]
                data_matrix[(int(i / 2) - 1), 7:12] = dire_team[0:5]
        # print(data_matrix.shape)
        heros_data = data_matrix[0:data_matrix.shape[0],2:12]
        results_data = data_matrix[0:data_matrix.shape[0],0:2]
        # results_data = np.transpose(results_data)
        # print(heros_data)
        return heros_data, results_data
    # def sort_heroes_positions(self, heros_data):
    #     '''
    #     sort heroes by positions: carry mid, initiate, support..
    #     '''
        pkl_file = open('heroes_info_dict.pkl', 'rb')
        heroes_info_dict = pickle.load(pkl_file)
        pkl_file.close()
        pkl_file = open('id_dict_1.pkl', 'rb')
        id_dict_1 = pickle.load(pkl_file)
        pkl_file.close()
        pkl_file = open('id_dict_2.pkl', 'rb')
        id_dict_2 = pickle.load(pkl_file)
        pkl_file.close()
        sorted_heroes_data = np.zeros(10)
        for i in range(5):
            hero_id = id_dict_1[heros_data[i]]
            hero_position = heroes_info_dict[hero_id]['roles']





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

    def map_heros_data_matrix(self, heros_data, id_dict):
        '''
        map heros id to one hot matrix
        :param heros_data:
        :param id_dict:
        :return:
        '''
        heros_features = np.zeros([heros_data.shape[0],230])
        for i in range(heros_data.shape[0]):
            for j in range(5):
                hero_id = id_dict[heros_data[i,j]]
                heros_features[i][hero_id] = 1
            for j in range(5):
                hero_id = id_dict[heros_data[i,j+5]]
                heros_features[i][hero_id+115] = 1
                # heros_features[i,(7*j):(7*j+7)] = heros_dict[hero_id]
        return heros_features

if __name__ == '__main__':
    data_process_1 = data_process()

    #  collect data from opendota
    # data_process_1.fetch_data(10)


    # heros_id = np.arange(1, 116)
    # heroes_info_dict = data_process_1.get_hero_data()
    # id_list = []
    # for i in range(115):
    #     id_list.append(heroes_info_dict[i]['id'])
    # print(id_list)
    # print(len(id_list))
    # id_dict = dict(zip(id_list, heros_id))
    # print(id_dict)

    # heros_id = np.arange(115)


    # heros_data, results_data = data_process_1.process_data('data_3.csv')
    # print(heros_data[0,:])
    # id_list = []
    # for i in range(115):
    #     id_list.append(int(heroes_info_dict[i]['id']))
    #
    # id_dict = dict(zip(id_list, heros_id))
    # print(id_dict)
    # # boxx.loga(heros_data)
    # heros_features = data_process_1.map_heros_data_matrix(heros_data,id_dict)
    # # for i in range(10000):
    # print(heros_features[0,: ])
    # print(np.argmax(heros_features[0, :]))

    data_process_1.get_hero_data()
    data_process_1.save_id_dict()
    pkl_file = open('heroes_info_dict.pkl', 'rb')
    heroes_info_dict = pickle.load(pkl_file)
    print(heroes_info_dict)

    # pkl_file.close()
    # pkl_file = open('id_dict_1.pkl', 'rb')
    # id_dict_1 = pickle.load(pkl_file)
    # pkl_file.close()
    # pkl_file = open('id_dict_2.pkl', 'rb')
    # id_dict_2 = pickle.load(pkl_file)
    # pkl_file.close()
    #
























