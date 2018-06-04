## get raw dota match data from opendota api
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
        # http = urllib3.PoolManager()
        heroes = requests.get('https://api.opendota.com/api/heroes')
        # print(heroes)
        # print(heroes.content)
        # print(heroes.content.decode('UTF-8'))
        heroes_dict = json.loads(heroes.content.decode('utf-8'))
        print(heroes_dict)
        # print(heroes_dict[hero_id-1]['localized_name'])
        # for i in range(115):
        #     print(heroes_dict[i])

    ##process csv match data for training
    def process_data(self,filename):
        with open(filename) as csv_file:
            f = csv.reader(csv_file)
            f = list(f)
            print(len(f))
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
                # print(radiant_team)
            # print(data_matrix)
            # print(np.shape(data_matrix))
        # print(data_matrix.shape)
        heros_data = data_matrix[0:data_matrix.shape[0],1:11]
        results_data = data_matrix[0:data_matrix.shape[0],0]
        print(heros_data.shape)
        return heros_data, results_data

    # transfer match data into one shot data
    def match_data_to_features(self,heros_data,results_data):
        heros_data_matrix = np.zeros([len(heros_data),70])

        return heros_data_matrix








# class train(object):

def train(heros_data_matrix, results_data):
    # match_data = np.asanyarray(match_data,np.float32)

    x = tf.placeholder(tf.int8,[None,1150])
    y = tf.placeholder(tf.int8,[None,1])

    features_placeholder = tf.placeholder(tf.float32,[1,1150])
    labels_placeholder = tf.placeholder(tf.float32,[1])

    dataset = tf.data.Dataset.from_tensor_slices((heros_data,results_data))
    print(dataset)





    # W = tf.Variable(tf.zeros[1150,1])
    # b = tf.Variable(tf.zeros[1])
    # prediction = tf.nn.softmax(tf.matmul(x,W)+b)
    # # build net work
    # loss = tf.reduce_min(tf.square(y-prediction))
    #
    # train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    #
    # # save result to a boolean list
    # correct_prediction = tf.equal(y, prediction)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     for epoch in range(20):
    #         sess.run(train,feed_dict={x:match_data[]}


if __name__ == '__main__':
    data_process_1 = data_process()

    #  collect data
    # match_data = []
    # for step in range(300):
    #     print(step)
    #     raw_data = data_process_1.get_data()
    #     raw_data = json.loads(raw_data.content.decode('utf-8'))
    #     for item in raw_data:
    #         match_data.append(item)
    #     time.sleep(60)
    # data_process_1.save_data(match_data)


    # data_process_1.get_hero_data(1)
    heros_data, results_data = data_process_1.process_data('D:/Noobdota2predictor/data.csv')
    # train(heros_data,results_data)

    #

    ###caogaozhi
    # a = np.array([[1,2,3],[2,3,4]])
    # print(a[0:100,1])
    # print(a)
    # print(a[1,2])
    # print(np.zeros([5,3]))
    b = np.zeros([5,3])
    print(b.shape[1])
    # print(b)
    # a = [11,12,12,1,12,33,4,5]
    # print(a[1:5])
    # b[0:5] = a[1:6]
    # print(b)
    #
    # a = np.array([1,2,3])
    # b = np.array([1,2,3])
    # c = tf.equal(a,b)
    # print(c)
    # a = tf.ones([100,1150])
    # # print(a)
    # w = tf.ones([1150,1])
    # b = tf.matmul(a






