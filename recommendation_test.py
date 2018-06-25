import requests
import numpy as np
import json
import csv
import time
import urllib3
import  tensorflow as tf
from data_process import data_process
import boxx
import pickle

data_process_1 = data_process()

# create dicts
heros_id = np.arange(115)
heroes_info_dict = data_process_1.get_hero_data()
id_list = []
for i in range(115):
    id_list.append(int(heroes_info_dict[i]['id']))
id_dict_1 = dict(zip(id_list, heros_id))
id_dict_2 = dict(zip(heros_id,id_list))
# print(id_dict_1)
# print(id_dict_2)
# print(heroes_info_dict)

# read match data
heros_data, results_data = data_process_1.process_data('data_3.csv')

size = len(heros_data)


results_data = np.reshape(results_data, [size, 2])
# map the heros data to a binary matrix
heros_features = data_process_1.map_heros_data_matrix(heros_data,id_dict_1)
# print(heros_features)
# print(results_data)
results = np.zeros(size)

for i in range(size):

    if results_data[i,0] == 1 and results_data[i,1] == 0:
        results[i] = 1
    else:
        results[i] = 0
    # print(results_data[i,:])
    # print(results[i])
print()
results_data = np.reshape(results,[size,1])

# test:
# pick one match for test
match_data = heros_data[6000,:]
# print(results_data[71,:])
print(match_data)
match_heros = []
for i in range(10):
    hero_id = id_dict_1[match_data[i]]
    hero_name = heroes_info_dict[hero_id]['localized_name']
    match_heros.append(hero_name)
print(match_heros)

# change the first position hero, others are all the same


X = tf.placeholder(tf.float64,[None,230])
Y = tf.placeholder(tf.float64,[None,1])

weights = {
    'layer_1' : tf.Variable(np.random.rand(115,115), name='w_layer_1'),
    'layer_2' : tf.Variable(np.random.rand(115,1), name='w_layer_2'),
    # 'layer_3' : tf.Variable(np.ones([70,70]), name='w_layer_3'),

    'out': tf.Variable(np.random.rand(115,1),name='w_out')
}

biases = {
    'layer_1' : tf.Variable(np.random.rand(115), name='b_layer_1'),
    'layer_2' : tf.Variable(np.random.rand(1), name='b_layer_2'),
    # 'layer_3' : tf.Variable(np.ones(70), name='b_layer_3'),
    'out': tf.Variable(np.random.rand(1), name='b_out')
}

def net(x,weights,biases):
    x_radiant = x[:,0:115]
    x_dire = x[:,115:230]
    # layer_1
    layer_1_radiant = tf.matmul(x_radiant, weights['layer_1']) + biases['layer_1']
    layer_1_radiant = tf.nn.relu(layer_1_radiant)
    layer_1_dire = tf.matmul(x_dire, weights['layer_1']) + biases['layer_1']
    layer_1_dire = tf.nn.relu(layer_1_dire)
    # layer_2
    layer_2_radiant = tf.matmul(layer_1_radiant, weights['layer_2']) + biases['layer_2']
    layer_2_radiant = tf.nn.relu(layer_2_radiant)
    layer_2_dire = tf.matmul(layer_1_dire, weights['layer_2']) + biases['layer_2']
    layer_2_dire = tf.nn.relu(layer_2_dire)


    # layer_2 = tf.matmul(layer_1,weights['layer_2']) + biases['layer_2']
    # layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.subtract(layer_2_radiant,layer_2_dire)
    # out_layer = tf.nn.sig
    return out_layer

pred = net(X,weights,biases)

# correct_prediction = tf.equal(tf.argmax(Y,axis=1), tf.argmax(pred,axis=1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=Y))
# train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'recc/model.ckpt')
    print('weights', sess.run(weights))
    print('biases', sess.run(biases))
    trained_weights = sess.run(weights)
    print(trained_weights)
    # result = sess.run(pred, feed_dict={X: heros_features})
    # test_result = []
    # for i in range(115):
    #     if id_dict_2[i] not in match_data[1:-1]:
    #         match_data[0] = id_dict_2[i]
    #         test_match_matrix = np.reshape(match_data,[1,10])
    #         print(test_match_matrix)
    #         heros_matrix = data_process_1.map_heros_data_matrix(test_match_matrix, id_dict_1)
    #         print(heros_matrix)
    #         result = sess.run(pred, feed_dict={X: heros_matrix})
    #         test_result.append(result[0][0])
    # print(np.max(test_result))
    # print(np.argmax(test_result))
    # print(test_result)

    output = sess.run(pred, feed_dict={X:heros_features})
    for i in range(10000):
        print(output[i], results_data[i])

    # for i in range(1000):
    #     print(result[i,:], results_data[i,:])
    #
    # acc = sess.run(accuracy, feed_dict={X:heros_features, Y: results_data})
    # print('acc: %s' % acc)
