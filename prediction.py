import requests
import numpy as np
import json
import csv
import time
import urllib3
import  tensorflow as tf
from data_process import data_process

data_process_1 = data_process()

# def get_picked_heros():

heros_data, results_data = data_process_1.process_data('D:/Noobdota2predictor/data_3.csv')
heros_id = np.arange(1, 121)
heros_id_matrix = data_process_1.vec_bin_array(heros_id, 7)
heros_dict = dict(zip(heros_id, heros_id_matrix))
heros_features = data_process_1.map_heros_data_matrix(heros_data, heros_dict)

# test:
match_data = heros_data[0,:]
result = results_data[0,:]
match_matrix = data_process_1.map_heros_data_matrix(heros_data, heros_dict)
# print(match_matrix)

for i in range(120):
    match_matrix[i, 0:7] = heros_dict[i+1]
    # print(match_matrix[i,:])




# print(match_data)
# print(result)

X = tf.placeholder(tf.float64,[None,70])
Y = tf.placeholder(tf.float64,[None,2])

weights = {
    'layer_1' : tf.Variable(np.ones([70,70]), name='w_layer_1'),
    # 'layer_2' : tf.Variable(np.ones([70,70]), name='w_layer_2'),
    # 'layer_3' : tf.Variable(np.ones([70,70]), name='w_layer_3'),

    'out': tf.Variable(np.ones([70,2]),name='w_out')
}

biases = {
    'layer_1' : tf.Variable(np.ones(70), name='b_layer_1'),
    # 'layer_2' : tf.Variable(np.ones(70), name='b_layer_2'),
    # 'layer_3' : tf.Variable(np.ones(70), name='b_layer_3'),
    'out': tf.Variable(np.ones(2), name='b_out')
}

def net(x,weights,biases):
    layer_1 = tf.matmul(x, weights['layer_1']) + biases['layer_1']
    layer_1 = tf.nn.relu(layer_1)
    # layer_2 = tf.matmul(layer_1,weights['layer_2']) + biases['layer_2']
    # layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

pred = net(X,weights,biases)
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'D:/Noobdota2predictor/net_2/model_2.ckpt')
    # print('weights', sess.run(weights))
    # print('biases', sess.run(biases))
    result = sess.run(pred, feed_dict={X: match_matrix})

    # acc = sess.run(accuracy, feed_dict={X:heros_features, Y: results_data})
    # print(acc)
diff = np.zeros([121])
for i in range(121):
    # print(result[i,:])
    diff[i] = np.subtract(result[i,0], result[i,1])

print(diff)
print(np.argmax(diff, axis=0))

# diff = np.sort(diff)
# print(diff)

# for i in range(121):
#     print(diff[i])







