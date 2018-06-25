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

id_dict = dict(zip(id_list, heros_id))
# read match data
heros_data, results_data = data_process_1.process_data('data_3.csv')

size = len(heros_data)


results_data = np.reshape(results_data, [size, 2])
# map the heros data to a binary matrix
heros_features = data_process_1.map_heros_data_matrix(heros_data,id_dict)
print(heros_features)
print(results_data)
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

#  todo: sort heros of both team by positions: carry, mid, initiate, support.

# generate training and test data
train_size = size-5000
train_heros_features = heros_features[0:train_size,:]
test_heros_features = heros_features[train_size:size,:]
train_results_data = results[0:train_size]
test_results_data = results[train_size:size]


# training parameters
learning_rate = 0.01
training_epochs = 200
batch_size = 100
display_step = 50

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

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=Y))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

saver = tf.train.Saver()


# launch the train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int((size-5000)/batch_size)
        x_batches = np.array_split(train_heros_features, total_batch)
        y_batches = np.array_split(train_results_data, total_batch)
        for i in range(total_batch):
            x = x_batches[i]
            y = y_batches[i]
            _, c = sess.run([train, cost], feed_dict={X:x, Y:y })

            avg_cost += c / total_batch
        if epoch % 2 == 0:
            print(avg_cost)
        # print(acc1)

    save_path = saver.save(sess, 'recc/model.ckpt')
    print("Model saved in path: %s" % save_path)
    print('training finished!')
    # acc = sess.run(accuracy, feed_dict={X:test_heros_features, Y:test_results_data})
    # print(str(acc))