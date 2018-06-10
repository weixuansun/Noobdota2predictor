import requests
import numpy as np
import json
import csv
import time
import urllib3
import  tensorflow as tf
from data_process import data_process


data_process_1 = data_process()
# data_process_1.get_hero_data(1)
heros_id = np.arange(1, 121)
heros_id_matrix = data_process_1.vec_bin_array(heros_id, 7)
heros_dict = dict(zip(heros_id, heros_id_matrix))
print(heros_dict)
heros_data, results_data = data_process_1.process_data('D:/Noobdota2predictor/data.csv')
test_heros_data, test_results_data = data_process_1.process_data('D:/Noobdota2predictor/data_2.csv')
# results_data = np.transpose(results_data)
# print(results_data.shape)
test_results_data = np.reshape(test_results_data, [30000, 1])
results_data = np.reshape(results_data, [30000, 1])
# print(results_data)
print(heros_data.shape)
#  todo: sort heros of both team by positions: carry, mid, initiate, support.
heros_features = data_process_1.map_heros_data_matrix(heros_data, heros_dict)
test_heros_features = data_process_1.map_heros_data_matrix(test_heros_data, heros_dict)
print(heros_features.shape)


learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 1
# match_data = np.asanyarray(match_data,np.float32)

x = tf.placeholder(tf.float32, [None, 70])
y = tf.placeholder(tf.float32, [None, 1])

# dataset = tf.data.Dataset.from_tensor_slices((x,y))
# iter = dataset.make_initializable_iterator()
# print(dataset)

def net(x,weights,biases):
    radiant_heros = x[:,0:35]
    dire_heros = x[:,35:70]
    team_radiant = tf.add(tf.matmul(radiant_heros,weights['h1']), biases['b1'])
    team_dire = tf.add(tf.matmul(radiant_heros, weights['h2']), biases['b2'])
    team_radiant_2 = tf.add(tf.matmul(team_radiant,weights['h3']), biases['b3'])
    team_dire_2 = tf.add(tf.matmul(team_dire, weights['h4']), biases['b4'])
    # Hidden layer with RELU activation
    # layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with RELU activation
    # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer with linear activation
    out_layer = tf.nn.sigmoid(tf.subtract(team_radiant_2,team_dire_2))
    return out_layer

weights = {
    'h1':tf.Variable(tf.random_normal([35,35])),
    'h2':tf.Variable(tf.random_normal([35,35])),
    'h3': tf.Variable(tf.random_normal([35, 1])),
    'h4': tf.Variable(tf.random_normal([35, 1])),
    'out': tf.Variable(tf.random_normal([70,1]))
}

biases = {
    'b1':tf.Variable(tf.random_normal([35])),
    'b2':tf.Variable(tf.random_normal([35])),
    'b3':tf.Variable(tf.random_normal([1])),
    'b4':tf.Variable(tf.random_normal([1])),
    'out': tf.Variable(tf.random_normal([1]))
}

prediction = net(x,weights,biases)

correct_prediction = tf.equal(tf.round(prediction), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))
loss = tf.reduce_min(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
#
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
#
# save result to a boolean list
# correct_prediction = tf.equal(y, tf.round(prediction))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(iter.initializer, feed_dict={x: heros_features, y: results_data})
    for epoch in range(training_epochs):
        total_batch = int(len(results_data) / batch_size)
        x_batches = np.array_split(heros_features, total_batch)
        y_batches = np.array_split(results_data, total_batch)
        # print(total_batch)
        # print(len(x_batches), len(y_batches))
        for i in range(total_batch):
            avg_cost = 0.
            batch_x, batch_y = x_batches[i], y_batches[i]
            # print(batch_x)
            # print(batch_y)
            # print(pre)
            # print(batch_x.shape,batch_y.shape)
            _, c = sess.run([train, loss], feed_dict={x: batch_x, y: batch_y})
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            pre = sess.run(prediction,feed_dict={x: batch_x, y: batch_y})
            # print(pre)


            # print(c)
            # avg_cost += c / total_batch

        if epoch % display_step == 0:
            # print(pre)
            print(acc)
        #     print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        correct_prediction = tf.equal(tf.round(prediction), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print('accuracy:', accuracy.eval({x: test_heros_features, y: test_results_data}))
    print('training finished')

    # test model








