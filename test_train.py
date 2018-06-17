import requests
import numpy as np
import json
import csv
import time
import urllib3
import  tensorflow as tf
from data_process import data_process
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

data_process_1 = data_process()

# create heros dict
heros_id = np.arange(1, 121)
heros_id_matrix = data_process_1.vec_bin_array(heros_id, 7)
heros_dict = dict(zip(heros_id, heros_id_matrix))

# get training data and test data
heros_data, results_data = data_process_1.process_data('D:/Noobdota2predictor/data_3.csv')
# test_heros_data, test_results_data = data_process_1.process_data('D:/Noobdota2predictor/data.csv')

# test_results_data = np.reshape(test_results_data, [30000, 2])

results_data = np.reshape(results_data, [40000, 2])

#  todo: sort heros of both team by positions: carry, mid, initiate, support.


##map the heros data to a binary matrix
heros_features = data_process_1.map_heros_data_matrix(heros_data, heros_dict)

train_heros_features = heros_features[0:35000,:]
test_heros_features = heros_features[35000:40000,:]
train_results_data = results_data[0:35000,:]
test_results_data = results_data[35000:40000,:]


# learning_rate = 0.0008
learning_rate = 0.3
training_epochs = 200
batch_size = 100
display_step = 50

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

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

saver = tf.train.Saver()



# launch the train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        total_batch = int(35000/batch_size)
        x_batches = np.array_split(train_heros_features, total_batch)
        y_batches = np.array_split(train_results_data, total_batch)
        for i in range(total_batch):
            x = x_batches[i]
            y = y_batches[i]
            sess.run(train, feed_dict={X:x, Y:y })
        # if epoch % 2 == 0:
        # acc1 = sess.run(accuracy, feed_dict={X: heros_features, Y: results_data})
        # print(acc1)
        acc = sess.run(accuracy, feed_dict={X:test_heros_features, Y:test_results_data})
        print(str(acc))
    save_path = saver.save(sess, 'D:/Noobdota2predictor/net_2/model_2.ckpt')
    print("Model saved in path: %s" % save_path)
    print('training finished!')

    # print_tensors_in_checkpoint_file('D:/Noobdota2predictor/net/save_net.ckpt', None, True)




