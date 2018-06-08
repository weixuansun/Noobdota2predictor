import requests
import numpy as np
import json
import csv
import time
import urllib3
import  tensorflow as tf

# a = np.array([[1,2,3,4],[2,3,4,5]])
# print(a[1,0:3])
# for i in range(10):
#     print(i)
# a = tf.Variable(tf.ones([100,10]))
# b = tf.Variable(tf.ones([10,100]))
# c = tf.matmul(a,b)
# d = tf.nn.sigmoid(tf.matmul(a,b))
# e = tf.round(d)
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     e1 = sess.run(c)
#     e2 = sess.run(d)
#     e3 = sess.run(e)
#     print(e1.shape)
#     print(e3)
#
a = np.zeros([30000,1])
b = np.array_split(a,100)
print(b[1])
