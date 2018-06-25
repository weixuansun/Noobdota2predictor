import requests
import numpy as np
import json
import csv
import time
import urllib3
import  tensorflow as tf
from data_process import data_process



data_process_1 = data_process()

list = np.array([[1, 2, 3],
                 [-1, 2, -4]])
# a = tf.argmax(list,1)
#
# with tf.Session() as sess:
#     b = sess.run(a)
#     print(b)
b = [1,2,3,4]

a = np.delete(list,0,1)
print(b[-1])

