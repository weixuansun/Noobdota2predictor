import requests
import numpy as np
import json
import csv
import time
import urllib3
import  tensorflow as tf
from data_process import data_process



data_process_1 = data_process()

# create heros dict
heros_id = np.arange(1, 121)
heros_id_matrix = data_process_1.vec_bin_array(heros_id, 7)
heros_dict = dict(zip(heros_id, heros_id_matrix))

# get training data and test data
heros_data, results_data = data_process_1.process_data('D:/Noobdota2predictor/data.csv')
test_heros_data, test_results_data = data_process_1.process_data('D:/Noobdota2predictor/data_2.csv')

test_results_data = np.reshape(test_results_data, [30000, 2])
results_data = np.reshape(results_data, [30000, 2])
# print(test_results_data)
# print(results_data)

# print(results_data)
# print(heros_data.shape)

#  todo: sort heros of both team by positions: carry, mid, initiate, support.

##map the heros data to a binary matrix
heros_features = data_process_1.map_heros_data_matrix(heros_data, heros_dict)
test_heros_features = data_process_1.map_heros_data_matrix(test_heros_data, heros_dict)
# print(heros_features)
# print(test_heros_features)

train_heros_features = heros_features[0:25000,:]
print(heros_features.shape)
test_heros_features = heros_features[25000:30000,:]
print(test_heros_features.shape)

train_results_data = results_data[0:25000,:]

test_results_data = results_data[25000:30000,:]