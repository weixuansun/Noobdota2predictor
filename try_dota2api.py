import dota2api
import numpy as np
import tensorflow as tf
import pandas as pd

dota_data = pd.read_csv("D:\dota2_predictor\dota_data.csv")
print(dota_data)

x = tf.placeholder(tf.float32,[None,1150])
y = tf.placeholder(tf.float32,[None,1])

def dataset_to_feature(dataset):
    x_matrix = np.zeros(110,10)
    


#from 万隼舞 of Zhihu
def _dataset_to_features(dataset_df):
    # 构造一个空的x目标矩阵，列数为英雄数量*2，行数为样本数量
    x_matrix = np.zeros((dataset_df.shape[0], 2 * 120))

    # 构造一个空的y目标矩阵，行数为样本数量
    y_matrix = np.zeros(dataset_df.shape[0])

    # 将原样本中的数据，用pandas的values函数导出为一个numpy的矩阵类型
    dataset_np = dataset_df.values

    # 对矩阵的每行每个英雄，分别映射到x的目标矩阵中
    for i, row in enumerate(dataset_np):
        radiant_win = row[10]
        for j in range(5):
            x_matrix[i, row[j] - 1] = 1
            x_matrix[i, row[j + 5] - 1] = 1
        # 将游戏胜负映射到y的目标矩阵中
        y_matrix[i] = 1 if radiant_win else 0

    return [x_matrix, y_matrix]