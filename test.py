import requests
import numpy as np
import json
import csv
import time
import urllib3
import  tensorflow as tf
from data_process import data_process
from random import seed
from random import randint

seed(1)

for _ in range(100):
    value = randint(0,9)
    print(value)
