import requests
import numpy as np
import json
import csv
import time
import urllib3
import  tensorflow as tf
from random import seed
from random import randint

class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # size = len(nums)
        # m = 1
        # loop = 0
        # for i in range(size-1):
        #     loop = loop + m
        #     m = m + 1
        # print(loop)

        # for i in range(loop):
        for i in range(len(nums)-1):
            rest_list = nums[i+1:len(nums)]
            print(rest_list)
            for j in range(len(rest_list)):
                if nums[i] + rest_list[j] == target:
                    pos_1 = i
                    pos_2 = j + i + 1
                    break
                else:
                    continue


        return [pos_1,pos_2]






s = Solution()
result = s.twoSum([2,7,11,15,16,4,8], 24)
print(result)

