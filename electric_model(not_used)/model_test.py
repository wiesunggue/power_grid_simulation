import gams
from gams import GamsWorkspace
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

import glob
import os.path as osp
import random
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from itertools import product
from math import sqrt
import time
from dataloader import *
from datanet import *

# 뭐든지 실험용으로 사용하는 파일

file_list1 = ['data_set_ess2_PW2020.csv',
             'data_set_ess2_PW2020.csv',
             'data_set_ess2_PW2020.csv',]

file_list2 = ['data_set_ess2_PW2012.csv',
             'data_set_ess2_PW2016.csv',
             'data_set_ess2_PW2020.csv']
dataset = []
data24 = []
minimax_dataset = []

for path in file_list1:
    dataset.append(pd.read_csv(path,index_col=0))

for i in range(len(file_list1)):
    data24.append(dataset[i].loc[[(i + 1) * 24 for i in range(365)]])

for i in range(len(file_list1)):
    tmp = pd.DataFrame()
    tmp[['Pump', 'Solar', 'Wind', 'demand', 'fuel_cell', 'off_wind']] = ((dataset[i] - dataset[i].min()) / (dataset[i].max() - dataset[i].min()))[['Pump', 'Solar', 'Wind', 'demand', 'fuel_cell', 'off_wind']]
    tmp[['ESS', 'sess']] = ((dataset[i] - data24[i].min()) / (data24[i].max() - data24[i].min()))[['ESS', 'sess']]

    minimax_dataset.append(tmp)



for timer in [1,3,6]:
    transform = DataTransform(0, 1)
    train_dataset = []
    train_dataloader = []
    for i in range(len(file_list1)):
        ds = MakevarDataSet(minimax_dataset[0], timer, transform, False, 'train')
        if i == 2:
            val_dataset = Make24DataSet(ds, None, 'val')
        else:
            train_dataset.append(Make24DataSet(ds, transform, 'train'))

    temp = iter(val_dataset)
    print(next(temp)[0].shape)