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

# 데이터 학습기
# 24시에 ESS 저장값을 예측하기 위한 추론 장치
# 가변 시간에 대해 진행 가능
file_list1 = ['data_set_ess2_PW2012.csv',
             'data_set_ess2_PW2016.csv',
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
    tmp[['Pump', 'Solar', 'Wind', 'demand', 'fuel_cell', 'off_wind']] = ((dataset[i] - dataset[-1].min()) / (dataset[-1].max() - dataset[-1].min()))[['Pump', 'Solar', 'Wind', 'demand', 'fuel_cell', 'off_wind']]
    tmp[['ESS', 'sess']] = ((dataset[i]) / (data24[-1].max()))[['ESS', 'sess']]

    minimax_dataset.append(tmp)

for time in [1,3,6]:
    transform = DataTransform(0,1)
    train_dataset = []
    train_dataloader = []
    val_dataset = []
    val_dataloader = []
    for i in range(len(file_list1)):
        ds = MakevarDataSet(minimax_dataset[i], time,transform, False, 'train')
        if i==(len(file_list1)-1):
            val_dataset.append(Make24DataSet(ds,None,'val'))
        else:
            train_dataset.append(Make24DataSet(ds,transform,'train'))

    batch_size = 32
    num_epochs = 100
    for i in range(len(file_list1[:-1])):
        train_dataloader.append(data.DataLoader(
            train_dataset[i], batch_size=batch_size, shuffle=True))

    val_dataloader.append(data.DataLoader(val_dataset[0], batch_size=batch_size, shuffle=False))
    dataloaders_dict = {"train1":train_dataloader[0],
                        "train2":train_dataloader[1],
                        "val":val_dataloader[0]}

    model = RegressionNetX((time+24)*6,'train')
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    criterion = nn.MSELoss()

    train_model(model,dataloaders_dict, criterion, optimizer, num_epochs)
    save_path = f'regression{time}th_time.pth'

    torch.save(model.state_dict(), save_path)
