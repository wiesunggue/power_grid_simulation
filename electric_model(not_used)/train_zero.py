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
# 안씀 - zero 가 붙은 상위 폴더 제작으로 인해
# 데이터 학습 파일
# 이진 분류 학습을 가변 시간에 대해서 실행
# ON/OFF 는 6개 동시 학습해도 된다.

file_list1 = ['data_set_ess1_PW2012.csv',
             'data_set_ess1_PW2016.csv',
             'data_set_ess1_PW2020.csv',
             'data_set_ess2_PW2012.csv',
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
    tmp[['ESS', 'sess']] = ((dataset[i] - data24[-1].min()) / (data24[-1].max() - data24[-1].min()))[['ESS', 'sess']]

    minimax_dataset.append(tmp)

transform = DataTransform(0,1)
train_dataset = []
train_dataloader = []
for i in range(len(file_list1)):
    ds = MakeDataSet(minimax_dataset[i], transform, True, 'train')
    if i==(len(file_list1)-1):
        val_dataset = Make24DataSet(ds,None,'val')
    train_dataset.append(Make24DataSet(ds,transform,'train'))

batch_size = 32
num_epochs = 1000
for i in range(len(file_list1[:-1])):
    train_dataloader.append(data.DataLoader(
        train_dataset[i], batch_size=batch_size, shuffle=True))

val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)
dataloaders_dict = {"train1":train_dataloader[0],
                    "train2":train_dataloader[1],
                    "train3": train_dataloader[2],
                    "train4": train_dataloader[3],
                    "train5": train_dataloader[4],
                    "val":train_dataloader[0]}

model = ZeroOrNot('train')

optimizer = optim.Adam(model.parameters(),lr=1e-2)
criterion = nn.BCELoss()

train_model(model,dataloaders_dict, criterion, optimizer, num_epochs)

save_path = 'zeroornotmodel4.pth'
torch.save(model.state_dict(), save_path)

log = pd.read_csv('log_data.csv',index_col=0)
plt.plot(log,label=log.columns)
plt.legend()
plt.show()
