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

transform = DataTransform(0,1)
train_dataset = []
train_dataloader = []
for i in range(len(file_list1)):
    ds = MakeDataSet(minimax_dataset[0], transform, False, 'train')
    if i==2:
        val_dataset = Make24DataSet(ds,None,'val')
    else:
        train_dataset.append(Make24DataSet(ds,transform,'train'))



save_path1 = 'zeroornotmodel4.pth'
save_path2 = 'regression2.pth'

model1 = ZeroOrNot('phase')
model1.load_state_dict(torch.load(save_path1))
model1.eval()

model2 = RegressionNet('phase')
model2.load_state_dict(torch.load(save_path2))
model2.eval()

data_max = np.array(data24[2].max()[['ESS', 'sess']])
data_min = np.array(data24[2].min()[['ESS', 'sess']])
print(data_min,data_max)
cnt = 0
acc = 0
log_estim = {'estim1':[],'estim2':[],'ans1':[],'ans2':[]}

for i in range(362):
    zero = (model1(val_dataset[i][0]) < 0.5)[0]
    estim = (model2(val_dataset[i][0]).detach().numpy()*(data_max))[0]
    print('True 는 0이 아님',zero, torch.Tensor(dataset[2].loc[(i+1)*24,:][['ESS','sess']])!=0)
    print(i,zero*estim,torch.Tensor(dataset[2].loc[(i+1)*24,:][['ESS','sess']]))
    cnt += (val_dataset[i][1][0]!=0)==((model1(val_dataset[i][0]))[0][0]<0.5)
    estimation = zero*estim
    log_estim['ans1'].append(float(val_dataset[i][1][0])*23.630615)
    log_estim['ans2'].append(float(val_dataset[i][1][1])*1.425309)
    log_estim['estim1'].append(float(estimation[0]))
    log_estim['estim2'].append(float(estimation[1]))

print('정답률',cnt/363*100)
log_estim = pd.DataFrame(log_estim)
log_estim.to_csv('estimation.csv')