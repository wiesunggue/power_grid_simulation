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

# 검증 파일
# 가변 시간에 대해서 데이터 추론 장치
# 해당 파일을 통해 24시에 ESS에 저장할지 말지에 대한 결정을 알 수 있다.

ess_state = 1 # ess state가 1이면 1.4GW 2이면 2.8GW
file_path = []
for year in range(2012,2021):
    file_path.append(f'data//data_set_ess{ess_state}_PW{year}.csv')

print('데이터 경로 리스트 : ',file_path)


dataset = [] # 전체 데이터 셋
data24 = [] # 24시의 데이터 셋(ESS 정규화 용도)
minimax_dataset = [] # 정규화된 데이터 셋


# 데이터 읽기
for path in file_path:
    dataset.append(pd.read_csv(path,index_col=0))

# 24시의 데이터만 추출하기
for temp_data in dataset:
    data24.append(temp_data.loc[[(k + 1) * 24 for k in range(365)]])

## 데이터 정규화하기
for i in range(len(file_path)):
    tmp = pd.DataFrame()
    tmp[['Solar', 'Wind', 'demand', 'fuel_cell', 'off_wind']] = ((dataset[i] - dataset[-1].min()) / (dataset[-1].max() - dataset[-1].min()))[['Solar', 'Wind', 'demand', 'fuel_cell', 'off_wind']]
    tmp[['ESS', 'sess']] = ((dataset[i]) / (data24[-1].max()))[['ESS', 'sess']] # ESS는 24시 데이터만 활용 나머지는 전 데이터 활용

    minimax_dataset.append(tmp)


for timer in [0,1,3,6,24,48]: # 수정해야 함 체크
    transform = DataTransform(0, 1)
    train_dataset = []
    train_dataloader = []
    for idx, var_dataset in enumerate(minimax_dataset):
        ds = MakevarDataSet2(var_dataset, timer, transform, False, 'train')
        if idx == len(minimax_dataset)-1:
            val_dataset = Make24DataSet(ds, None, 'val')
        else:
            train_dataset.append(Make24DataSet(ds, transform, 'train'))
    save_path1 = f'model_save//zeroornotmodelCNN{timer}th_time_epoch100.pth' ## 수정해야 함 체크
    save_path2 = f'modeL_save//regressionCNN2{timer}th_time_epoch100.pth'

    model1 = ZeroOrNotCNNX((timer+24)*5,'phase') # 수정해야 함 체크
    model1.load_state_dict(torch.load(save_path1))
    model1.eval()

    model2 = RegressionCNNX((timer+24)*5,'phase')
    model2.load_state_dict(torch.load(save_path2))
    model2.eval()

    data_max = np.array(data24[-1].max()[['ESS', 'sess']])
    data_min = np.array(data24[-1].min()[['ESS', 'sess']])
    #print(data_min,data_max)
    cnt = 0
    acc = 0
    log_estim = {'Ess':[],'sess':[],'ans1':[],'ans2':[]}

    for i in range(362):
        zero = (model1(val_dataset[i][0]) < 0.5)[0] # 0 or 1 판별하기
        estim = (model2(val_dataset[i][0]).detach().numpy()*(data_max))[0] # 구체적인 값 판별하기
        #print('True 는 0이 아님',zero, torch.Tensor(dataset[2].loc[(i+1)*24,:][['ESS','sess']])!=0)
        #print(i,zero*estim,torch.Tensor(dataset[2].loc[(i+1)*24,:][['ESS','sess']]))
        #print(val_dataset[i][1][0],(model1(val_dataset[i][0]))[0][0])
        #print(zero, estim)

        cnt += (val_dataset[i][1][0]!=0)==((model1(val_dataset[i][0]))[0][0]<0.5)
        estimation = zero*estim
        log_estim['ans1'].append(float(val_dataset[i][1][0])*data_max[0])
        log_estim['ans2'].append(float(val_dataset[i][1][1])*data_max[1])
        log_estim['Ess'].append(float(estimation[0]) if float(estimation[0])>0 else 0)
        log_estim['sess'].append(float(estimation[1]) if float(estimation[1])>0 else 0)

    print('정답률',cnt/363*100, max(log_estim['Ess']),max(log_estim['sess']))
    log_estim = pd.DataFrame(log_estim)
    log_estim.index = log_estim.index + 1
    log_estim.to_csv(f'estimation_result//estimation{timer}th_time.csv')
