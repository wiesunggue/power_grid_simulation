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
from torchvision import models, transforms

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from dataloader import *
from datanet import *

# 데이터 학습기
# 24시에 ESS 저장값을 예측하기 위한 추론 장치
# 가변 시간에 대해 진행 가능

ess_state = 1 # ess state가 1이면 1.4 2이면 2.8
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

# 학습하기
for timer in [0,1,3,6,24,48]:
    # 학습 파라메터
    batch_size = 32
    num_epochs = 300

    # 학습 정보 저장 경로
    log_path = f'log//train_valueCNN2_{timer}th_log_data.csv'
    save_model_path = f'modeL_save//regressionCNN2{timer}th_time.pth'
    save_tmp_path = f'modeL_save//regressionCNN2{timer}th_time_epoch%d.pth'

    # 데이터 셋 구성
    transform = DataTransform(0,1)
    train_dataset = []
    train_dataloader = []
    val_dataset = []
    val_dataloader = []
    for idx, var_dataset in enumerate(minimax_dataset):
        ds = MakevarDataSet2(var_dataset, timer,transform, False, 'train')
        if idx==(len(minimax_dataset)-1):
            val_dataset.append(Make24DataSet(ds,None,'val'))
        else:
            train_dataset.append(Make24DataSet(ds,transform,'train'))
    # 데이터 로더 구성하기 - 학습
    for _, train in enumerate(train_dataset):
        train_dataloader.append(data.DataLoader(
            train, batch_size=batch_size, shuffle=True))

    # 데이터 로더 구성하기 - 평가
    for _,val in enumerate(val_dataset):
        val_dataloader.append(data.DataLoader(
            val, batch_size=batch_size, shuffle=True))

    # 데이터 로더의 딕셔너리 구축
    dataloaders_dict =  {f"train{idx}":loader for idx,loader in enumerate(train_dataloader)} | {f"val{idx}":loader for idx,loader in enumerate(val_dataloader)}

    # 학습 모델 클래스 정의(값 예측용)
    # 모델에서 softmax제거 버젼
    model = RegressionCNNX((timer+24)*5,'train')

    # 최적화 탐색자 정의
    optimizer = optim.Adam(model.parameters(),lr=1e-3)

    # 평가 기준 정의
    criterion = nn.MSELoss()

    # 학습 시작
    train_model(model,dataloaders_dict, criterion, optimizer, num_epochs,log_path,save_tmp_path)


    # 학습 정보 저장하기
    torch.save(model.state_dict(), save_model_path)
