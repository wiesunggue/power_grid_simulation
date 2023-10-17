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
import dataloader

# 비실행 파일
# 데이터 모델 구조 정의하기
# 학습 함수 정의하기
class RegressionNet(nn.Module):
    '''데이터 네트워크 만들기'''
    def __init__(self,phase):
        super(RegressionNet,self).__init__()
        self.phase = phase
        self.Linearnet = nn.Sequential(
            nn.Linear(72*5,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,128),
            nn.LeakyReLU(),
            nn.Linear(128,16),
            nn.LeakyReLU(),
            nn.Linear(16,2)
        )

    def forward(self, x):
        x=x.reshape([-1,72*5])
        x = self.Linearnet(x)
        return x

class RegressionNetX(nn.Module):
    '''데이터 네트워크 만들기'''
    def __init__(self,X,phase):
        super(RegressionNetX,self).__init__()
        self.phase = phase
        self.X = X
        self.Linearnet = nn.Sequential(
            nn.Linear(self.X,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,128),
            nn.LeakyReLU(),
            nn.Linear(128,16),
            nn.LeakyReLU(),
            nn.Linear(16,2)
        )

    def forward(self, x):
        x=x.reshape([-1,self.X])
        x = self.Linearnet(x)
        return x

class RegressionCNNX(nn.Module):
    '''ON/OFF 예측용 CNN'''
    def __init__(self,timer,phase):
        super(RegressionCNNX,self).__init__()
        self.timer_dict = {24*5:1, 25*5:1, 27*5:1, 30*5:1, 48*5:2, 72*5:4}
        self.phase = phase
        self.timer = timer
        self.Layer1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=64, kernel_size=5,stride=3,padding=1,dilation=1,bias=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.Layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=5,stride=3,padding=1,dilation=1,bias=True),
            nn.ReLU(),
            )

        self.Layer3 = nn.Sequential(
            nn.Linear(256*self.timer_dict[self.timer],128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16,2),
            nn.Softmax(dim=0)
        )
    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x=x.reshape([-1,256*self.timer_dict[self.timer]])
        x = self.Layer3(x)
        return x

class DeepNeuralNet(nn.Module):
    '''실패한 듯 좋지 않다.'''
    def __init__(self,phase):
        super(DeepNeuralNet,self).__init__()
        self.phase = phase
        self.Linearnet = nn.Sequential(
            nn.Linear(72*5,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512,128),
            nn.LeakyReLU(),
            nn.Linear(128, 16),
            nn.LeakyReLU(),
            nn.Linear(16,2)
        )

    def forward(self, x):
        x=x.reshape([-1,72*5])
        x = self.Linearnet(x)
        return x

class ZeroOrNot(nn.Module):
    '''데이터 네트워크 만들기'''
    def __init__(self,phase):
        super(ZeroOrNot,self).__init__()
        self.phase = phase
        self.Linearnet = nn.Sequential(
            nn.Linear(72*5,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16,2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=x.reshape([-1,72*5])
        x = self.Linearnet(x)
        return x
class ZeroOrNotX(nn.Module):
    '''데이터 네트워크 만들기'''
    def __init__(self,X,phase):
        super(ZeroOrNotX,self).__init__()
        self.phase = phase
        self.X = X
        self.dropout = nn.Dropout(0.25)
        self.Linearnet = nn.Sequential(
            nn.Linear(self.X,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16,2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=x.reshape([-1,self.X])
        x = self.Linearnet(x)
        return x

class NewModel(nn.Module):
    '''데이터 네트워크 만들기'''
    def __init__(self,X,phase):
        super(NewModel,self).__init__()
        self.phase = phase
        self.X = X
        self.dropout = nn.Dropout(0.25)
        self.Linearnet = nn.Sequential(
            nn.Linear(self.X,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            self.dropout,
            nn.Linear(1024,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            self.dropout,
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            self.dropout,
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            self.dropout,
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            self.dropout,
            nn.Linear(32, 4),
            nn.LeakyReLU(),
            nn.Linear(4,2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=x.reshape([-1,self.X])
        x = self.Linearnet(x)
        return x
class SimpleZeroOrNot(nn.Module):
    '''데이터 네트워크 만들기'''
    def __init__(self,phase):
        super(SimpleZeroOrNot,self).__init__()
        self.phase = phase
        self.Linearnet = nn.Sequential(
            nn.Linear(72*5,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=x.reshape([-1,72*5])
        x = self.Linearnet(x)
        return x

class SimpleZeroOrNotX(nn.Module):
    '''데이터 네트워크 만들기'''
    def __init__(self,X, phase):
        super(SimpleZeroOrNotX,self).__init__()
        self.X = X
        self.phase = phase
        self.dropout = nn.Dropout(0.25)
        self.Linearnet = nn.Sequential(
            nn.Linear(self.X,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=x.reshape([-1,self.X])
        x = self.Linearnet(x)
        return x

class SimpleZeroOrNotX(nn.Module):
    '''데이터 네트워크 만들기'''
    def __init__(self,X, phase):
        super(SimpleZeroOrNotX,self).__init__()
        self.X = X
        self.phase = phase
        self.dropout = nn.Dropout(0.25)
        self.Linearnet = nn.Sequential(
            nn.Linear(self.X,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=x.reshape([-1,self.X])
        x = self.Linearnet(x)
        return x
def train_model(model,dataloaders_dict, criterion, optimizer, num_epochs, log_path = 'log_data.csv', save_tmp_path='model_save//temp_epoch_%d.pth'):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("사용 장치: ", device)

    model.to(device)
    before_best = 100000
    torch.backends.cudnn.benchmark=True
    log_data = defaultdict(list)
    for epoch in range(num_epochs+1):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('-'*30)
        epoch_loss_dict={}
        for phase in dataloaders_dict:
            if phase.startswith('val'):
                model.eval()
            else:
                model.train()

            epoch_loss=0.0
            epoch_corrects=0

            if(epoch==0) and (phase.startswith('val')):
                continue
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase.startswith('train')):
                    outputs = model(inputs)
                    loss=criterion(outputs,labels.to(torch.float32))
                    if phase.startswith('train'):
                        loss.backward()
                        optimizer.step()
                    epoch_loss +=loss.item()*inputs.size(0)

            epoch_loss = epoch_loss/len(dataloaders_dict[phase])
            epoch_loss_dict[phase]=epoch_loss
        for key in epoch_loss_dict:
            print('{} Loss: {:.4f}'.format(key,epoch_loss_dict[key]), end=' ')
            if epoch!=0:
                log_data[key].append(epoch_loss_dict[key])

        # 100 번마다 파일 임시 저장하기
        if epoch%100 == 0:
            torch.save(model.state_dict(), save_tmp_path % (epoch))
            print('임시 파일 저장!')
        print()
        print()
    log_data = pd.DataFrame(log_data)
    log_data.index.name = 'epoch' # 인덱스 이름 설정
    log_data.index = log_data.index + 1 # 인덱스 변경(epoch와 동기화)
    log_data.to_csv(log_path)

