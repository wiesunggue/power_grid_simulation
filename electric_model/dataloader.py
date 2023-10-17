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

# 비실행 파일
# 데이터 전처리기, 데이터 셋 구성 클래스

class DataTransform():
    def __init__(self, mean, std):
        resize=1
        self.data_transform = {
            'train': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ])
        }

    def __call__(self, data, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            전처리 모드 지정
        """
        return self.data_transform[phase](data)

class MakeDataSet(data.Dataset):
    '''Data Set Generator'''
    def __init__(self, data_pd, transform=None, zero=False,phase='train'):
        self.data_pd = data_pd
        self.transform = transform
        self.zero = zero
        self.phase = phase

    def __len__(self):
        """화상 개수를 반환"""
        return len(self.data_pd)

    def __getitem__(self, index):
        """
        전처리한 화상의 텐서 형식의 데이터와 라벨 취득
        """

        # index번째의 화상 로드
        model_data = self.data_pd[index-24:index+48]
        model_data = np.array(model_data[['fuel_cell','demand','Solar','Wind','off_wind']])
        target_data = [self.data_pd.loc[index,:]['ESS'],self.data_pd.loc[index,:]['sess']]
        if self.zero==False:
            return self.transform(model_data).squeeze(dim=1).float(), torch.Tensor(target_data).float()
        else:
            return self.transform(model_data).squeeze(dim=1).float(), torch.Tensor(target_data).float()==0

class MakevarDataSet(data.Dataset):
    '''Data Set Generator'''
    def __init__(self, data_pd, X=48, transform=None, zero=False,phase='train'):
        self.data_pd = data_pd
        self.transform = transform
        self.zero = zero
        self.phase = phase
        self.X = X
    def __len__(self):
        """화상 개수를 반환"""
        return len(self.data_pd)

    def __getitem__(self, index):
        """
        전처리한 화상의 텐서 형식의 데이터와 라벨 취득
        """

        # index번째의 화상 로드
        model_data = self.data_pd[index-24:index+self.X]
        model_data = np.array(model_data[['fuel_cell','demand','Solar','Wind','off_wind']])


        target_data = [self.data_pd.loc[index,:]['ESS'],self.data_pd.loc[index,:]['sess']]
        if self.zero==False:
            return self.transform(model_data).squeeze(dim=1).float(), torch.Tensor(target_data).float()
        else:
            return self.transform(model_data).squeeze(dim=1).float(), torch.Tensor(target_data).float()==0
class MakevarDataSet2(data.Dataset):
    '''Data Set Generator'''
    def __init__(self, data_pd, X=48, transform=None, zero=False,phase='train'):
        self.data_pd = data_pd
        self.transform = transform
        self.zero = zero
        self.phase = phase
        self.X = X
    def __len__(self):
        """화상 개수를 반환"""
        return len(self.data_pd)

    def __getitem__(self, index):
        """
        전처리한 화상의 텐서 형식의 데이터와 라벨 취득
        """

        # index번째의 화상 로드
        model_data = self.data_pd[index-24:index+self.X]
        model_data = np.array(model_data[['fuel_cell','demand','Solar','Wind','off_wind']])
        model_data = np.transpose(model_data)
        target_data = [self.data_pd.loc[index,:]['ESS'],self.data_pd.loc[index,:]['sess']]
        if self.zero==False:
            return self.transform(model_data).squeeze(dim=0).float(), torch.Tensor(target_data).float()
        else:
            return self.transform(model_data).squeeze(dim=0).float(), torch.Tensor(target_data).float()==0

class Make24DataSet(data.Dataset):
    '''24시의 데이터를 가져올 수 있도록 하는 Dataset'''
    def __init__(self,ds,transform=None, phase='train'):
        self.ds = ds
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.ds)//24-3 # 첫날과 마지막 2일 제거

    def __getitem__(self,index):
        return self.ds[(index+1)*24]

class Make24DataSet2(data.Dataset):
    '''24시의 데이터를 가져올 수 있도록 하는 Dataset'''
    def __init__(self,ds,transform=None, phase='train'):
        self.ds = ds
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.ds)//24-3 # 첫날과 마지막 2일 제거

    def __getitem__(self,index):
        return self.ds[(index+1)*24]

