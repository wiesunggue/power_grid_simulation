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

dataset = pd.read_csv('data_set_ess1_PW2012.csv', index_col=0)
data24 = dataset.loc[[(i+1)*24 for i in range(365)]]

normalized_dataset = (dataset-dataset.mean())/dataset.std()

minimax_dataset = pd.DataFrame()
minimax_dataset[['Pump','Solar','Wind','demand','fuel_cell','off_wind']] = ((dataset-dataset.min())/(dataset.max()-dataset.min()))[['Pump','Solar','Wind','demand','fuel_cell','off_wind']]
minimax_dataset[['ESS','sess']] = ((dataset-data24.min())/(data24.max()-data24.min()))[['ESS','sess']]

transform = DataTransform(0,1)
ds = MakeDataSet(minimax_dataset,transform,'train')
train_dataset = Make24DataSet(ds,transform,'train')

#save_path = 'deepmodel.pth'
#save_path = 'regression.pth'
save_path = 'zeroornotmodel.pth'

#model = DeepNeuralNet('phase')
#model = RegressionNet('phase')
model = ZeroOrNot('phase')
model.load_state_dict(torch.load(save_path))
data_max = np.array(data24.max()[['ESS','sess']])
data_min = np.array(data24.min()[['ESS','sess']])

for i in range(362):
    tmp = (model(train_dataset[i][0]).detach().numpy()*(data_max-data_min)+data_min)
    if tmp[0][0]>0.3:
        pass
    print(i,(model(train_dataset[i][0])>0.5),torch.Tensor(np.array(dataset.loc[(i+1)*24,:][['ESS','sess']])==0))
