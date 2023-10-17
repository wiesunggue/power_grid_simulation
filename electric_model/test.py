from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
import torchvision
from torchvision import models, transforms
from datanet import RegressionCNNX, RegressionNetX
timer = 72
data = torch.Tensor(np.ones((32, 5,timer)))
print(data.shape)
net = RegressionCNNX(timer*5,'train')
net(data)
