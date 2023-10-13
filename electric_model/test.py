from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
import torchvision
from torchvision import models, transforms

path = 'C://Users//wie02//Desktop//김동현.jpg'
img = Image.open(path)
print(img)
trans = transforms.ToTensor()
trans2 = transforms.Normalize((0.3,0.3,0.3),(0.5,0.7,2))
img_transformed = trans(img)
img_transformed = trans2(img_transformed)
img_transformed = np.transpose(img_transformed.numpy(),(1,2,0))
img_transformed = np.clip(img_transformed,0,1)
plt.imshow(img_transformed)
plt.savefig('blue.png')
plt.show()