import gams
from gams import GamsWorkspace
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn import linear_model

data1 = pd.read_csv('get_every_thing_ess2_year2012.csv',index_col=0)
data2 = pd.read_csv('get_every_thing_ess2_year2016.csv',index_col=0)
data3 = pd.read_csv('get_every_thing_ess2_year2020.csv',index_col=0)

train_set = pd.concat([data1[23:-48],data2[23:-48]])
val_set = data3[23:-48]

#train_set = train_set[train_set.index%24==0]
#val_set = val_set[val_set.index%24==0]

train_set -= train_set.mean()
train_set /= train_set.std()

val_set -= train_set.mean()
val_set /= train_set.std()

print(train_set.head())
print(val_set.head())

train_X = train_set.drop(['ESS','sess'],axis=1)
train_y = train_set[['ESS']]

val_X = val_set.drop(['ESS','sess'],axis=1)
val_y = val_set['ESS']

# alpha 값을 찾아서 alpha 탐색을 없애자
alpha = 0.0007236797507860115
#lasso_alphas, lasso_coefs, _ = linear_model.lasso_path(train_X,train_y)
#log_alphas = -np.log10(lasso_alphas)

#lasso_best = linear_model.LassoCV(cv = 10, alphas = alpha)

#lasso_best.fit(train_X,train_y)
lasso = linear_model.Lasso(alpha=alpha)
lasso.fit(train_X,train_y)
print(lasso.coef_)
#print(pd.DataFrame(lasso.predict(val_X))*train_set.std()['ESS']+train_set.mean()['ESS'])
#print(val_y*train_set.std()['ESS']+train_set.mean()['ESS'])

ans_y = pd.DataFrame(lasso.predict(val_X))*train_set.std()['ESS']+train_set.mean()['ESS']
#print(ans_y)
#val_y = val_y[val_y.index%24==0]
#ans_y = ans_y[ans_y.index%24==0]

print(*lasso.predict(train_X))
print(train_y)

print(lasso.predict(val_X))
print(val_y)