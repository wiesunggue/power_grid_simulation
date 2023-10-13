# GAMS의 모든 변수를 입력으로 받아서 csv 파일을 생성하는 파일

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

year_list = [2012, 2016, 2020]
ess_size_list = [1.4,2.8]
abs_path = 'E://graduate//update_ess//RESULT//'
name = 'Case1_weather_%d_esssize_%.1f_min_ess_0_3day.gdx'

path_list = []
for i in year_list:
    for j in ess_size_list:
        path_list.append(abs_path+name%(i,j))
print(path_list[5])

ws = GamsWorkspace(working_directory=abs_path)
db = ws.add_database_from_gdx(path_list[5])

data_dict = {}
# 분석할 데이터 선언
temp_dict = {'Pump':defaultdict(float),'ESS':defaultdict(float),'sess':defaultdict(float)}
for i in db['st_dh']:
    time = int(i.keys[1])+(int(i.keys[2])-1)*24
    key = i.keys[0]
    value = i.value
    temp_dict[key].update({time:value})

fuel_cell = defaultdict(float)
for i in db['plan_Fuelcell_t_slid']:
    if int(i.keys[0])>24:
        continue
    time = int(i.keys[0])+(int(i.keys[1])-1)*24
    fuel_cell[time] = i.value

demand = defaultdict(float)
for i in db['Demandt_slid']:
    if int(i.keys[0])>24:
        continue
    time = int(i.keys[0])+(int(i.keys[1])-1)*24
    demand[time] = i.value

generation = {'Solar':defaultdict(float),'Wind':defaultdict(float),'off_wind':defaultdict(float)}
for i in db['z_dh']:
    if not i.keys[0] in generation:
        continue
    key = i.keys[0]
    time = int(i.keys[1])+(int(i.keys[2])-1)*24
    value = i.value
    generation[key].update({time:value})

data_dict.update(generation)
data_dict['demand']=demand
data_dict['fuel_cell']=fuel_cell
data_dict.update(temp_dict)

data_df = pd.DataFrame(data_dict)
data_df.fillna(0,inplace=True)
data_df.sort_index()

# 72개의 데이터로 풀어서 다시 딕셔너리 구성
row_data = {}
key_list = list(data_dict.keys())
print(key_list)


for key in key_list:
    if key in ['ESS','sess']:
        row_data[key]=defaultdict(float)
        continue
    for i in range(1,73):
        row_data[key+str(i)]=defaultdict(float)

for key in key_list:
    for ran in range(1,73):
        for hour in range(1,8761):
            if key in ['ESS', 'sess']:
                row_data[key][hour] = data_dict[key][hour]
                continue
            row_data[key+str(ran)][hour] = data_dict[key][hour+ran-24]

row_df = pd.DataFrame(row_data)
row_df.to_csv('get_every_thing_ess2_year2020.csv')
##row_df = row_df.loc[[(i+1)*24 for i in range(363)]]
'''
row_df -= row_df.mean()
row_df /= row_df.std()
# df.index[df['Values'] % 24 == 0]
X = row_df.drop(['ESS','sess'],axis=1)
y = row_df[['ESS']]
lasso_alphas, lasso_coefs, _ = linear_model.lasso_path(X,y)
log_alphas = -np.log10(lasso_alphas)


# Lasso 추정의 최적의 alpha를 구하기
lasso_best = linear_model.LassoCV(
    cv = 10, alphas = lasso_alphas)
print("학습 시작")
lasso_best.fit(X[24:-48],y[24:-48])

# Best인 alpha와 회귀 계수 출력하기
print(lasso_best.alpha_)
print(X.columns)
print(lasso_best.coef_)
ans = pd.DataFrame()

data=pd.DataFrame(lasso_best.coef_).transpose()
data.columns = X.columns
print(data)
data.to_csv('simulation_output.csv')
'''