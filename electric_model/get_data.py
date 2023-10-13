import gams
from gams import GamsWorkspace
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from collections import defaultdict

# gdx파일을 활용해서 데이터 추출 및 처리 함수
# 한번 실행하여 csv 파일을 생성

study_param = 2020
ess_state = 1.4
path = 'E://graduate//update_ess//RESULT//'
name = f'Case1_weather_{study_param}_esssize_{ess_state}_min_ess_0_3day.gdx'

ws = GamsWorkspace(working_directory=path)
db = ws.add_database_from_gdx(path+name)

# 데이터를 저장할 딕셔너리
dataset = {}

# 분석할 데이터 선언
temp_dict = {'ESS':defaultdict(float),'sess':defaultdict(float)}
for i in db['st_dh']:
    if i.keys[0]=='Pump':
        continue
    time = int(i.keys[1])+(int(i.keys[2])-1)*24
    key = i.keys[0]
    value = i.value
    temp_dict[key].update({time:value})

fuel_cell = {}
for i in db['plan_Fuelcell_t_slid']:
    if int(i.keys[0])>24:
        continue
    time = int(i.keys[0])+(int(i.keys[1])-1)*24
    fuel_cell[time] = i.value

demand = {}
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

dataset['fuel_cell']=fuel_cell
dataset['demand']=demand

dataset.update(generation)
dataset.update(temp_dict)

df = pd.DataFrame(dataset)
df = df.sort_index()
df.fillna(0,inplace=True)

print(df.head())
df.to_csv(f'data//data_set_ess{int(ess_state)}_PW{study_param}.csv')