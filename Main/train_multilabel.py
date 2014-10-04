#############################################
# Program: set_baseline_error.py
#
# Notes:
#
# Date:
#
#############################################

import os as os
import numpy as np
from pandas import Series, DataFrame
from datetime import datetime
import pandas as pd
from scipy import linspace, io
from math import *
from cmath import phase

def norm(x, by):
    y = (x-np.mean(by)) / np.std(by)
    return y

base_dir = "/home/bulby/Documents/Kaggle/Belkin/Data"
train_files = {
    "H1": ['04_13_1334300401', '10_22_1350889201', '10_23_1350975601', '10_24_1351062001', '10_25_1351148401', '12_27_1356595201'],
    "H2": ['02_15_1360915201', '06_13_1339570801', '06_14_1339657201', '06_15_1339743601'],
    "H3": ['07_30_1343631601', '07_31_1343718001', '08_01_1343804401'],
    "H4": ['07_26_1343286001']}
houses = ["H1", "H2", "H3", "H4"]

test_days = ['H1_10_22_1350889201', 'H1_10_24_1351062001', 'H2_02_15_1360915201', 'H2_06_14_1339657201', 'H3_08_01_1343804401']

os.chdir(base_dir)
TrainSub = pd.read_csv('TrainingSet.csv')
NumAppliance = max(TrainSub['Appliance'])
TrainSub['Test'] = False

TrainAggDtypes = {}
TrainAggCols = pd.read_csv('Aggregated_v2.csv', nrows=1)
for TrainAggCol in TrainAggCols.columns:
    if TrainAggCol[3:7]=='mean':
        TrainAggDtypes.update({TrainAggCol: 'f4'})
    elif TrainAggCol[3:8]=='range' or TrainAggCol[3:6] in ('max','min') or TrainAggCol[10:] in ('mean','max','min','range'):
        TrainAggDtypes.update({TrainAggCol: 'f2'})
TrainAgg = pd.read_csv('Aggregated_v2.csv', dtype=TrainAggDtypes)

TrainAgg['Test'] = False
TrainAgg = TrainAgg.rename(columns={'Eval_TimeStamp': 'TimeStamp'})
for test_day in test_days:
    TrainSub['Test'][TrainSub['House']+'_'+TrainSub['Day']==test_day] = True
    TrainAgg['Test'][TrainAgg['House']+'_'+TrainAgg['Day']==test_day] = True

GB_Vars = ['TimeStamp','House','Day','Test'] + ['On_'+str(Appliance) for Appliance in range(1, NumAppliance+1)]
for Appliance in range(1, NumAppliance+1):
    # TrainSub['On_'+str(Appliance)] = (TrainSub['Predicted']==1) & (TrainSub['Appliance']==Appliance)
    TrainSub['On_'+str(Appliance)] = 0
    TrainSub['On_'+str(Appliance)][(TrainSub['Predicted']==1) & (TrainSub['Appliance']==Appliance)] = 1

TrainSub_TS = TrainSub[GB_Vars].groupby(['TimeStamp','House','Day']).max()

TrainAgg.index = [TrainAgg['TimeStamp'], TrainAgg['House'], TrainAgg['Day']]

TrainAppSub = pd.merge(TrainSub_TS[(TrainSub_TS['Test']==False)], TrainAgg[TrainAgg['Test']==False], left_index=True, right_index=True, how='inner')
TestAppSub = pd.merge(TrainSub_TS[(TrainSub_TS['Test']==True)], TrainAgg[TrainAgg['Test']==True], left_index=True, right_index=True, how='inner')

vars_to_include = [False]*len(TrainAppSub.columns)
for col in TrainAppSub.columns:
    # if col[10:]=='mean' or col[:7]=='HF_mean':
    if col[:7]=='HF_mean':
        TrainAppSub[col+'_norm'] = norm(TrainAppSub[col], TrainAppSub[col])
        TestAppSub[col+'_norm'] = norm(TestAppSub[col], TrainAppSub[col])
        vars_to_include.append(True)

#### non-Invertible matrix
# import statsmodels.api as sm
# logit = sm.Logit(TrainAppSub[['On_'+str(Appliance) for Appliance in range(1, NumAppliance+1)]], TrainAppSub[TrainAppSub.columns[vars_to_include]])
# result = logit.fit()
# print result.summary()

# from sklearn import svm
# clf = svm.SVC()
# clf.fit(TrainAppSub[TrainAppSub.columns[vars_to_include]], TrainAppSub['Predicted'])
# clf.fit(TrainAppSub[TrainAppSub.columns[vars_to_include]], TrainAppSub['On_1'])

### TrainAppSub[TrainAppSub.columns[vars_to_include]]
### TrainAppSub['On_1']
### TrainAppSub[['On_'+str(Appliance) for Appliance in range(1, NumAppliance+1)]]

from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import TanhLayer

ds_train = SupervisedDataSet(TrainAppSub[TrainAppSub.columns[vars_to_include]].shape[1], 1)
# for row in range(TrainAppSub[TrainAppSub.columns[vars_to_include]].shape[0]):
for row in range(200):
    ds_train.addSample(np.array(TrainAppSub[TrainAppSub.columns[vars_to_include]])[row,:], TrainAppSub['On_1'][row])

ds_test = SupervisedDataSet(TestAppSub[TrainAppSub.columns[vars_to_include]].shape[1], 1)
for row in range(TestAppSub[TestAppSub.columns[vars_to_include]].shape[0]):
    ds_test.addSample(np.array(TestAppSub[TestAppSub.columns[vars_to_include]])[row,:], TestAppSub['On_1'][row])

net = buildNetwork(TrainAppSub[TrainAppSub.columns[vars_to_include]].shape[1], TrainAppSub[TrainAppSub.columns[vars_to_include]].shape[1]-100, TrainAppSub[TrainAppSub.columns[vars_to_include]].shape[1]-1000, TrainAppSub[TrainAppSub.columns[vars_to_include]].shape[1]-3000,1)
trainer = BackpropTrainer(net, ds_train)
trainer.train()
trainer.trainUntilConvergence()


###  ds_train = SupervisedDataSet(TrainAppSub[TrainAppSub.columns[vars_to_include]].shape[1], TrainAppSub[['On_'+str(Appliance) for Appliance in range(1, NumAppliance+1)]].shape[1])
###  ds_test = SupervisedDataSet(TestAppSub[TrainAppSub.columns[vars_to_include]].shape[1], TestAppSub[['On_'+str(Appliance) for Appliance in range(1, NumAppliance+1)]].shape[1])
###  # for row in range(TrainAppSub[TrainAppSub.columns[vars_to_include]].shape[0]):
###  ds_train.clear()
###  for row in range(140):
###      ds_train.addSample(np.array(TrainAppSub[TrainAppSub.columns[vars_to_include]])[row,:], np.array(TrainAppSub[['On_'+str(Appliance) for Appliance in range(1, NumAppliance+1)]])[row])
###  # for row in range(TestAppSub[TestAppSub.columns[vars_to_include]].shape[0]):
###  #     ds_test.addSample(np.array(TestAppSub[TestAppSub.columns[vars_to_include]])[row,:], np.array(TestAppSub[['On_'+str(Appliance) for Appliance in range(1, NumAppliance+1)]])[row])
###  # 
###  
###  net = buildNetwork(TrainAppSub[TrainAppSub.columns[vars_to_include]].shape[1], TrainAppSub[TrainAppSub.columns[vars_to_include]].shape[1]+50, TrainAppSub[['On_'+str(Appliance) for Appliance in range(1, NumAppliance+1)]].shape[1])
###  trainer = BackpropTrainer(net, ds_train)
###  trainer.train()


#151: nan
#148: nan
#143: nan
#141: nan
#140: 1.4570168033458767e+295
#139: 2.9871187124105627e+297
#135: 6.0780061690345589e+301

