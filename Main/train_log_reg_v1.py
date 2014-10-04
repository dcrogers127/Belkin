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
import statsmodels.api as sm

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

# TrainAgg = pd.read_csv('Aggregated_v2.csv')
TrainAggDtypes = {}
TrainAggCols = pd.read_csv('Aggregated_v2.csv', nrows=1)
for TrainAggCol in TrainAggCols.columns:
    if TrainAggCol[3:7]=='mean' or TrainAggCol[3:8]=='range' or TrainAggCol[3:6] in ('max','min') or TrainAggCol[10:] in ('mean','max','min','range'):
        TrainAggDtypes.update({TrainAggCol: 'f2'})
TrainAgg = pd.read_csv('Aggregated_v2.csv', dtype=TrainAggDtypes)

HF_vars = [False]*len(TrainAgg.columns)
colInd = 0
for col in TrainAgg.columns:
    if col[:7]=='HF_mean':
        HF_vars[colInd] = True
    colInd = colInd+1

HF_mean_array = np.array(TrainAgg[TrainAgg.columns[HF_vars]])
TrainAgg['HF_mean_mean'] = np.mean(HF_mean_array, axis=1, dtype='f3')
TrainAgg['HF_mean_std'] = np.std(HF_mean_array, axis=1, dtype='f3')
TrainAgg['HF_mean_max'] = np.max(HF_mean_array, axis=1)
TrainAgg['HF_mean_min'] = np.min(HF_mean_array, axis=1)
TrainAgg['HF_mean_med'] = np.median(HF_mean_array, axis=1)
TrainAgg['HF_mean_p75'] = np.percentile(HF_mean_array, 75, axis=1)
TrainAgg['HF_mean_p25'] = np.percentile(HF_mean_array, 25, axis=1)

TrainAgg['Test'] = False
TrainAgg = TrainAgg.rename(columns={'Eval_TimeStamp': 'TimeStamp'})
for test_day in test_days:
    TrainSub['Test'][TrainSub['House']+'_'+TrainSub['Day']==test_day] = True
    TrainAgg['Test'][TrainAgg['House']+'_'+TrainAgg['Day']==test_day] = True

# for Appliance in range(1, NumAppliance+1):
Appliance=9
# this step drops 4 obs from the H4 day.
TrainAppSub = pd.merge(TrainSub[(TrainSub['Test']==False) & (TrainSub['Appliance']==Appliance)], TrainAgg[TrainAgg['Test']==False], on=['TimeStamp','House','Day'], how='inner')
TestAppSub = pd.merge(TrainSub[(TrainSub['Test']==True) & (TrainSub['Appliance']==Appliance)], TrainAgg[TrainAgg['Test']==True], on=['TimeStamp','House','Day'], how='inner')

vars_to_include = [False]*len(TrainAppSub.columns)
colInd = 0
for col in TrainAppSub.columns:
    if col[10:]=='mean' or (col in ('HF_mean_mean','HF_mean_std','HF_mean_max','HF_mean_min','HF_mean_med','HF_mean_p75','HF_mean_p25')):
        vars_to_include[colInd] = True
    colInd = colInd+1

logit = sm.Logit(TrainAppSub['Predicted'], TrainAppSub[TrainAppSub.columns[vars_to_include]])
result = logit.fit()
print result.summary()
TestPred = result.predict(TestAppSub[TrainAppSub.columns[vars_to_include]])>.5
sum(TestPred)
sum(TestAppSub['Predicted'])
len(TestPred)

BaseErrRate = float(sum(TestAppSub['Predicted']!=0)) / len(TestPred)
ErrRate = float(sum(TestAppSub['Predicted']!=TestPred)) / len(TestPred)
pd.crosstab(TestAppSub['Predicted'],TestPred)

### for col in TrainAppSub.columns[7:199]:
###     num_NaA = sum(TrainAppSub[col].isnull()) 
###     num_is0 = sum(TrainAppSub[col]==0) 
###     print "col "+col+" has %s null and %s are 0" % (num_NaA, num_is0)
### TrainAppSub[TrainAppSub.columns[7:199]].shape



