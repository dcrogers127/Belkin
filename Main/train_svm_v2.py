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
from sklearn import svm

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

# TrainAgg = pd.read_csv('Aggregated_v2.csv')
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

# for Appliance in range(1, NumAppliance+1):
Appliance=9
## this step drops 4 obs from the H4 day.
TrainAppSub = pd.merge(TrainSub[(TrainSub['Test']==False) & (TrainSub['Appliance']==Appliance)], TrainAgg[TrainAgg['Test']==False], on=['TimeStamp','House','Day'], how='inner')
TestAppSub = pd.merge(TrainSub[(TrainSub['Test']==True) & (TrainSub['Appliance']==Appliance)], TrainAgg[TrainAgg['Test']==True], on=['TimeStamp','House','Day'], how='inner')

vars_to_include = [False]*len(TrainAppSub.columns)
for col in TrainAppSub.columns:
    if col[10:]=='mean' or col[:7]=='HF_mean':
        TrainAppSub[col+'_norm'] = norm(TrainAppSub[col], TrainAppSub[col])
        TestAppSub[col+'_norm'] = norm(TestAppSub[col], TrainAppSub[col])
        vars_to_include.append(True)

clf = svm.SVC()
clf.fit(TrainAppSub[TrainAppSub.columns[vars_to_include]], TrainAppSub['Predicted'])
TrainPred = clf.predict(TrainAppSub[TrainAppSub.columns[vars_to_include]])
TestPred = clf.predict(TestAppSub[TrainAppSub.columns[vars_to_include]])

BaseErrRate = float(sum(TestAppSub['Predicted']!=0)) / len(TestPred)
ErrRate = float(sum(TestAppSub['Predicted']!=TestPred)) / len(TestPred)
print (BaseErrRate, ErrRate)
pd.crosstab(TrainAppSub['Predicted'], TrainPred)
pd.crosstab(TestAppSub['Predicted'], TestPred)

os.chdir(base_dir+'/Reports')
TrainOut = DataFrame(TrainAppSub[['TimeStamp','House','Day','Predicted']])
TrainOut = TrainOut.rename(columns={'Predicted': 'Actual'})
TrainOut['Predicted'] = TrainPred
TrainOut.to_csv('TrainOut_svm_v2.csv', index=False)
TestOut = DataFrame(TestAppSub[['TimeStamp','House','Day','Predicted']])
TestOut = TestOut.rename(columns={'Predicted': 'Actual'})
TestOut['Predicted'] = TestPred
TestOut.to_csv('TestOut_svm_v2.csv', index=False)

##  
##  In [25]: print (BaseErrRate, ErrRate)
##  (0.006666666666666667, 0.015277777777777777)
##  
##  In [26]: pd.crosstab(TrainAppSub['Predicted'], TrainPred)
##  Out[26]: 
##  col_0         0    1
##  Predicted           
##  0          5847    3
##  1           283  105
##  
##  In [27]: pd.crosstab(TestAppSub['Predicted'], TestPred)
##  Out[27]: 
##  col_0         0   1
##  Predicted          
##  0          3545  31
##  1            24   0
##  

###### USE TO LOOK FOR INF OR NA VALUEES
ColNum = 0
for IncludedVar in vars_to_include:
    if IncludedVar:
        num_NaA = sum(TrainAppSub[TrainAppSub.columns[ColNum]].isnull()) 
        num_inf = sum(np.isinf(TrainAppSub[TrainAppSub.columns[ColNum]]))
        print "col "+TrainAppSub.columns[ColNum]+" has %s null and %s are inf" % (num_NaA, num_inf)
    ColNum = ColNum+1
######
