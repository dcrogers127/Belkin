#############################################
# Program: aggregate_data_v2_cmb.py
#
# Notes:
#
#
#
#
# Date:
#
#############################################

import os as os
import numpy as np
from pandas import Series, DataFrame
from datetime import datetime
import pandas as pd

base_dir = "/home/bulby/Documents/Kaggle/Belkin/Data"
train_files = {
    "H1": ['04_13_1334300401', '10_22_1350889201', '10_23_1350975601', '10_24_1351062001', '10_25_1351148401', '12_27_1356595201'],
    "H2": ['02_15_1360915201', '06_13_1339570801', '06_14_1339657201', '06_15_1339743601'],
    "H3": ['07_30_1343631601', '07_31_1343718001', '08_01_1343804401'],
    "H4": ['07_26_1343286001']}
houses = ["H1", "H2", "H3", "H4"]

os.chdir(base_dir+"/Temp")
for house in houses:
    for i in range(len(train_files[house])):
        day = train_files[house][i]
        if i==0 and house=='H1':
            AggFiles = pd.read_csv('Agg_v2_'+house+'_'+day+'.csv')
        else:
            AggFiles = pd.concat([AggFiles, pd.read_csv('Agg_v2_'+house+'_'+day+'.csv')])

os.chdir(base_dir)
AggFiles.to_csv('Aggregated_v2.csv', index=False)

os.chdir(base_dir+'/Agg_v2')
BaseVars = ['Eval_TimeStamp','Eval_Id','House','Day','LF1_size','LF2_size','HF_size']
VarGroups = ['LF1V','LF1I','LF2V','LF2I','HF_mean','HF_max','HF_min','HF_range']
TotVars = 4

for VarGroup in VarGroups:
    KeepList = [((AggFiles.columns[i] in BaseVars) or AggFiles.columns[i][:4]==VarGroup or AggFiles.columns[i][:6]==VarGroup or AggFiles.columns[i][:7]==VarGroup or AggFiles.columns[i][:8]==VarGroup) for i in range(AggFiles.columns.shape[0])]
    NumKept = sum(KeepList)
    print VarGroup+' has %s variables' % (NumKept)
    TotVars = TotVars + NumKept - 7
    AggFiles[AggFiles.columns[KeepList]].to_csv('Agg_v2_'+VarGroup+'.csv', index=False)

print "Tot vars equals num vars: %s" % (TotVars==AggFiles.columns.shape[0])

