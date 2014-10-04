#############################################
# Program: build_train_test_cohorts.py
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
import pandas as pd
from scipy import linspace, io

min_time = .2
max_time = .8
interval = 15
base_dir = "/home/bulby/Documents/Kaggle/Belkin/Data"

os.chdir(base_dir)
SampleSub = pd.read_csv("SampleSubmission.csv")
NumAppliance = max(SampleSub['Appliance'])

# H4 07_27_1343372401 doesnt have tick marks
train_files = {
    "H1": ['04_13_1334300401', '10_22_1350889201', '10_23_1350975601', '10_24_1351062001', '10_25_1351148401', '12_27_1356595201'],
    "H2": ['02_15_1360915201', '06_13_1339570801', '06_14_1339657201', '06_15_1339743601'],
    "H3": ['07_30_1343631601', '07_31_1343718001', '08_01_1343804401'],
    "H4": ['07_26_1343286001']}
houses = ["H1", "H2", "H3", "H4"]

for house in houses:
    os.chdir(base_dir+"/"+house)
    for i in range(len(train_files[house])):
        train_data = io.loadmat('Tagged_Training_'+train_files[house][i]+'.mat')
        buf = train_data['Buffer']

        L1_TimeTicks = DataFrame(buf['TimeTicks1'][0][0], columns=['TimeStamp'])
        min_L1TT = min(L1_TimeTicks['TimeStamp'])
        max_L1TT = max(L1_TimeTicks['TimeStamp'])
        min_samp = int(min_L1TT + (max_L1TT-min_L1TT)*min_time)
        max_samp = int(min_L1TT + (max_L1TT-min_L1TT)*max_time)
        sampTT = range(min_samp, max_samp, interval)*NumAppliance
        sampTT.sort()

        TrainingSub = DataFrame(sampTT, columns=['TimeStamp'])
        TrainingSub['Appliance'] = range(1, NumAppliance+1)*(len(sampTT)/NumAppliance)
        TrainingSub['House'] = house
        TrainingSub['Day'] = train_files[house][i]
        TrainingSub['Predicted'] = 0

        taggingInfo = buf['TaggingInfo'][0][0]
        for j in range(0, taggingInfo.shape[0]):
            TrainingSub['Predicted'][(TrainingSub['Appliance']==taggingInfo[j][0][0][0]) & (taggingInfo[j][2][0][0]<=TrainingSub['TimeStamp']) & (TrainingSub['TimeStamp']<=taggingInfo[j][3][0][0])] = 1

        if i==0 and house=='H1': 
            TrainSubPD = TrainingSub
        else:
            TrainSubPD = pd.concat([TrainSubPD, TrainingSub])

os.chdir(base_dir)
TrainSubPD.to_csv('TrainingSet_20130901.csv', index=False)

