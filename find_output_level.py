# coding: utf-8
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from scipy import linspace, io
from pylab import *
from cmath import phase
from math import *
import os as os

base_dir = "/home/bulby/Documents/Kaggle/Belkin/Data"
test_files = {
    "H1": ["07_09_1341817201", "07_11_1341990001", "07_12_1342076401", "07_16_1342422001"],
    "H2": ["07_17_1342508401", "07_18_1342594801", "07_19_1342681201", "07_20_1342767601"],
    "H3": ["01_21_1358755201", "08_02_1343890801", "08_09_1344495601", "08_22_1345618801"],
    "H4": ["09_12_1347433201", "09_13_1347519601", "09_18_1347951601", "09_19_1348038001"]}
houses = ["H1", "H2", "H3", "H4"]

for house in houses:
    os.chdir(base_dir+"/"+house)
    for i in range(len(test_files[house])):
        test_data = io.loadmat("Testing_"+test_files[house][i]+".mat")
        buf = test_data['Buffer']
        if house=="H1" and i==0:
            L1_TimeTicks = DataFrame(buf['TimeTicks1'][0][0], columns=['TimeStamp'])
            L2_TimeTicks = DataFrame(buf['TimeTicks2'][0][0], columns=['TimeStamp'])
            HF_TimeTicks = DataFrame(buf['TimeTicksHF'][0][0], columns=['TimeStamp'])

            L1_TimeTicks['House'] = house
            L2_TimeTicks['House'] = house
            HF_TimeTicks['House'] = house
            L1_TimeTicks['Day'] = test_files[house][i]
            L2_TimeTicks['Day'] = test_files[house][i]
            HF_TimeTicks['Day'] = test_files[house][i]
        else:
            L1_TimeTicks = pd.concat([L1_TimeTicks, DataFrame(buf['TimeTicks1'][0][0], columns=['TimeStamp'])])
            L2_TimeTicks = pd.concat([L2_TimeTicks, DataFrame(buf['TimeTicks2'][0][0], columns=['TimeStamp'])])
            HF_TimeTicks = pd.concat([HF_TimeTicks, DataFrame(buf['TimeTicksHF'][0][0], columns=['TimeStamp'])])
            L1_TimeTicks['House'] = L1_TimeTicks['House'].fillna(house)
            L2_TimeTicks['House'] = L2_TimeTicks['House'].fillna(house)
            HF_TimeTicks['House'] = HF_TimeTicks['House'].fillna(house)
            L1_TimeTicks['Day'] = L1_TimeTicks['Day'].fillna(test_files[house][i])
            L2_TimeTicks['Day'] = L2_TimeTicks['Day'].fillna(test_files[house][i])
            HF_TimeTicks['Day'] = HF_TimeTicks['Day'].fillna(test_files[house][i])

test_data = ""
buf = ""

os.chdir(base_dir)
SampleSub = pd.read_csv("SampleSubmission.csv")
SampleSubEvt = SampleSub[['House', 'TimeStamp']].drop_duplicates()
SampleSub['TimeStamp_DT'] = pd.DatetimeIndex(SampleSub['TimeStamp'])

# L1_TT_Uniq = L1_TimeTicks.drop_duplicates() 
# L2_TT_Uniq = L2_TimeTicks.drop_duplicates()
# HF_TT_Uniq = HF_TimeTicks.drop_duplicates()
# print(L1_TT_Uniq.shape == L1_TimeTicks.shape)
## True
# print(L2_TT_Uniq.shape == L2_TimeTicks.shape)
## True
# print(HF_TT_Uniq.shape == HF_TimeTicks.shape)
## True

# SampleSub_L1_TimeTicks = pd.merge(SampleSubEvt, L1_TimeTicks, on=['House','TimeStamp'], how='inner')
## empty
# SampleSub_L2_TimeTicks = pd.merge(SampleSubEvt, L2_TimeTicks, on=['House','TimeStamp'], how='inner')
## empty
# SampleSub_HF_TimeTicks = pd.merge(SampleSubEvt, HF_TimeTicks, on=['House','TimeStamp'], how='inner')
## empty

house = "H1"

SampleTime = SampleSubEvt['TimeStamp'][SampleSubEvt['House']==house]
L1_Time = L1_TimeTicks['TimeStamp'][L1_TimeTicks['House']==house] 
L2_Time = L2_TimeTicks['TimeStamp'][L2_TimeTicks['House']==house]
HF_Time = HF_TimeTicks['TimeStamp'][HF_TimeTicks['House']==house]

fig = figure(1)
ax = fig.add_subplot(111)
ax.plot(SampleTime, repeat(4, len(SampleTime)) , '.')
ax.plot(L1_Time, repeat(3, len(L1_Time)) , '.')
ax.plot(L2_Time, repeat(2, len(L2_Time)) , '.')
ax.plot(HF_Time, repeat(1, len(HF_Time)) , '.')
ax.set_ylim((0,5))
show()
