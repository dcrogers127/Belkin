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

os.chdir(base_dir)
SampleSub = pd.read_csv("SampleSubmission.csv")
SampleSubEvt = SampleSub[['House', 'TimeStamp']].drop_duplicates()
# SampleSub['TimeStamp_DT'] = pd.DatetimeIndex(SampleSub['TimeStamp'])

L1_TimeTicks = {}

for house in houses:
    os.chdir(base_dir+"/"+house)
    for i in range(len(test_files[house])):
        test_data = io.loadmat("Testing_"+test_files[house][i]+".mat")
        buf = test_data['Buffer']
        house_day = house+"_"+test_files[house][i]
        L1_TimeTicks.update({house_day :DataFrame(buf['TimeTicks1'][0][0], columns=['TimeStamp'])})
        max_L1TT = max(L1_TimeTicks[house_day]['TimeStamp'])
        min_L1TT = min(L1_TimeTicks[house_day]['TimeStamp'])

        if ((SampleSubEvt['House']==house) & (min_L1TT<=SampleSubEvt['TimeStamp']) & (SampleSubEvt['TimeStamp']<=max_L1TT)).sum()>0:
            TestTT = DataFrame(SampleSubEvt['TimeStamp'][(SampleSubEvt['House']==house) & (min_L1TT<=SampleSubEvt['TimeStamp']) & (SampleSubEvt['TimeStamp']<=max_L1TT)], columns=['TimeStamp'])
            TestTT.index = range(0,len(TestTT.index))
            TimeInt = TestTT['TimeStamp'][range(1,(len(TestTT)))] - Series(TestTT['TimeStamp'][range(0,len(TestTT)-1)], index=TestTT['TimeStamp'][range(1,(len(TestTT)))].index)
            ProcFreq = DataFrame(TimeInt.value_counts())
            ProcFreq['House_Day'] = house_day
            ProcFreq['Min_Time'] = (min(TestTT['TimeStamp'])-min_L1TT)/(max_L1TT-min_L1TT)
            ProcFreq['Max_Time'] = (max(TestTT['TimeStamp'])-min_L1TT)/(max_L1TT-min_L1TT)
            print(ProcFreq)
        else:
            print(house_day+" has no tick marks")

##  
##        0            House_Day  Min_Time  Max_Time
##  60  441  H1_07_09_1341817201  0.444443  0.750701
##  60  441  H1_07_11_1341990001  0.415276  0.721534
##  60  440  H1_07_12_1342076401  0.439582  0.745145
##  60  436  H1_07_16_1342422001  0.444443  0.747229
##  60  437  H2_07_17_1342508401  0.440277  0.743757
##  60  482  H2_07_18_1342594801  0.415970  0.750700
##  60  477  H2_07_19_1342681201  0.420136  0.751395
##  60  480  H2_07_20_1342767601  0.418748  0.752089
##  60  254  H3_01_21_1358755201  0.443055  0.619448
##  60  283  H3_08_02_1343890801  0.380552  0.577085
##  60  369  H3_08_09_1344495601  0.373608  0.629864
##  60  361  H3_08_22_1345618801  0.419443  0.670143
##  60  308  H4_09_12_1347433201  0.401387  0.615281
##  60  334  H4_09_13_1347519601  0.394442  0.626392
##  60  356  H4_09_18_1347951601  0.404858  0.652087
##  H4_09_19_1348038001 has no tick marks
##  
