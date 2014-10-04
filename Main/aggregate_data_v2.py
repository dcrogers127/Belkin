#############################################
# Program: aggregate_data_v2.py
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
from scipy import linspace, io
from math import *
from cmath import phase
import sys

top_lim = 30
bot_lim = 30

base_dir = "/home/bulby/Documents/Kaggle/Belkin/Data"
os.chdir(base_dir)
TrainSub = pd.read_csv('TrainingSet.csv')

# house_day = 'H1_04_13_1334300401'
house_day = str(sys.argv[1])
house = house_day[:2]
day = house_day[3:]
print "Starting house %s and day %s" % (house, day)
os.chdir(base_dir+"/"+house)

TrainSubPart = TrainSub[(TrainSub['House']==house) & (TrainSub['Day']==day)]
Eval_TT = DataFrame(np.unique(TrainSubPart['TimeStamp']), columns=['Eval_TimeStamp'])
Eval_TT['Eval_Id'] = range(1, Eval_TT.shape[0]+1)
Eval_TT.index = range(Eval_TT.shape[0])
Eval_TT['House'] = house
Eval_TT['Day'] = day

train_data = io.loadmat('Tagged_Training_'+day+'.mat')
buf = train_data['Buffer']
L1_TimeTicks = DataFrame(buf['TimeTicks1'][0][0], columns=['TimeStamp'])
L2_TimeTicks = DataFrame(buf['TimeTicks2'][0][0], columns=['TimeStamp'])
HF_TimeTicks = DataFrame(buf['TimeTicksHF'][0][0], columns=['TimeStamp'])

L1_TimeTicks['Eval_Id'] = 0
L2_TimeTicks['Eval_Id'] = 0
HF_TimeTicks['Eval_Id'] = 0

for ETTob in range(Eval_TT.shape[0]):
    L1_TimeTicks['Eval_Id'][(Eval_TT['Eval_TimeStamp'][ETTob]-bot_lim<L1_TimeTicks['TimeStamp']) & (L1_TimeTicks['TimeStamp']<=Eval_TT['Eval_TimeStamp'][ETTob]+top_lim)] = Eval_TT['Eval_Id'][ETTob]
    L2_TimeTicks['Eval_Id'][(Eval_TT['Eval_TimeStamp'][ETTob]-bot_lim<L2_TimeTicks['TimeStamp']) & (L2_TimeTicks['TimeStamp']<=Eval_TT['Eval_TimeStamp'][ETTob]+top_lim)] = Eval_TT['Eval_Id'][ETTob]
    HF_TimeTicks['Eval_Id'][(Eval_TT['Eval_TimeStamp'][ETTob]-bot_lim<HF_TimeTicks['TimeStamp']) & (HF_TimeTicks['TimeStamp']<=Eval_TT['Eval_TimeStamp'][ETTob]+top_lim)] = Eval_TT['Eval_Id'][ETTob]

HF = buf['HF'][0][0].transpose()
HF = HF[HF_TimeTicks['Eval_Id']!=0, :]
HF_TimeTicks = HF_TimeTicks[HF_TimeTicks['Eval_Id']!=0]
HF_DF = DataFrame(HF, columns=[['_'+str(HF_Col) for HF_Col in range(1,HF.shape[1]+1)]])
HF_TimeTicks.index = range(HF_TimeTicks.shape[0])
HF_DF = pd.merge(HF_TimeTicks, HF_DF, left_index=True, right_index=True)
HF_GB = HF_DF[HF_DF.columns[2:]].groupby(HF_DF['Eval_Id'])

HF_mean = HF_GB.mean()
HF_max = HF_GB.max()
HF_min = HF_GB.min()
HF_range = HF_max-HF_min
HF_size = DataFrame(HF_GB.size(), columns=['HF_size'])

HF_mean = HF_mean.rename(columns=dict(zip(HF_DF.columns[2:], 'HF_mean'+np.array(HF_DF.columns[2:]))))
HF_max = HF_max.rename(columns=dict(zip(HF_DF.columns[2:], 'HF_max'+np.array(HF_DF.columns[2:]))))
HF_min = HF_min.rename(columns=dict(zip(HF_DF.columns[2:], 'HF_min'+np.array(HF_DF.columns[2:]))))
HF_range = HF_range.rename(columns=dict(zip(HF_DF.columns[2:], 'HF_range'+np.array(HF_DF.columns[2:]))))

LF1V = buf['LF1V'][0][0]
LF1I = buf['LF1I'][0][0]
LF2V = buf['LF2V'][0][0]
LF2I = buf['LF2I'][0][0]

LF1V_DF = DataFrame(LF1V[:, 0].real, columns=['LF1V_R_V1'])
LF1I_DF = DataFrame(LF1I[:, 0].real, columns=['LF1I_R_V1'])
LF2V_DF = DataFrame(LF2V[:, 0].real, columns=['LF2V_R_V1'])
LF2I_DF = DataFrame(LF2I[:, 0].real, columns=['LF2I_R_V1'])

LF1V_DF['LF1V_I_V1'] = LF1V[:, 0].imag
LF1I_DF['LF1I_I_V1'] = LF1I[:, 0].imag
LF2V_DF['LF2V_I_V1'] = LF2V[:, 0].imag
LF2I_DF['LF2I_I_V1'] = LF2I[:, 0].imag

for LF_ind in range(1,LF1V.shape[1]):
    LF1V_DF['LF1V_R_V'+str(LF_ind+1)] = LF1V[:, LF_ind].real
    LF1I_DF['LF1I_R_V'+str(LF_ind+1)] = LF1I[:, LF_ind].real
    LF2V_DF['LF2V_R_V'+str(LF_ind+1)] = LF2V[:, LF_ind].real
    LF2I_DF['LF2I_R_V'+str(LF_ind+1)] = LF2I[:, LF_ind].real

    LF1V_DF['LF1V_I_V'+str(LF_ind+1)] = LF1V[:, LF_ind].imag
    LF1I_DF['LF1I_I_V'+str(LF_ind+1)] = LF1I[:, LF_ind].imag
    LF2V_DF['LF2V_I_V'+str(LF_ind+1)] = LF2V[:, LF_ind].imag
    LF2I_DF['LF2I_I_V'+str(LF_ind+1)] = LF2I[:, LF_ind].imag

LF1V_DF = pd.merge(L1_TimeTicks, LF1V_DF, left_index=True, right_index=True)
LF1I_DF = pd.merge(L1_TimeTicks, LF1I_DF, left_index=True, right_index=True)
LF2V_DF = pd.merge(L2_TimeTicks, LF2V_DF, left_index=True, right_index=True)
LF2I_DF = pd.merge(L2_TimeTicks, LF2I_DF, left_index=True, right_index=True)

LF1V_DF = LF1V_DF[LF1V_DF['Eval_Id']!=0]
LF1I_DF = LF1I_DF[LF1I_DF['Eval_Id']!=0]
LF2V_DF = LF2V_DF[LF2V_DF['Eval_Id']!=0]
LF2I_DF = LF2I_DF[LF2I_DF['Eval_Id']!=0]

LF1V_GB = LF1V_DF[LF1V_DF.columns[2:]].groupby(LF1V_DF['Eval_Id'])
LF1I_GB = LF1I_DF[LF1I_DF.columns[2:]].groupby(LF1I_DF['Eval_Id'])
LF2V_GB = LF2V_DF[LF2V_DF.columns[2:]].groupby(LF2V_DF['Eval_Id'])
LF2I_GB = LF2I_DF[LF2I_DF.columns[2:]].groupby(LF2I_DF['Eval_Id'])

LF1V_mean = LF1V_GB.mean()
LF1V_max = LF1V_GB.max()
LF1V_min = LF1V_GB.min()
LF1V_range = LF1V_max-LF1V_min

LF1I_mean = LF1I_GB.mean()
LF1I_max = LF1I_GB.max()
LF1I_min = LF1I_GB.min()
LF1I_range = LF1I_max-LF1I_min

LF2V_mean = LF2V_GB.mean()
LF2V_max = LF2V_GB.max()
LF2V_min = LF2V_GB.min()
LF2V_range = LF2V_max-LF2V_min

LF2I_mean = LF2I_GB.mean()
LF2I_max = LF2I_GB.max()
LF2I_min = LF2I_GB.min()
LF2I_range = LF2I_max-LF2I_min

LF1_size = DataFrame(LF1V_GB.size(), columns=['LF1_size'])
LF2_size = DataFrame(LF2V_GB.size(), columns=['LF2_size'])

LF1V_mean = LF1V_mean.rename(columns=dict(zip(LF1V_DF.columns[2:], LF1V_DF.columns[2:]+'_mean')))
LF1V_max = LF1V_max.rename(columns=dict(zip(LF1V_DF.columns[2:], LF1V_DF.columns[2:]+'_max')))
LF1V_min = LF1V_min.rename(columns=dict(zip(LF1V_DF.columns[2:], LF1V_DF.columns[2:]+'_min')))
LF1V_range = LF1V_range.rename(columns=dict(zip(LF1V_DF.columns[2:], LF1V_DF.columns[2:]+'_range')))

LF1I_mean = LF1I_mean.rename(columns=dict(zip(LF1I_DF.columns[2:], LF1I_DF.columns[2:]+'_mean')))
LF1I_max = LF1I_max.rename(columns=dict(zip(LF1I_DF.columns[2:], LF1I_DF.columns[2:]+'_max')))
LF1I_min = LF1I_min.rename(columns=dict(zip(LF1I_DF.columns[2:], LF1I_DF.columns[2:]+'_min')))
LF1I_range = LF1I_range.rename(columns=dict(zip(LF1I_DF.columns[2:], LF1I_DF.columns[2:]+'_range')))

LF2V_mean = LF2V_mean.rename(columns=dict(zip(LF2V_DF.columns[2:], LF2V_DF.columns[2:]+'_mean')))
LF2V_max = LF2V_max.rename(columns=dict(zip(LF2V_DF.columns[2:], LF2V_DF.columns[2:]+'_max')))
LF2V_min = LF2V_min.rename(columns=dict(zip(LF2V_DF.columns[2:], LF2V_DF.columns[2:]+'_min')))
LF2V_range = LF2V_range.rename(columns=dict(zip(LF2V_DF.columns[2:], LF2V_DF.columns[2:]+'_range')))

LF2I_mean = LF2I_mean.rename(columns=dict(zip(LF2I_DF.columns[2:], LF2I_DF.columns[2:]+'_mean')))
LF2I_max = LF2I_max.rename(columns=dict(zip(LF2I_DF.columns[2:], LF2I_DF.columns[2:]+'_max')))
LF2I_min = LF2I_min.rename(columns=dict(zip(LF2I_DF.columns[2:], LF2I_DF.columns[2:]+'_min')))
LF2I_range = LF2I_range.rename(columns=dict(zip(LF2I_DF.columns[2:], LF2I_DF.columns[2:]+'_range')))

Eval_TT = pd.merge(Eval_TT, LF1_size, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF2_size, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, HF_size, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF1V_mean, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF1V_max, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF1V_min, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF1V_range, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF1I_mean, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF1I_max, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF1I_min, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF1I_range, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF2V_mean, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF2V_max, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF2V_min, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF2V_range, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF2I_mean, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF2I_max, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF2I_min, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, LF2I_range, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, HF_mean, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, HF_max, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, HF_min, left_on='Eval_Id', right_index=True)
Eval_TT = pd.merge(Eval_TT, HF_range, left_on='Eval_Id', right_index=True)

for Eval_TT_col in Eval_TT.columns:
    if Eval_TT_col[3:7]=='mean':
        Eval_TT[Eval_TT_col] = np.round(Eval_TT[Eval_TT_col], 1)
    elif Eval_TT_col[10:] in ('mean','max','min','range'):
        Eval_TT[Eval_TT_col] = np.round(Eval_TT[Eval_TT_col], 4)

os.chdir(base_dir+'/Temp')
Eval_TT.to_csv('Agg_v2_'+house_day+'.csv', index=False)

