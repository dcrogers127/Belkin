# coding: utf-8
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from scipy import linspace, io
from pylab import *
from cmath import phase
from math import *
import os as os

# Initialize vars
base_dir = "/home/bulby/Documents/Kaggle/Belkin/Data"
training_files = [
    "12_27_1356595201",
    "04_13_1334300401",
    "10_22_1350889201",
    "10_23_1350975601",
    "10_24_1351062001",
    "10_25_1351148401"]

# change dir
house = "H1"
os.chdir(base_dir+"/"+house)
 
for i in range(len(training_files)):
    testData = io.loadmat('Tagged_Training_'+training_files[i]+'.mat')
    buf = testData['Buffer']
    if i==0:
        LF1V = buf['LF1V'][0][0]
        LF1I = buf['LF1I'][0][0]
        L1_TimeTicks = buf['TimeTicks1'][0][0]
        LF2V = buf['LF2V'][0][0]
        LF2I = buf['LF2I'][0][0]
        L2_TimeTicks = buf['TimeTicks2'][0][0]
        HF = buf['HF'][0][0]
        HF_TimeTicks = buf['TimeTicksHF'][0][0]
        taggingInfo = buf['TaggingInfo'][0][0]
    else:
        LF1V = np.concatenate([LF1V, buf['LF1V'][0][0]], axis=0)
        LF1I = np.concatenate([LF1I, buf['LF1I'][0][0]], axis=0)
        L1_TimeTicks = np.concatenate([L1_TimeTicks, buf['TimeTicks1'][0][0]], axis=0)
        LF2V = np.concatenate([LF2V, buf['LF2V'][0][0]], axis=0)
        LF2I = np.concatenate([LF2I, buf['LF2I'][0][0]], axis=0)
        L2_TimeTicks = np.concatenate([L2_TimeTicks, buf['TimeTicks2'][0][0]], axis=0)
        HF = np.concatenate([HF, buf['HF'][0][0]], axis=0)
        HF_TimeTicks = np.concatenate([HF_TimeTicks, buf['TimeTicksHF'][0][0]], axis=0)
        taggingInfo = np.concatenate([taggingInfo, buf['TaggingInfo'][0][0]], axis=0)

# look at info
buf.dtype

LF1V.shape # Nx6, fundamental and first 5 harm of 60Hz volatage of Phase-1
LF1I.shape # Nx6, fundamental and first 5 harm of 60Hz current of Phase-1
L1_TimeTicks.shape
LF2V.shape # Nx6, fundamental and first 5 harm of 60Hz volatage of Phase-2
LF2I.shape # Nx6, fundamental and first 5 harm of 60Hz current of Phase-2
L2_TimeTicks.shape
HF.shape # spectograph of high freq noise.  Calc ~1.067 secs. 4096xN N=# of FFT vectors
HF_TimeTicks.shape # Nx1 UNIX timestamps.  N=# of FFT vectors in HF
taggingInfo.shape

shapes = [LF1V.shape, LF1I.shape , L1_TimeTicks.shape, LF2V.shape, LF2I.shape, L2_TimeTicks.shape, HF.shape, HF_TimeTicks.shape, taggingInfo.shape]
shapes

# first H1
# [(347410, 6),
#  (347410, 6),
#  (347410, 1),
#  (347398, 6),
#  (347398, 6),
#  (347398, 1),
#  (4096, 54222),
#  (54222, 1),
#  (3, 4)]
# 
# All H1
# [(2942088, 6),
#  (2942088, 6),
#  (2942088, 1),
#  (2942092, 6),
#  (2942092, 6),
#  (2942092, 1),
#  (4096, 459218),
#  (459218, 1),
#  (111, 4)]
