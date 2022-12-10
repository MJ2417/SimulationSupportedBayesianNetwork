import os, sys, sklearn

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.0.3"
os.environ["PATH"] = r"C:\Program Files\R\R-4.0.3\bin\x64" + ";" + os.environ["PATH"]
import random
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import STAP

# ro.conversion.py2rpy=numpy2ri
numpy2ri.activate()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc

plt.rcParams["font.family"] = "serif"
# from Functions11 import Functions


os.getcwd()


class Discretization():
    def __init__(self):
        pass

    #####Discretization function
    def discretize_data_frame(self, DataFrameForDis, Measure, nDiscretise):
        selData00 = DataFrameForDis[Measure]
        selData11 = selData00.to_numpy()  # np.array(selData00)#.flatten() #selData=np.reshape(selData,-1)
        selData = np.asmatrix(
            selData11)  # np.concatenate(selData11).flatten()#np.transpose(selData11).reshape(len(selData11))
        [LowInt, HighInt] = self.CalHighDensityReg(selData, 2)
        bins = [0]
        labels = [0]
        lowIndex = 0
        if LowInt <= 0:
            LowIntAlter = 0
            nDiscretiseAlter = nDiscretise + 1
            lowIndex += 1
        else:
            nDiscretiseAlter = nDiscretise
            LowIntAlter = LowInt
        for www in range(lowIndex, nDiscretiseAlter, 1):
            bins.append(LowIntAlter + www * ((HighInt - LowIntAlter) / (nDiscretiseAlter)))
            labels.append(www + 1)
        if Measure != 'Avail' and Measure != 'TravelTime':
            bins.append(max(np.max(selData), HighInt + 200))
        if Measure == 'TravelTime':
            bins.append(max(np.max(selData), HighInt + 2))
        elif Measure == 'Avail':
            bins.append(max(np.max(selData), 1.0))
        Measure11 = Measure + 'Disc'
        Measure12 = Measure + 'Disc11'
        bins11 = pd.IntervalIndex.from_arrays(bins[:len(bins) - 1], bins[1:])
        # selData12=DataFrameForDis[Measure]
        DataFrameForDis[Measure11] = pd.cut(DataFrameForDis[Measure], bins11).astype(str).str.strip(
            '()[]')  # ,labels=["1","2","3","4","5","6","7"])
        UniqBins = set(DataFrameForDis[Measure11].unique())
        key = range(len(UniqBins))
        Disc11List = dict(zip(sorted(UniqBins), key))
        print('yesss', Disc11List)
        DataFrameForDis[Measure12] = DataFrameForDis[Measure11].map(Disc11List)

        return DataFrameForDis, Disc11List, bins11

    def discretize_data_frame_algorithm2_and_3(self, DataFrameForDis, Measure, nDiscretise):
        selData00 = DataFrameForDis[Measure]
        selData11 = selData00.to_numpy()  # np.array(selData00)#.flatten() #selData=np.reshape(selData,-1)
        selData = np.asmatrix(
            selData11)  # np.concatenate(selData11).flatten()#np.transpose(selData11).reshape(len(selData11))
        [LowInt, HighInt] = self.CalHighDensityReg(selData, 2)
        bins = [0]
        labels = [0]
        lowIndex = 0
        if LowInt <= 0 or Measure != 'TotalCost':
            LowIntAlter = 0
            nDiscretiseAlter = nDiscretise + 1
            lowIndex += 1
        else:
            LowIntAlter = 0
            nDiscretiseAlter = nDiscretise + 1
            lowIndex += 1
            # nDiscretiseAlter = nDiscretise
            # LowIntAlter = LowInt
        for www in range(lowIndex, nDiscretiseAlter, 1):
            bins.append(LowIntAlter + www * ((HighInt - LowIntAlter) / (nDiscretiseAlter)))
            labels.append(www + 1)
        if Measure != 'TotalCost' and Measure != 'TravelTime':
            bins.append(max(np.max(selData), HighInt + 200))
        if Measure == 'TravelTime':
            bins.append(max(np.max(selData), HighInt + 2))
        elif Measure == 'TotalCost':
            bins.append(max(np.max(selData), HighInt + 200))
        Measure11 = Measure + 'Disc'
        Measure12 = Measure + 'Disc11'
        bins11 = pd.IntervalIndex.from_arrays(bins[:len(bins) - 1], bins[1:])
        DataFrameForDis[Measure11] = pd.cut(DataFrameForDis[Measure], bins11).astype(str).str.strip(
            '()[]')  # ,labels=["1","2","3","4","5","6","7"])
        UniqBins = set(DataFrameForDis[Measure11].unique())
        # TODO check things ...
        key = range(len(UniqBins))
        Disc11List = dict(zip(sorted(list(UniqBins)), key))
        # print('nooooomooo', sorted(list(UniqBins)))
        # print('yesss::::::', Disc11List)
        DataFrameForDis[Measure12] = DataFrameForDis[Measure11].map(Disc11List)

        return DataFrameForDis, Disc11List, bins11

    #####Discretization function
    def discretize_data_frame_with_bins(self, DataFrameForDis, Measure, bins11):
        Measure11 = Measure + 'Disc'
        Measure12 = Measure + 'Disc11'
        # bins11 = pd.IntervalIndex.from_arrays(bins[:len(bins) - 1], bins[1:])
        # selData12=DataFrameForDis[Measure]
        # print(DataFrameForDis)
        # print(bins11)
        DataFrameForDis[Measure11] = pd.cut(DataFrameForDis[Measure], bins11).astype(str).str.strip(
            '()[]')  # ,labels=["1","2","3","4","5","6","7"])
        UniqBins = set(DataFrameForDis[Measure11].unique())
        # print('yes', UniqBins)
        # print('uu', DataFrameForDis[Measure11])
        key = range(len(UniqBins))
        Disc11List = dict(zip(sorted(list(UniqBins)), key))
        # print('hoouu', Disc11List)
        DataFrameForDis[Measure12] = DataFrameForDis[Measure11].map(Disc11List)
        # apply(lambda x: Disc11List.get(x,x))
        # DataFrameForDis[Measure11].map(Disc11List)

        return DataFrameForDis, Disc11List

    #####DensisityDiscretisization Function
    ###########################################
    def CalHighDensityReg(self, Vector, Mod):
        nr, nc = Vector.shape
        rweightMatrixFinalMatrix10 = robjects.r.matrix(Vector, nrow=nr, ncol=nc)
        robjects.r.assign("weightMatrixFinalMatrix10", rweightMatrixFinalMatrix10)
        if Mod == 1:
            r_fct_string = """ 
            M_MFPT <- function(matrix){
                library("stats")
                library("hdrcde")
                library(stringi)
                Intervals<-hdr(matrix,prob=0.7)
                UpperInt<-ifelse(Intervals$hdr[2]>1,1,Intervals$hdr[2])
                LowerInt<-ifelse(Intervals$hdr[1]<0,0.00001,Intervals$hdr[1]) 
                return(c(LowerInt,UpperInt))
            }
            """
        if Mod == 2:
            r_fct_string = """ 
            M_MFPT <- function(matrix){
                library("stats")
                library("hdrcde")
                library(stringi)
                Intervals<-hdr(matrix,prob=0.7)
                UpperInt<-Intervals$hdr[2]
                LowerInt<-Intervals$hdr[1]
                return(c(LowerInt,UpperInt))
            }
            """

        r_pkg = STAP(r_fct_string, "r_pkg")
        # CONVERT PYTHON NUMPY MATRICES TO R OBJECTS
        # rmatrix,rstate00 = map(numpy2ri, matrix,state00)
        # PASS R OBJECTS INTO FUNCTION (WILL NEED TO EXTRACT DFs FROM RESULT)
        p_res = r_pkg.M_MFPT(rweightMatrixFinalMatrix10)
        return p_res
