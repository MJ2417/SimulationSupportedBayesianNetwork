import os, sys, sklearn

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.0.3"
os.environ["PATH"] = r"C:\Program Files\R\R-4.0.3\bin\x64" + ";" + os.environ["PATH"]
import random
import rpy2.robjects as robjects
import networkx as nx
from scipy.stats import truncexpon
from networkx.drawing.nx_agraph import to_agraph
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import STAP
from scipy.sparse import csr_matrix
import scipy
from numpy import random
from numpy import savetxt
from pandas.core.common import flatten
import rpy2.robjects as ro

# ro.conversion.py2rpy=numpy2ri
numpy2ri.activate()
import copy
import networkx as nx
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from matplotlib import rc

plt.rcParams["font.family"] = "serif"
# from Functions11 import Functions
from scipy.spatial import distance as dist

os.getcwd()
from itertools import product, combinations, combinations_with_replacement


class ConditionalTables():
    def __init__(self):
        pass

    ########## ConditionalProbTabLveles21GenerateJuly2022 function
    def ConditionalProbTabLveles21GenerateJuly2022(self, DataFrameForDisLay2out, CondiProbTableLevels21, nDiscretise,
                                                   InputVarLavelNum, RM, nEdges, AllEdges):
        for edge in range(0, nEdges, 1):
            for variableO in range(0, 2, 1):
                if variableO == 0:
                    MeasureO = 'AvailDisc11'
                    DepenVar = 'Road' + str(AllEdges[edge]) + 'Avail'
                    IndepenVar = 'MaintenanceRate' + str(variableO)  # MaintenanceRates[edge])
                else:
                    DepenVar = 'Road' + str(AllEdges[edge]) + 'Costs'
                    IndepenVar = 'MaintenanceRate' + str(variableO)  # MaintenanceRates[edge])
                    MeasureO = 'TotalCostDisc11'
                for inleval in product(range(InputVarLavelNum + 1), repeat=1):
                    # for variableIOption in range(0, 2, 1):
                    # selData=DataFrameForDisLay2out[(DataFrameForDisLay2out['Edge']==edge)&(DataFrameForDisLay2out['MaintenanceRate']==RM[edge][variableIOption] )]
                    # print(inleval[0])
                    UniqselData = DataFrameForDisLay2out[(DataFrameForDisLay2out['Edge'] == AllEdges[edge])]
                    selData = DataFrameForDisLay2out[(DataFrameForDisLay2out['Edge'] == AllEdges[edge]) & (
                            DataFrameForDisLay2out['ListMaintenanceRatesCoded'] == inleval[0])]
                    lst_dic1 = []
                    lst_dic2 = []
                    conProbList = []
                    UniqnDiscretise = UniqselData[MeasureO].nunique()
                    BotVal = len(selData)
                    # for outleval in product(range(nDiscretise), repeat=1):
                    for outleval in product(range(UniqnDiscretise), repeat=1):
                        UpVal = len(selData[(selData[MeasureO] == outleval[0])])
                        # print(inleval,outleval,UpVal,BotVal,UpVal/BotVal)
                        # conProbList.append(UpVal/BotVal)
                        if BotVal != 0:
                            lst_dic1.append(
                                {'DepenVar': DepenVar, 'DepenVarLvel': outleval[0], 'IndepenVar': IndepenVar,
                                 'IndepenVarLvel': inleval[0], 'CondProb': UpVal / BotVal})
                        lst_dic2.append({'DepenVar': DepenVar, 'DepenVarLvel': outleval[0], 'IndepenVar': IndepenVar,
                                         'IndepenVarLvel': inleval[0], 'CondProb': 1 / UniqnDiscretise})
                    if BotVal != 0:  # sum(conProbList)==0:
                        CondiProbTableLevels21 = CondiProbTableLevels21.append(lst_dic1)
                    else:
                        CondiProbTableLevels21 = CondiProbTableLevels21.append(lst_dic2)

            print(DataFrameForDisLay2out['Edge'], AllEdges[edge])  # edge,CondiProbTableLevels21)
        return CondiProbTableLevels21

    ################### ConditionalProbTabLveles10Generate function
    def ConditionalProbTabLveles10GenerateJuly2022(self, CondiProbTableLevels10, DataFrameForDisLay1Input,
                                                   DataFrameForDisLay0out, nDiscretise, OverallSample, nEdges, AllEdges,
                                                   Subnet):

        for variableO in range(0, 2, 1):
            MeasureToWrite = []
            for edge in range(0, nEdges, 1):
                if variableO == 0:
                    MeasureToWrite.append('Road' + str(AllEdges[edge]) + 'Avail')
                else:
                    MeasureToWrite.append('Road' + str(AllEdges[edge]) + 'Costs')

            if variableO == 0:
                MeasureI00 = 'AvailDisc11'
                MeasureO = 'TravelTimeDisc11'
            else:
                MeasureO = 'TotalCostDisc11'
                MeasureI00 = 'TotalCostDisc11'

            UniqnDiscretise = []
            ForLevelsselDatasimrunO = DataFrameForDisLay0out
            UniqnDiscretiseO = ForLevelsselDatasimrunO[MeasureO].nunique()
            for edge in range(0, nEdges, 1):
                ForLevelsselDatasimrun = DataFrameForDisLay1Input[(DataFrameForDisLay1Input['Edge'] == AllEdges[edge])]
                UniqnDiscretise.append(ForLevelsselDatasimrun[MeasureI00].nunique())

            for inleval in product(range(nDiscretise), repeat=nEdges):
                for outleval in product(range(UniqnDiscretiseO), repeat=1):
                    lst_dic1 = []
                    BotVal = 0
                    UpVal = 0
                    for sample_run in range(OverallSample):
                        LenselDatasimrun = []
                        for index11 in range(0, nEdges, 1):
                            selDatasimrun = DataFrameForDisLay1Input[
                                (DataFrameForDisLay1Input["sample_run"] == sample_run) & (
                                        DataFrameForDisLay1Input['Edge'] == AllEdges[index11]) & (
                                        DataFrameForDisLay1Input[MeasureI00] == inleval[index11])]
                            LenselDatasimrun.append(len(selDatasimrun))
                        selDatasimrunO = DataFrameForDisLay0out[
                            (DataFrameForDisLay0out["sample_run"] == sample_run) & (
                                    DataFrameForDisLay0out[MeasureO] == outleval[0])]
                        if len(set(LenselDatasimrun)) == 1 and list(LenselDatasimrun)[0] == 1:
                            # print(LenselDatasimrun, 'hooieeee', selDatasimrun, list(selDatasimrun)[0] == 1,
                            #      len(set(LenselDatasimrun)), LenselDatasimrun)
                            BotVal += 1
                            if len(selDatasimrunO) == 1 and list(selDatasimrunO)[0] == 1:
                                UpVal += 1

                    print(inleval, outleval, UpVal, BotVal, UpVal / (BotVal + 1))
                    values = [MeasureO, outleval[0]]
                    keys = ['DepenVar', 'DepenVarLvel', 'IndepenVar', 'IndepenVarLvel',
                            'IndepenVar1', 'IndepenVar1Lvel', 'IndepenVar2', 'IndepenVar2Lvel', 'IndepenVar3',
                            'IndepenVar3Lvel', 'CondProb']
                    for index11 in range(0, 4, 1):
                        if index11 < nEdges:
                            values.append(MeasureToWrite[index11])
                            values.append(inleval[index11])
                        else:
                            values.append(str(index11))
                            values.append("-")

                    if BotVal > 0 and UpVal > 0:
                        values.append(UpVal / BotVal)
                        lst_dic1.append(dict(zip(keys, values)))
                    else:
                        values.append(1 / (UniqnDiscretiseO))
                        lst_dic1.append(dict(zip(keys, values)))

                    print(lst_dic1)
                    # if sum(conProbList)==0:
                    CondiProbTableLevels10 = CondiProbTableLevels10.append(lst_dic1)

        return CondiProbTableLevels10

    def ConditionalProbTabSystemLevelJuly2022(self, CondiProbTableSystemLevel, DataFrameInput,
                                              DataFrameout, nDiscretise, OverallSample, nSubnet):

        for variableO in range(0, 2, 1):
            MeasureToWrite = []
            for subnet in range(0, nSubnet, 1):
                if variableO == 0:
                    MeasureToWrite.append('Subnet' + str(subnet) + 'TravelTime')
                else:
                    MeasureToWrite.append('Subnet' + str(subnet) + 'Costs')

            if variableO == 0:
                Measure = 'TravelTimeDisc11'
            else:
                Measure = 'TotalCostDisc11'

            UniqnDiscretiseInput = []
            NumUniqnDiscretise = DataFrameout[Measure].nunique()
            for subnet in range(0, nSubnet, 1):
                DataFrameInputGenerated = DataFrameInput[(DataFrameInput['subnetwork'] == subnet)]
                UniqnDiscretiseInput.append(DataFrameInputGenerated[Measure].nunique())
            for outputleval in product(range(NumUniqnDiscretise), repeat=1):
                for inleval in product(range(nDiscretise), repeat=nSubnet):
                    Correct = 1
                    for index11 in range(0, nSubnet, 1):
                        if inleval[index11] >= UniqnDiscretiseInput[index11]:
                            Correct = 0
                    if Correct == 1:
                        lst_dic1 = []
                        lst_dic2 = []
                        conProbList = []
                        BotVal = 0
                        UpVal = 0
                        for sample_run in range(OverallSample):
                            LenselDatasimrun = []
                            for index11 in range(0, nEdges, 1):
                                selDatasimrun = DataFrameInput[
                                    (DataFrameInput["sample_run"] == sample_run) & (
                                            DataFrameInput['subnetwork'] == subnet) & (
                                            DataFrameInput[Measure] == inleval[index11])]
                                LenselDatasimrun.append(len(selDatasimrun))
                            selDatasimrunO = DataFrameForDisLay0out[
                                (DataFrameForDisLay0out["sample_run"] == sample_run) & (
                                        DataFrameForDisLay0out[MeasureO] == outleval[0])]
                            if len(set(LenselDatasimrun)) == 1:
                                BotVal += 1
                                if len(selDatasimrunO) == 1:
                                    UpVal += 1

                        # print(inleval,outleval,UpVal,BotVal,UpVal/(BotVal+1))
                        keys = range(7)
                        values = [MeasureO, outleval[0]]
                        keys = ['DepenVar', 'DepenVarLvel', 'IndepenVar', 'IndepenVarLvel',
                                'IndepenVar1', 'IndepenVar1Lvel', 'IndepenVar2', 'IndepenVar2Lvel', 'IndepenVar3',
                                'IndepenVar3Lvel', 'CondProb']
                        for index11 in range(0, 4, 1):
                            if index11 < nEdges:
                                values.append(MeasureToWrite[index11])
                                values.append(inleval[index11])
                            else:
                                values.append(str(index11))
                                values.append("-")

                        if BotVal != 0:
                            conProbList.append(UpVal / BotVal)
                            values.append(UpVal / BotVal)
                            lst_dic1.append(dict(zip(keys, values)))
                        else:
                            conProbList.append(0)
                            values.append(1 / (UniqnDiscretiseO))
                            lst_dic1.append(dict(zip(keys, values)))

                        # if sum(conProbList)==0:
                        CondiProbTableLevels10 = CondiProbTableLevels10.append(lst_dic1)

        return CondiProbTableLevels10
