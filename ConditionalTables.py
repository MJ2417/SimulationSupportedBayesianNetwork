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
    def ConditionalProbTabLveles21GenerateJuly2022(self, DataFrameForDisLay2out, CondiProbTableLevels21,
                                                   nEdges, AllEdges, OverallSample):
        for edge in range(0, nEdges, 1):
            for variableO in range(0, 2, 1):
                if variableO == 0:
                    MeasureO = 'AvailDisc11'
                    MeasureO0 = 'Road' + str(AllEdges[edge]) + 'Avail'
                    IndepenVar = 'MaintenanceRate' + str(variableO)  # MaintenanceRates[edge])
                else:
                    MeasureO0 = 'Road' + str(AllEdges[edge]) + 'Costs'
                    IndepenVar = 'MaintenanceRate' + str(variableO)  # MaintenanceRates[edge])
                    MeasureO = 'TotalCostDisc11'
                UniqselData = DataFrameForDisLay2out[(DataFrameForDisLay2out['Edge'] == AllEdges[edge])]
                UniqnDiscretiseInput = []
                for column_label in ['bridge1Rate', 'bridge2Rate', 'bridge3Rate']:
                    unique_values = UniqselData[column_label].unique()
                    UniqnDiscretiseInput.append(list(unique_values))
                UniqnDiscretise = UniqselData[MeasureO].nunique()
                LsMaintenanceRatesUniq11 = list(UniqselData['NumBrdg'].explode().unique())
                LsMaintenanceRatesUniq = list(UniqselData['ListMaintenanceRates'].explode().unique())
                NumBridges = int(LsMaintenanceRatesUniq11[0])
                UniqnDiscretiseInput = list([item for sublist in UniqnDiscretiseInput for item in sublist])
                UniqnDiscretiseInput = [elem for elem in UniqnDiscretiseInput if elem > -1]
                # print(LsMaintenanceRatesUniq11, list(UniqnDiscretiseInput))
                # print('hhhhhh', UniqnDiscretiseInput)

                ####here
                # print('*******************************************************************')
                # print(UniqnDiscretiseInput, list(set(UniqnDiscretiseInput)), NumBridges)

                # inleval_num = 0
                #outleval_num = 0
                for inleval in product(list(set(UniqnDiscretiseInput)), repeat=NumBridges):
                    # inleval_num += 1

                    for outleval in product(range(UniqnDiscretise), repeat=1):
                        # outleval_num += 1
                        lst_dic1 = []
                        BotVal = 0
                        UpVal = 0
                        for sample_run in range(OverallSample):
                            if NumBridges == 1:
                                selData = UniqselData[(UniqselData["sim_run"] == sample_run) &
                                                      (UniqselData['Edge'] == AllEdges[edge]) &
                                                      (UniqselData['bridge1Rate'] == inleval[0])]
                            elif NumBridges == 2:
                                selData = UniqselData[(UniqselData["sim_run"] == sample_run) &
                                                      (UniqselData['Edge'] == AllEdges[edge]) &
                                                      (UniqselData['bridge1Rate'] == inleval[0]) &
                                                      (UniqselData['bridge2Rate'] == inleval[1])]
                            elif NumBridges == 3:
                                selData = UniqselData[(UniqselData["sim_run"] == sample_run) &
                                                      (UniqselData['Edge'] == AllEdges[edge]) &
                                                      (UniqselData['bridge1Rate'] == inleval[0]) &
                                                      (UniqselData['bridge2Rate'] == inleval[1]) &
                                                      (UniqselData['bridge3Rate'] == inleval[2])]

                            selDatasimrunO = selData[(selData[MeasureO] == outleval[0])]
                            if len(set(selData)) == 1:
                                BotVal += 1
                                if len(selDatasimrunO.index) > 0:
                                    UpVal += 1

                        values = [MeasureO0, outleval[0]]
                        keys = ['DepenVar', 'DepenVarLvel', 'IndepenVar', 'IndepenVarLvel',
                                'IndepenVar1', 'IndepenVar1Lvel', 'IndepenVar2', 'IndepenVar2Lvel', 'IndepenVar3',
                                'IndepenVar3Lvel', 'CondProb']
                        MeasureToWrite = []
                        for brdgeId in range(0, 4, 1):
                            if variableO == 0:
                                MeasureToWrite.append('Bridge' + str(brdgeId) + 'Road' + str(AllEdges[edge]))
                            else:
                                MeasureToWrite.append('Bridge' + str(brdgeId) + 'Road' + str(AllEdges[edge]))

                        for index11 in range(0, 4, 1):
                            if index11 < NumBridges:
                                values.append(MeasureToWrite[index11])
                                values.append(inleval[index11])
                            else:
                                values.append(str(index11))
                                values.append("-")

                        if BotVal > 0 and UpVal > 0:
                            values.append(UpVal / BotVal)
                            lst_dic1.append(dict(zip(keys, values)))
                        elif BotVal > 0:
                            values.append(UpVal / BotVal)
                            lst_dic1.append(dict(zip(keys, values)))
                        else:
                            values.append(1 / UniqnDiscretise)
                            lst_dic1.append(dict(zip(keys, values)))

                        CondiProbTableLevels21 = pd.concat(
                            [CondiProbTableLevels21, pd.DataFrame.from_dict(lst_dic1, orient='columns')])

                ####here
                # print(AllEdges[edge], inleval_num, outleval_num, edge, 'edge', NumBridges, 'NumBridges', inleval, len(UniqnDiscretiseInput), 'inleval', outleval, (UniqnDiscretise), 'outleval')

            # print(DataFrameForDisLay2out['Edge'], AllEdges[edge])  # edge,CondiProbTableLevels21)
        return CondiProbTableLevels21

    ### BackUp
    def ConditionalProbTabLveles21GenerateJuly2022BackUp(self, DataFrameForDisLay2out, CondiProbTableLevels21,
                                                         nEdges, AllEdges):
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
                UniqselData = DataFrameForDisLay2out[(DataFrameForDisLay2out['Edge'] == AllEdges[edge])]
                UniqnDiscretiseInput = UniqselData['ListMaintenanceRatesCoded'].nunique()
                UniqnDiscretise = UniqselData[MeasureO].nunique()

                for inleval in product(range(UniqnDiscretiseInput), repeat=1):
                    lst_dic1 = []
                    for outleval in product(range(UniqnDiscretise), repeat=1):
                        selData = DataFrameForDisLay2out[(DataFrameForDisLay2out['Edge'] == AllEdges[edge]) & (
                                DataFrameForDisLay2out['ListMaintenanceRatesCoded'] == inleval[0])]

                        BotVal = len(selData.index)
                        values = []
                        UpVal = len(selData[(selData[MeasureO] == outleval[0])])
                        if BotVal > 0 and UpVal > 0:
                            # print(inleval, outleval, UpVal, BotVal, UpVal / BotVal)
                            values.append(UpVal / BotVal)
                        elif BotVal > 0:
                            values.append(UpVal / BotVal)
                        else:
                            values.append(1 / UniqnDiscretise)

                        lst_dic1.append(
                            {'DepenVar': DepenVar, 'DepenVarLvel': outleval[0], 'IndepenVar': IndepenVar,
                             'IndepenVarLvel': inleval[0], 'CondProb': values[0]})
                    CondiProbTableLevels21 = pd.concat(
                        [CondiProbTableLevels21, pd.DataFrame.from_dict(lst_dic1, orient='columns')])

            # print(DataFrameForDisLay2out['Edge'], AllEdges[edge])  # edge,CondiProbTableLevels21)
        return CondiProbTableLevels21

    ################### ConditionalProbTabLveles10Generate function
    def ConditionalProbTabLveles10GenerateJuly2022(self, CondiProbTableLevels10, DataFrameForDisLay1Input,
                                                   DataFrameForDisLay0out, DataFrameAsli3, nDiscretise, OverallSample, nEdges, AllEdges,
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
                MeasureO00 = 'Subnet' + str(Subnet - 1) + 'TravelTime'
            else:
                MeasureO = 'TotalCostDisc11'
                MeasureI00 = 'TotalCostDisc11'
                MeasureO00 = 'Subnet' + str(Subnet - 1) + 'Costs'

            UniqnDiscretise = []
            ForLevelsselDatasimrunO = DataFrameForDisLay0out
            UniqnDiscretiseO = ForLevelsselDatasimrunO[MeasureO].nunique()
            for edge in range(0, nEdges, 1):
                ForLevelsselDatasimrun = DataFrameForDisLay1Input[(DataFrameForDisLay1Input['Edge'] == AllEdges[edge])]
                UniqnDiscretise.append(ForLevelsselDatasimrun[MeasureI00].nunique())

            #for inleval in product(list(set(nDiscretiseAsli)), repeat=nEdges):
            for inleval in product(range(nDiscretise), repeat=nEdges):
                eligible = 1
                for edge in range(0, nEdges, 1):
                    if variableO == 0:
                        MeasureO001 = 'Road' + str(AllEdges[edge]) + 'Avail'
                    else:
                        MeasureO001 = 'Road' + str(AllEdges[edge]) + 'Costs'

                    ForLevelsselDatasimrunAsli = DataFrameAsli3[(DataFrameAsli3['DepenVar'] == MeasureO001)]
                    nDiscretiseAsli = ForLevelsselDatasimrunAsli['DepenVarLvel'].unique()
                    nDiscretiseAsli11 = list(set(nDiscretiseAsli))
                    if inleval[edge] > max(nDiscretiseAsli11):
                        eligible = 0

                if eligible:
                    # print('hi', inleval)
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
                            # print(AllEdges[index11], nDiscretiseAsli, set(LenselDatasimrun), list(LenselDatasimrun)[0],)
                            if len(set(LenselDatasimrun)) == 1 and list(LenselDatasimrun)[0] == 1:
                                BotVal += 1
                                if len(selDatasimrunO.index) > 0:
                                    UpVal += 1

                        values = [MeasureO00, outleval[0]]
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
                            # print(inleval, outleval, UpVal, BotVal, UpVal / BotVal)
                            values.append(UpVal / BotVal)
                            lst_dic1.append(dict(zip(keys, values)))
                        elif BotVal > 0:
                            values.append(UpVal / BotVal)
                            lst_dic1.append(dict(zip(keys, values)))
                        else:
                            values.append(1 / (UniqnDiscretiseO))
                            lst_dic1.append(dict(zip(keys, values)))

                        # print(lst_dic1)
                        # CondiProbTableLevels10 = CondiProbTableLevels10.append(lst_dic1)
                        CondiProbTableLevels10 = pd.concat(
                            [CondiProbTableLevels10, pd.DataFrame.from_dict(lst_dic1, orient='columns')])

        return CondiProbTableLevels10

    def ConditionalProbTabSystemLevelJuly2022(self, CondiProbTableSystemLevel, DataFrameInput,
                                              DataFrameout, DataFrameAsli3, nDiscretise, OverallSample, nSubnet):

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
            NumUniqnDiscretiseOut = DataFrameout[Measure].nunique()
            for subnet in range(0, nSubnet, 1):
                DataFrameInputGenerated = DataFrameInput[(DataFrameInput['subnetwork'] == subnet + 1)]
                UniqnDiscretiseInput.append(DataFrameInputGenerated[Measure].nunique())

            for inleval in product(range(nDiscretise), repeat=nSubnet):
                eligible = 1
                for subnet in range(0, nSubnet, 1):
                    if variableO == 0:
                        MeasureO001 = 'Subnet' + str(subnet) + 'TravelTime'
                    else:
                        MeasureO001 = 'Subnet' + str(subnet) + 'Costs'

                    ForLevelsselDatasimrunAsli = DataFrameAsli3[(DataFrameAsli3['DepenVar'] == MeasureO001)]
                    nDiscretiseAsli = ForLevelsselDatasimrunAsli['DepenVarLvel'].unique()
                    nDiscretiseAsli11 = list(set(nDiscretiseAsli))
                    # print(MeasureO001, nDiscretiseAsli11, subnet)
                    if inleval[subnet] > max(nDiscretiseAsli11):
                        eligible = 0

                if eligible:
                    for outputleval in product(range(NumUniqnDiscretiseOut), repeat=1):
                        lst_dic1 = []
                        BotVal = 0
                        UpVal = 0
                        for sample_run in range(OverallSample):
                            LenselDatasimrun = []
                            for index11 in range(0, nSubnet, 1):
                                selDatasimrun = DataFrameInput[
                                    (DataFrameInput["sample_run"] == sample_run) & (
                                            DataFrameInput['subnetwork'] == index11 + 1) & (
                                            DataFrameInput[Measure] == inleval[index11])]
                                LenselDatasimrun.append(len(selDatasimrun))
                            selDatasimrunOut = DataFrameout[
                                (DataFrameout["sample_run"] == sample_run) & (
                                        DataFrameout[Measure] == outputleval[0])]
                            if len(set(LenselDatasimrun)) == 1 and list(LenselDatasimrun)[0] == 1:
                                BotVal += 1
                                if len(selDatasimrunOut.index) > 0:
                                    UpVal += 1

                        values = [Measure, outputleval[0]]
                        keys = ['DepenVar', 'DepenVarLvel', 'IndepenVar', 'IndepenVarLvel',
                                'IndepenVar1', 'IndepenVar1Lvel', 'IndepenVar2', 'IndepenVar2Lvel', 'IndepenVar3',
                                'IndepenVar3Lvel',  'CondProb']
                        for index11 in range(0, 4, 1):
                            if index11 < nSubnet:
                                values.append(MeasureToWrite[index11])
                                values.append(inleval[index11])
                            else:
                                values.append(str(index11))
                                values.append("-")

                        if BotVal > 0 and UpVal > 0:
                            # print(inleval, outputleval, UpVal, BotVal, UpVal / BotVal)
                            values.append(UpVal / BotVal)
                            lst_dic1.append(dict(zip(keys, values)))
                        elif BotVal > 0:
                            values.append(UpVal / BotVal)
                            lst_dic1.append(dict(zip(keys, values)))
                        else:
                            values.append(1 / NumUniqnDiscretiseOut)
                            lst_dic1.append(dict(zip(keys, values)))

                        # CondiProbTableSystemLevel = CondiProbTableSystemLevel.append(lst_dic1)
                        CondiProbTableSystemLevel = pd.concat(
                            [CondiProbTableSystemLevel, pd.DataFrame.from_dict(lst_dic1, orient='columns')])

        return CondiProbTableSystemLevel

    def ConditionalProbTabSystemLevelJuly2022BackUp(self, CondiProbTableSystemLevel, DataFrameInput,
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
            NumUniqnDiscretiseOut = DataFrameout[Measure].nunique()
            for subnet in range(0, nSubnet, 1):
                DataFrameInputGenerated = DataFrameInput[(DataFrameInput['subnetwork'] == subnet + 1)]
                UniqnDiscretiseInput.append(DataFrameInputGenerated[Measure].nunique())

            for inleval in product(range(nDiscretise), repeat=nSubnet):
                for outputleval in product(range(NumUniqnDiscretiseOut), repeat=1):
                    lst_dic1 = []
                    BotVal = 0
                    UpVal = 0
                    for sample_run in range(OverallSample):
                        LenselDatasimrun = []
                        for index11 in range(0, nSubnet, 1):
                            selDatasimrun = DataFrameInput[
                                (DataFrameInput["sample_run"] == sample_run) & (
                                        DataFrameInput['subnetwork'] == index11 + 1) & (
                                        DataFrameInput[Measure] == inleval[index11])]
                            LenselDatasimrun.append(len(selDatasimrun))
                        selDatasimrunOut = DataFrameout[
                            (DataFrameout["sample_run"] == sample_run) & (
                                    DataFrameout[Measure] == outputleval[0])]
                        if len(set(LenselDatasimrun)) == 1 and list(LenselDatasimrun)[0] == 1:
                            BotVal += 1
                            if len(selDatasimrunOut.index) > 0:
                                UpVal += 1

                    values = [Measure, outputleval[0]]
                    keys = ['DepenVar', 'DepenVarLvel', 'IndepenVar', 'IndepenVarLvel',
                            'IndepenVar1', 'IndepenVar1Lvel', 'IndepenVar2', 'IndepenVar2Lvel', 'IndepenVar3',
                            'IndepenVar3Lvel', 'IndepenVar4', 'IndepenVar4Lvel', 'CondProb']
                    for index11 in range(0, 4, 1):
                        if index11 < nSubnet:
                            values.append(MeasureToWrite[index11])
                            values.append(inleval[index11])
                        else:
                            values.append(str(index11))
                            values.append("-")

                    if BotVal > 0 and UpVal > 0:
                        # print(inleval, outputleval, UpVal, BotVal, UpVal / BotVal)
                        values.append(UpVal / BotVal)
                        lst_dic1.append(dict(zip(keys, values)))
                    elif BotVal > 0:
                        values.append(UpVal / BotVal)
                        lst_dic1.append(dict(zip(keys, values)))
                    else:
                        values.append(1 / NumUniqnDiscretiseOut)
                        lst_dic1.append(dict(zip(keys, values)))

                    # CondiProbTableSystemLevel = CondiProbTableSystemLevel.append(lst_dic1)
                    CondiProbTableSystemLevel = pd.concat(
                        [CondiProbTableSystemLevel, pd.DataFrame.from_dict(lst_dic1, orient='columns')])

        return CondiProbTableSystemLevel

