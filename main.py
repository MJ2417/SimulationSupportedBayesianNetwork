##Bridges, California State Highway System, 2015
##https://purl.stanford.edu/td948by1459
##Traffic Volume, California, 2014
##https://searchworks.stanford.edu/view/fx041qj6799
##

# os.environ["PATH"] += os.pathsep+ r"C:\Program Files\Graphviz\bin"

import os, sys

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.2.1"
os.environ["PATH"] = r"C:\Program Files\R\R-4.2.1\bin\x64" + ";" + os.environ["PATH"]
from rpy2.robjects import numpy2ri, pandas2ri

# ro.conversion.py2rpy=numpy2ri
numpy2ri.activate()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "serif"
# from Functions11 import Functions
os.getcwd()
from CaseStudyNetwork.Network import *
from Roads.Roads import *
from Subnetwork.subnet import *
from System.system import *
from Discretization import *
from ConditionalTables import *

resultsPath = 'C:\\Users\\Mohsen\\Documents\\PythonProjects\\Opac\\OutputResults\\'
RM = [[0.2, 0.5], [0.15, 0.4]]
Q10 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.25, 0.05, RM[0][0]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
Q11 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.2, 0.05, RM[1][0]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
Q20 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.55, 0.05, RM[0][1]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
Q21 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.45, 0.05, RM[1][1]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
Simulatingtime = 1200
warm_up = 500
RehabFullCost = 200
MaintenanceCost = 10
AvailCoeff = [1.0, 0.8, 0.0, 0.6]

NumSimRun = 100
n_sample = 50  # for building density
OverallSample = 60  # for generation
PlotDist = 0
MeasureList = ['Avail', 'TotalCost']
MeasureList22 = ['TravelTime', 'TotalCost']
# Subnet=4
nDiscretise = 4
network_instance = Network()
roads_instance = Roads()
subnetwork_instance = Subnetworks()
system_instance = SystemAll()
discretization_instance = Discretization()
conditional_tables_instance = ConditionalTables()
for Subnet in range(1, 5, 1):

    [EdgeAll, EdgeListWithBridges, EdgeListWithBridgesShortDis, EdgesWithTraffic, EdgesWithTrafficProb,
     EdgesWithTrafficOtherTime, BridgeRoadMat, TurningMat, TurningMatMod, TravelingTimeEdge,
     EdgeNumWithBridges] = network_instance.CharacterizeSubnet(Subnet)
    # print(TurningMatMod)
    pdAlg1BridgeLevel = pd.DataFrame(
        columns=['subnet', 'sim_run', 'BridgeID', 'ListMaintenanceRates', 'ListMaintenanceRatesCoded', 'Edge', 'Avail',
                 'TotalCost'])
    if Subnet == 1:
        pdAlg1BridgeLevelConCat = pd.DataFrame({}, columns=pdAlg1BridgeLevel.columns.values, index=None)
    pdAlg1BridgeLevel = roads_instance.EdgeSimulationJuly2022(pdAlg1BridgeLevel, BridgeRoadMat, EdgeListWithBridges,
                                                              Subnet, Q10, Q11, Q20, Q21, RM, NumSimRun,
                                                              Simulatingtime, warm_up, RehabFullCost, MaintenanceCost,
                                                              AvailCoeff)
    # print(pdAlg1BridgeLevel[['ListMaintenanceRates','ListMaintenanceRatesCoded']])
    pdAlg1BridgeLevelConCat = pd.concat([pdAlg1BridgeLevelConCat, pdAlg1BridgeLevel])
    if Subnet == 4:
        pdAlg1BridgeLevelConCat.to_csv(os.path.join(resultsPath + "pdAlg1BridgeLevelConCat.csv"))
    # print(BridgeRoadMat)
    pdAlg2InputSample = pd.DataFrame(columns=['subnet', 'sample_run', 'Edge', 'Avail', 'TotalCost'])
    if Subnet == 1:
        pdAlg2InputSampleConCat = pd.DataFrame({}, columns=pdAlg2InputSample.columns.values, index=None)
    pdAlg2InputSample = roads_instance.Sampling4InputAlg2(resultsPath, pdAlg1BridgeLevel, pdAlg2InputSample,
                                                          MeasureList, EdgeListWithBridges, EdgeNumWithBridges, Subnet,
                                                          OverallSample, PlotDist, n_sample)
    pdAlg2InputSampleConCat = pd.concat([pdAlg2InputSampleConCat, pdAlg2InputSample])
    if Subnet == 4:
        pdAlg2InputSampleConCat.to_csv(os.path.join(resultsPath + "pdAlg2InputSampleConCat.csv"))
    # print(pdAlg2InputSample)

    # Inputs of third algorithm
    # IncomMat1 = ([[0, 0.2, 0.4, 0.4], [0.15, 0.5, 0.15, 0.2], [0.2, 0.3, 0.5, 0], [0.23, 0.1, 0, .67]])
    # IncomMat2 = ([[0, 0.2, 0.4, 0.4], [0.15, 0.5, 0.15, 0.2], [0.1, 0.15, 0.75, 0], [0.23, 0.1, 0, .67]])
    # print(IncomMat1[0][3])
    pdAlg2output = pd.DataFrame(columns=['subnetwork', 'sample_run', 'TravelTime', 'TotalCost'])
    if Subnet == 1:
        pdAlg2outputConCat = pd.DataFrame({}, columns=pdAlg2output.columns.values, index=None)
    pdAlg2output = subnetwork_instance.SimulateAlg2Output(Subnet, EdgeAll, EdgeListWithBridges,
                                                          EdgeListWithBridgesShortDis, EdgesWithTraffic,
                                                          EdgesWithTrafficProb, EdgesWithTrafficOtherTime,
                                                          BridgeRoadMat,
                                                          TurningMat, TurningMatMod, TravelingTimeEdge,
                                                          EdgeNumWithBridges,
                                                          pdAlg2output, pdAlg2InputSample, OverallSample)
    pdAlg2outputConCat = pd.concat([pdAlg2outputConCat, pdAlg2output])
    if Subnet == 4:
        pdAlg2outputConCat.to_csv(os.path.join(resultsPath + "pdAlg2outputConCat.csv"))

    # IncomMat1,IncomMat2,pdAlg2output,pdAlg2InputSample,OverallSample,aaa,bbb)
    indexLoop = 0
    np.set_printoptions(threshold=sys.maxsize)
    DataFrameForDisLay0out = pdAlg2output
    for edgeBri in EdgeListWithBridges:
        DataFrameForDisLay2out = pdAlg1BridgeLevel[(pdAlg1BridgeLevel['Edge'] == edgeBri)]
        DataFrameForDisLay1Input = pdAlg2InputSample[(pdAlg2InputSample['Edge'] == edgeBri)]  # layer 1-input
        if indexLoop == 0:
            DataFrameForDisLay2outConCat = pd.DataFrame({}, columns=DataFrameForDisLay2out.columns.values,
                                                        index=None)  # copy.deepcopy(DataFrameForDisLay2out)
            DataFrameForDisLay1InputConCat = pd.DataFrame({}, columns=DataFrameForDisLay1Input.columns.values,
                                                          index=None)  # copy.deepcopy(DataFrameForDisLay1Input)
        indexLoop += 1
        for MeasureId in range(0, 2, 1):
            Measure = MeasureList[MeasureId]
            [DataFrameForDisLay2out, DiscLay2outRoad12, binsLay2out] = discretization_instance.DiscretizeDataFrame(
                DataFrameForDisLay2out, Measure, nDiscretise)
            DataFrameForDisLay1Input[Measure] = DataFrameForDisLay1Input[Measure].astype(float)
            [DataFrameForDisLay1Input, DiscLay1InputRoad12] = discretization_instance.DiscretizeDataFrameWithBins(
                DataFrameForDisLay1Input, Measure, binsLay2out)
            # print(DataFrameForDisLay2out)
            DataFrameForDisLay2outConCat = pd.concat([DataFrameForDisLay2outConCat, DataFrameForDisLay2out])
            DataFrameForDisLay1InputConCat = pd.concat([DataFrameForDisLay1InputConCat, DataFrameForDisLay1Input])

    if Subnet == 1:
        DataFrameForDisLay0outConCat = pd.DataFrame({}, columns=DataFrameForDisLay0out.columns.values, index=None)
    for MeasureId in range(0, 2, 1):
        Measure22 = MeasureList22[MeasureId]
        [DataFrameForDisLay0out, DiscLay0out, binsLay0out] = discretization_instance.DiscretizeDataFrame(
            DataFrameForDisLay0out, Measure22, nDiscretise)
        DataFrameForDisLay0outConCat = pd.concat([DataFrameForDisLay0outConCat, DataFrameForDisLay0out])
    if Subnet == 4:
        DataFrameForDisLay0outConCat.to_csv(os.path.join(resultsPath + "DataFrameForDisLay0outConCat.csv"))

    # Inputs of conditional prob functions
    InputVarLavelNum = 2
    CondiProbTableLevels21 = pd.DataFrame(
        columns=['DepenVar', 'DepenVarLvel', 'IndepenVar', 'IndepenVarLvel', 'CondProb'])
    CondiProbTableLevels10 = pd.DataFrame(columns=['DepenVar', 'DepenVarLvel', 'IndepenVar', 'IndepenVarLvel',
                                                   'IndepenVar1', 'IndepenVar1Lvel', 'IndepenVar2', 'IndepenVar2Lvel',
                                                   'IndepenVar3', 'IndepenVar3Lvel', 'CondProb'])
    if Subnet == 1:
        CondiProbTableLevels21ConCat = pd.DataFrame({}, columns=CondiProbTableLevels21.columns.values, index=None)
        CondiProbTableLevels10ConCat = pd.DataFrame({}, columns=CondiProbTableLevels10.columns.values, index=None)

    # 'DepenVar', 'DepenVarLvel','IndepenVar','IndepenVarLvel','IndepenVar1', 'IndepenVar1Lvel','CondProb'])

    CondiProbTableLevels21 = conditional_tables_instance.ConditionalProbTabLveles21GenerateJuly2022(
        DataFrameForDisLay2outConCat, CondiProbTableLevels21, nDiscretise, InputVarLavelNum, RM, EdgeNumWithBridges,
        EdgeListWithBridges)
    # here#DataFrameForDisLay2outConCat.to_csv(os.path.join(resultsPath+"DataFrameForDisLay2outConCat"+".csv"))
    CondiProbTableLevels21ConCat = pd.concat([CondiProbTableLevels21ConCat, CondiProbTableLevels21])
    # CondiProbTableLevels21=ConditionalProbTabLveles21Generate(DataFrameForDisLay2out,CondiProbTableLevels21,nDiscretise,InputVarLavelNum,RM )
    CondiProbTableLevels10 = conditional_tables_instance.ConditionalProbTabLveles10GenerateJuly2022(
        CondiProbTableLevels10, DataFrameForDisLay1InputConCat, DataFrameForDisLay0out, nDiscretise, OverallSample,
        EdgeNumWithBridges, EdgeListWithBridges)
    CondiProbTableLevels10ConCat = pd.concat([CondiProbTableLevels10ConCat, CondiProbTableLevels10])
    # ConditionalProbTabLveles21GenerateJuly2022(DataFrameForDisLay2out,CondiProbTableLevels21,nDiscretise,InputVarLavelNum,RM,nEdges,AllEdges,MaintenanceRates)
    # here#DataFrameForDisLay1Input.to_csv(os.path.join(resultsPath+"2022-04-01-DataFrameForDisLay1Input" + '.csv'))
    # here#DataFrameForDisLay0out.to_csv(os.path.join(resultsPath+"2022-04-01-DataFrameForDisLay0out" + '.csv'))
    if Subnet == 4:
        CondiProbTableLevels21ConCat.to_csv(
            os.path.join(resultsPath + "2022-04-01-CondiProbTableLevels21ConCat" + '.csv'))
        CondiProbTableLevels10ConCat.to_csv(
            os.path.join(resultsPath + "2022-04-01-CondiProbTableLevels10ConCat" + '.csv'))

# Algorithm 3-sampling
for Subnet in range(1, 5, 1):
    pdAlg3InputSample = pd.DataFrame(columns=['subnetwork', 'sample_run', 'TravelTime', 'TotalCost'])
    if Subnet == 1:
        pdAlg3InputSampleConCat = pd.DataFrame({}, columns=pdAlg3InputSample.columns.values, index=None)
    pdAlg3InputSample = system_instance.Sampling4InputAlg3(resultsPath, pdAlg2outputConCat, pdAlg3InputSample,
                                                           MeasureList22, Subnet, OverallSample, PlotDist, n_sample)
    pdAlg3InputSampleConCat = pd.concat([pdAlg3InputSampleConCat, pdAlg3InputSample])
    if Subnet == 4:
        pdAlg3InputSampleConCat.to_csv(os.path.join(resultsPath + "DataFrameForpdAlg3InputSampleConCat.csv"))

pdAlg3Output = pd.DataFrame(columns=['sample_run', 'TravelTime', 'TotalCost'])
pdAlg3Output = system_instance.SimulateCostAlg3(network_instance, pdAlg3InputSampleConCat, pdAlg3Output, OverallSample)
pdAlg3Output.to_csv(os.path.join(resultsPath + "DataFrameForpdAlg3Output.csv"))
# write summation of costs /travel time across subnetworks..
# pdAlg3output=SimulateAlg2Output(Subnet, EdgeAll,EdgeListWithBridges,EdgeListWithBridgesShortDis,EdgesWithTraffic,EdgesWithTrafficProb,EdgesWithTrafficOtherTime,BridgeRoadMat,
#                       TurningMat,TurningMatMod,TravelingTimeEdge,EdgeNumWithBridges,
#                       pdAlg2output,pdAlg2InputSample,OverallSample)


# Search for BigAssumptionMohsen this is an important about the complicated equation
# of subnetwork simulation here we examine the set AA in which those edges with higher than median costs, and
# there is an or condition edded in order to handle situations in which the number of edges/roads
# wwith bridges are less than 2-3, then that set AA could be empty or with only one element, and
# that makes calculation of pair of edges impossible!
exit()
