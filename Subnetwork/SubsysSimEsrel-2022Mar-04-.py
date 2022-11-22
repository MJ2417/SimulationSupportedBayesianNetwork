import os,sys,sklearn
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.0.3"
os.environ["PATH"]   = r"C:\Program Files\R\R-4.0.3\bin\x64" + ";" + os.environ["PATH"]
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
#ro.conversion.py2rpy=numpy2ri
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
#from Functions11 import Functions
from scipy.spatial import distance as dist
os.getcwd()
from random import seed
from osgeo import ogr
import osmnx as ox
import momepy
import shapefile
import geopandas
import geojson
import fiona
from shapely.geometry import shape, LineString, Point
from shapely.geometry import Polygon
from networkx.readwrite import json_graph
from mpl_toolkits.basemap import Basemap as Basemap
from difflib import SequenceMatcher
import re
#from Markov_chain_new import MarkovChain
#from KDA import KDA
import itertools
import more_itertools
from scipy.interpolate import interp1d
from discreteMarkovChain import markovChain
import datetime
from Markov_chain_new import MarkovChain
from scipy.stats import beta
from itertools import product, combinations, combinations_with_replacement

##Bridges, California State Highway System, 2015
##https://purl.stanford.edu/td948by1459
##Traffic Volume, California, 2014
##https://searchworks.stanford.edu/view/fx041qj6799
##

#os.environ["PATH"] += os.pathsep+ r"C:\Program Files\Graphviz\bin"

###############################################################################################################################################
mode=1
seed(4)

#SimulationTime=1000
#Bridge='12345'

def TransitionMat(ModeMat,eta,vartheta):
    if ModeMat==0:
        Mat = np.array([[0.91+0.02 * (eta+vartheta), 0.09-0.02 * (eta+vartheta), 0.0, 0.0],
                      [0.0, 0.68, 0.2, 0.3],
                      [0.06, 0.01, 0.93, 0.0],
                      [0.95, 0.0, 0.0, 0.05]])
    else:
        Mat = np.array([[0.97 + 0.005 * (eta + vartheta), 0.03 - 0.005 * (eta + vartheta), 0.0, 0.0],
                        [0.0, 0.68, 0.2, 0.3],
                        [0.06, 0.01, 0.93, 0.0],
                        [0.95, 0.0, 0.0, 0.05]])

    return Mat



###########EdgeSimulation function

def EdgeSimulation(pdAlg1BridgeLevel,Q10,Q11, Q20,Q21,RM,NumSimRun):
    #print(RM[0][1])
    lst_dic1=[]
    for BridgeID in range(0,2,1):
        #Q1 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.25, 0.05, RM[BridgeID][0]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
        #Q2 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.55, 0.05, RM[BridgeID][1]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
        Edge=BridgeID
        for sim_run in range(NumSimRun):
            lst_dic1=[]
            if np.random.random()<0.6:
                MaintenanceRatePolicy=0
                MaintenanceRate=RM[BridgeID][0]
                if BridgeID==0:
                    QQ=Q10
                else:
                    QQ = Q11
            else:
                MaintenanceRatePolicy=1
                MaintenanceRate = RM[BridgeID][1]
                if BridgeID == 0:
                    QQ=Q20
                else:
                    QQ = Q21
                #[TotalAva,TotalCost,pi, time_spent] = simulate_cmc(Q2, 1100, 500)
                #sim_run,BridgeID,MaintenanceRate,MaintenanceRatePolicy,Edge
            [TotalAva,TotalCost,pi,time_spent]=simulate_cmc(QQ)
            lst_dic1.append({'sim_run': sim_run, 'BridgeID': BridgeID,
                 'MaintenanceRate': MaintenanceRate, 'Edge': Edge,
                 'Avail': TotalAva, 'TotalCost': TotalCost})
            pdAlg1BridgeLevel = pdAlg1BridgeLevel.append(lst_dic1)
            #print(TotalAva,TotalCost,time_spent,pi)
    return pdAlg1BridgeLevel



############### SimulateAlg2OutputOLD function

def SimulateAlg2OutputOLD(IncomMat1,IncomMat2,pdAlg2output,pdAlg2InputSample,OverallSample,aaa,bbb):

    #for index, row in pdAlg2InputSample.iterrows():
    for sample_run in range(OverallSample):#OverallSample
        #if row["sample_run"] == sample_run:
        lst_dic1=[]
        selData = pdAlg2InputSample.loc[(pdAlg2InputSample["sample_run"] == sample_run)]
        if np.random.random() < 0.6:
            IncomMat = IncomMat1
        else:
            IncomMat = IncomMat2
        IncomMatEdit = []
        #IncomMatEdit =  IncomMat
        for jjj in range(0,4,1):
            if jjj<=1:
                for index11, row11 in selData.iterrows():
                    for EdgeID in range(0, 2, 1):
                        if row11["Edge"] == EdgeID and jjj==EdgeID and row11["Avail"]<1.0:
                            rowToAdd = []
                            if IncomMat[EdgeID][EdgeID]==0:
                                for iii in range(0, 4, 1):
                                    if iii==EdgeID:
                                        #IncomMatEdit[EdgeID][EdgeID] = \
                                        rowToAdd.append(1.0 - row11["Avail"])
                                    elif IncomMat[EdgeID][iii]>0:
                                        #IncomMatEdit[EdgeID][iii] = \
                                        rowToAdd.append(IncomMat[EdgeID][iii]*row11["Avail"])
                            else:
                                for iii in range(0, 4, 1):
                                    if iii==EdgeID:
                                        # IncomMatEdit[EdgeID][EdgeID]=\
                                        rowToAdd.append(IncomMat[EdgeID][EdgeID] + (1.0 - IncomMat[EdgeID][EdgeID]) * (1.0 - row11["Avail"]))

                                    elif IncomMat[EdgeID][iii] > 0:
                                        #IncomMatEdit[EdgeID][iii] =
                                        rowToAdd.append(IncomMat[EdgeID][iii]*row11["Avail"])
                            IncomMatEdit.append(np.array(rowToAdd).ravel())
            else:
                IncomMatEdit.append(np.array(IncomMat[jjj]).ravel())


        #print(IncomMatEdit)
        ##Here the matrix is ready
        IncomMatEdit=np.asarray(IncomMatEdit)
        #IncomMatEditArray=np.concatenate(IncomMatEdit).ravel()#np.array(IncomMatEdit)#.reshape(4,4)
        #print(type(IncomMatEdit))
        #print((IncomMatEdit))
        MC = MarkovChain(IncomMatEdit, verbose=True)
        #print(MC.K)
        #for index11, row11 in selData.iterrows():
        #    for EdgeID in range(0, 2, 1):
        #        if row11["Edge"] == EdgeID:
        term=np.sum(selData['TotalCost'].to_list())
        #term1=term-(np.sqrt(0.1*term)/(2*beta.rvs(aaa,bbb)))
        term1 = term +(0.1 * term*2 * beta.rvs(aaa, bbb))
        #print('cost',term1)
        #print(selData)
        lst_dic1.append({'sample_run': sample_run,'TravelTime': MC.K, 'TotalCost': term1})
        pdAlg2output = pdAlg2output.append(lst_dic1)

    return pdAlg2output
    #print(pdAlg2output)


############### SimulateAlg3Output function
#write function with input
def SimulateAlg3Output(network_instance, subnetwork, EdgeAll,EdgeListWithBridges,EdgeListWithBridgesShortDis,EdgesWithTraffic,EdgesWithTrafficProb,EdgesWithTrafficOtherTime,BridgeRoadMat,
                       TurningMat,TurningMatMod,TravelingTimeEdge,EdgeNumWithBridges,
                       pdAlg2output,pdAlg2InputSample,OverallSample):

    MincostAll = 0
    for sample_run in range(OverallSample):
        selData = pdAlg3InputSample.loc[(pdAlg3InputSample["sample_run"] == sample_run)]

        EdgewithBridgeHighMedinacost=[]
        Mediancost=np.median(selData['TotalCost'].to_list())
        # TODO BigAssumptionMohsen
        for edge in EdgeAll:
            if edge in EdgeListWithBridges:
                selDataCostEdge= pdAlg2InputSample.loc[(pdAlg2InputSample["sample_run"] == sample_run)&(pdAlg2InputSample["Edge"] == edge)]
                selDataCostEdgeVal=selDataCostEdge['TotalCost'].to_list()
                if selDataCostEdgeVal>Mediancost or len(EdgeListWithBridges)<=3:
                    EdgewithBridgeHighMedinacost.append(edge)









########## ConditionalProbTabLveles21Generate function
def ConditionalProbTabLveles21Generate(DataFrameForDisLay2out,CondiProbTableLevels21,nDiscretise,InputVarLavelNum,RM ):
    for edge in range(0,2,1):
        for variableO in range(0, 2, 1):
            if variableO==0:
                MeasureO='AvailDisc11'
                if edge==0:
                    DepenVar='Road1-2Avail'
                    IndepenVar='MaintenanceRate1'
                else:
                    DepenVar = 'Road1-3Avail'
                    IndepenVar='MaintenanceRate2'
            else:
                if edge==0:
                    DepenVar='Road1-2Costs'
                    IndepenVar='MaintenanceRate1'
                else:
                    DepenVar = 'Road1-3Costs'
                    IndepenVar = 'MaintenanceRate2'
                MeasureO='TotalCostDisc11'
            for inleval in product(range(InputVarLavelNum), repeat=1):
                # for variableIOption in range(0, 2, 1):
                # selData=DataFrameForDisLay2out[(DataFrameForDisLay2out['Edge']==edge)&(DataFrameForDisLay2out['MaintenanceRate']==RM[edge][variableIOption] )]
                #print(inleval[0])
                UniqselData = DataFrameForDisLay2out[(DataFrameForDisLay2out['Edge'] == edge)]
                selData = DataFrameForDisLay2out[(DataFrameForDisLay2out['Edge'] == edge)&(DataFrameForDisLay2out['MaintenanceRate'] == RM[edge][inleval[0]])]
                lst_dic1=[]
                lst_dic2 = []
                conProbList=[]
                UniqnDiscretise=UniqselData[MeasureO].nunique()
                BotVal = len(selData)
                #for outleval in product(range(nDiscretise), repeat=1):
                for outleval in product(range(UniqnDiscretise), repeat=1):
                    UpVal=len(selData[(selData[MeasureO]==outleval[0])])
                    print(inleval,outleval,UpVal,BotVal,UpVal/BotVal)
                    conProbList.append(UpVal/BotVal)
                    lst_dic1.append({'DepenVar':DepenVar, 'DepenVarLvel':outleval[0],'IndepenVar':IndepenVar,
                         'IndepenVarLvel':inleval[0],'CondProb':UpVal/BotVal})
                    lst_dic2.append({'DepenVar': DepenVar, 'DepenVarLvel': outleval[0], 'IndepenVar': IndepenVar,
                                 'IndepenVarLvel': inleval[0], 'CondProb': 1 / UniqnDiscretise})
                if BotVal!=0:#sum(conProbList)==0:
                    CondiProbTableLevels21 = CondiProbTableLevels21.append(lst_dic1)
                else:
                    CondiProbTableLevels21 = CondiProbTableLevels21.append(lst_dic2)

    return CondiProbTableLevels21




################### ConditionalProbTabLveles10Generate function
def ConditionalProbTabLveles10Generate(CondiProbTableLevels10,DataFrameForDisLay1Input,DataFrameForDisLay0out,nDiscretise,OverallSample):
    for variableO in range(0, 2, 1):
            if variableO==0:
                MeasureI00='AvailDisc11'
                MeasureI='AvailDisc11RD12'
                MeasureII='AvailDisc11RD13'
                MeasureO='TravelTimeDisc11'
                MeasureCore='Avail'
                IndepenVar='Road1-2Avail'
                IndepenVar1 = 'Road1-3Avail'
            else:
                MeasureO='TotalCostDisc11'
                MeasureI00='TotalCostDisc11'
                MeasureI='TotalCostDisc11RD12'
                MeasureII='TotalCostDisc11RD13'
                IndepenVar='Road1-2Costs'
                IndepenVar1 = 'Road1-3Costs'
            ForLevelsselDatasimrun = DataFrameForDisLay1Input[(DataFrameForDisLay1Input['Edge']==0)]
            ForLevelsselDatasimrun1 = DataFrameForDisLay1Input[(DataFrameForDisLay1Input['Edge']==1)]
            ForLevelsselDatasimrunO = DataFrameForDisLay0out
            UniqnDiscretise=ForLevelsselDatasimrun[MeasureI00].nunique()
            UniqnDiscretise1=ForLevelsselDatasimrun1[MeasureI00].nunique()
            UniqnDiscretiseO=ForLevelsselDatasimrunO[MeasureO].nunique()
            # for inleval in product(range(nDiscretise), repeat=1):
            #     for inleval1 in product(range(nDiscretise), repeat=1):
            #         for outleval in product(range(nDiscretise), repeat=1):
            for inleval in product(range(UniqnDiscretise), repeat=1):
                for inleval1 in product(range(UniqnDiscretise1), repeat=1):
                    for outleval in product(range(UniqnDiscretiseO), repeat=1):
                        lst_dic1 = []
                        lst_dic2 = []
                        conProbList = []
                        BotVal=0
                        UpVal=0
                        for sample_run in range(OverallSample):
                            selDatasimrun = DataFrameForDisLay1Input[(DataFrameForDisLay1Input["sample_run"] == sample_run)&(DataFrameForDisLay1Input['Edge']==0)&(DataFrameForDisLay1Input[MeasureI00]==inleval[0])]
                            selDatasimrun1 = DataFrameForDisLay1Input[(DataFrameForDisLay1Input["sample_run"] == sample_run)&(DataFrameForDisLay1Input['Edge']==1)&(DataFrameForDisLay1Input[MeasureI00]==inleval1[0])]
                            selDatasimrunO = DataFrameForDisLay0out[(DataFrameForDisLay0out["sample_run"] == sample_run)&(DataFrameForDisLay0out[MeasureO]==outleval[0])]
                            if len(selDatasimrun)==1 and len(selDatasimrun1)==1:
                                BotVal+=1
                                if len(selDatasimrunO)==1:
                                    UpVal+=1

                        #print(inleval,outleval,UpVal,BotVal,UpVal/(BotVal+1))
                        if BotVal!=0:
                            conProbList.append(UpVal/BotVal)
                            lst_dic1.append({'DepenVar': MeasureO, 'DepenVarLvel': outleval[0], 'IndepenVar': MeasureI,
                                             'IndepenVarLvel': inleval[0], 'IndepenVar1': MeasureII,
                                             'IndepenVar1Lvel': inleval1[0], 'CondProb': UpVal / BotVal})
                        else:
                            conProbList.append(0)
                            lst_dic1.append({'DepenVar': MeasureO, 'DepenVarLvel': outleval[0], 'IndepenVar': MeasureI,
                                             'IndepenVarLvel': inleval[0], 'IndepenVar1': MeasureII,
                                             'IndepenVar1Lvel': inleval1[0], 'CondProb': 1 / (UniqnDiscretiseO)})
                        #if sum(conProbList)==0:
                        CondiProbTableLevels10 = CondiProbTableLevels10.append(lst_dic1)

    return CondiProbTableLevels10

#########MCSimulate
def MCSimulate(pd1,pd2,AvailCoeff,RehabpartialCost,ExpID,eta,vartheta,NumRun,SimulationTime,Bridge,Edge,Subset,Subsystem,P):
    # RehabFullCost=200
    # MaintenanceCost=10
    RehabFullCost11=200
    MaintenanceCost11=10
    RehabFullCost22=300
    MaintenanceCost22=20
    stateChangeHist = np.array([[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]])
    # Simulate from multinomial distribution
    def simulate_multinomial(vmultinomial):
        r=np.random.uniform(0.0, 1.0)
        CS=np.cumsum(vmultinomial)
        #print(CS)
        CS=np.insert(CS,0,0)
        #print(CS)
        m=(np.where(CS<r))[0]
        nextState=m[len(m)-1]
        #print(CS,m,r,nextState,(np.where(CS<r)))
        return nextState
    for Run in range(NumRun):
        lst_dic1 = []
        lst_dic2 = []
        state = np.array([[1.0, 0.0, 0.0, 0.0]])
        currentState = 0
        stateHist = state
        dfStateHist = pd.DataFrame(state)
        distr_hist = [[0, 0, 0, 0]]
        TotalAvail=0.0
        TotalCost11 = 0.0
        TotalCost12 = 0.0
        TotalCost21 = 0.0
        TotalCost22 = 0.0
        for Simulationt in range(SimulationTime):
            currentRow=np.ma.masked_values((P[currentState]), 0.0)
            nextState=simulate_multinomial(currentRow)
            TotalAvail+=AvailCoeff[nextState]
            #print(nextState)
            # Keep track of state changes
            stateChangeHist[currentState,nextState]+=1
            # current is 3 and next 2 then RehabpartialCost,current is 3 and next 1 then RehabFullCost,
            # next is 4 and current other than 4, then MaintenanceCost
            #print(currentState,nextState)
            stateCost11 = 0.0
            stateCost12 = 0.0
            stateCost21 = 0.0
            stateCost22 = 0.0

            if currentState==2 and nextState==1:
                TotalCost11+=RehabpartialCost
                TotalCost12 += RehabpartialCost
                TotalCost21 += RehabpartialCost
                TotalCost22 += RehabpartialCost
                stateCost11 = RehabpartialCost
                stateCost12 = RehabpartialCost
                stateCost21 = RehabpartialCost
                stateCost22 = RehabpartialCost
            elif currentState==2 and nextState==0:
                TotalCost11 += RehabFullCost11
                TotalCost12 += RehabFullCost11
                TotalCost21 += RehabFullCost22
                TotalCost22 += RehabFullCost22
                stateCost11 = RehabFullCost11
                stateCost12 = RehabFullCost11
                stateCost21 = RehabFullCost22
                stateCost22 = RehabFullCost22
            elif currentState!=3 and nextState==3:
                TotalCost11 += MaintenanceCost11
                TotalCost12 += MaintenanceCost22
                TotalCost21 += MaintenanceCost11
                TotalCost22 += MaintenanceCost22
                stateCost11 = MaintenanceCost11
                stateCost12 = MaintenanceCost22
                stateCost21 = MaintenanceCost11
                stateCost22 = MaintenanceCost22

                #print(TotalCost,'cost')

            lst_dic1.append({'ExpID':ExpID,'eta':eta,'vartheta':vartheta,'Bridge': Bridge,'Edge':Edge,'Subset':Subset,'Subsystem':Subsystem, 'Run': Run, 'SimulationTime': Simulationt,'State':nextState,'Avail':AvailCoeff[currentState],
                             'stateCost11':stateCost11,'stateCost12':stateCost12,'stateCost21':stateCost21,'stateCost22':stateCost22})

            # Keep track of the state vector itself
            state=np.array([[0,0,0,0]])
            state[0,nextState]=1.0
            # Keep track of state history
            stateHist=np.append(stateHist,state,axis=0)
            currentState=nextState
            # calculate the actual distribution over the 3 states so far
            totals=np.sum(stateHist,axis=0)
            gt=np.sum(totals)
            #print(totals,'totals',gt)
            distrib=totals/gt
            distrib=np.reshape(distrib,(1,4))
            distr_hist=np.append(distr_hist,distrib,axis=0)

        lst_dic2.append({'ExpID': ExpID,  'eta': eta, 'vartheta': vartheta, 'Bridge': Bridge,'Edge':Edge,'Subset':Subset,'Subsystem':Subsystem, 'Run': Run,
                         'TotalAvail': TotalAvail/SimulationTime,'TotalCost11':TotalCost11,'TotalCost12':TotalCost12,'TotalCost21':TotalCost21,'TotalCost22':TotalCost22})
        pd1 = pd1.append(lst_dic1)
        # print('distrib',distrib)
        # print('distrib-hist',distr_hist)
        # print(stateHist,'stateHist')
        #P_hat = stateChangeHist / stateChangeHist.sum(axis=1)[:, None]
        # Check estimated state transition probabilities based on history so far:
        # print(P_hat)
        #dfDistrHist = pd.DataFrame(distr_hist)
        # Plot the distribution as the simulation progresses over time
        #dfDistrHist.plot(title="Simulation History")
        #plt.show()

    pd2 = pd2.append(lst_dic2)
    return pd1,pd2,stateChangeHist,distr_hist


############SimulateEdgeJune2022

def EdgeSimulationJune2022(pdAlg1BridgeLevel,BridgeRoadMat,EdgeList,Subnet,Q10,Q11, Q20,Q21,RM,NumSimRun):
    lst_dic1=[]
    for eedge in range(0,len(BridgeRoadMat),1):
        Edge=EdgeList[eedge]
        NumBrdg=0
        for brdg in range(0,3,1):
            if BridgeRoadMat[eedge][brdg]>-1:
                NumBrdg+=1
        print(Edge,NumBrdg)
        for sim_run in range(NumSimRun):
            #lst_dic1=[]
            for BridgeID in range(0,NumBrdg,1):
                if np.random.random()<0.6:
                    MaintenanceRate=RM[BridgeRoadMat[eedge][BridgeID]][0]
                    if BridgeRoadMat[eedge][BridgeID]==0:
                        if BridgeID==0:
                            QQ=Q10
                        elif BridgeID==1:
                            QQ2Road=Q10
                        else:
                            QQ3Road = Q10
                    else:
                        if BridgeID==0:
                            QQ=Q11
                        elif BridgeID==1:
                            QQ2Road=Q11
                        else:
                            QQ3Road = Q11
                else:
                    MaintenanceRate = RM[BridgeRoadMat[eedge][BridgeID]][1]
                    if BridgeRoadMat[eedge][BridgeID] == 0:
                        if BridgeID==0:
                            QQ=Q20
                        elif BridgeID==1:
                            QQ2Road=Q20
                        else:
                            QQ3Road = Q20
                    else:
                        if BridgeID==0:
                            QQ=Q21
                        elif BridgeID==1:
                            QQ2Road=Q21
                        else:
                            QQ3Road = Q21
                    #[TotalAva,TotalCost,pi, time_spent] = simulate_cmc(Q2, 1100, 500)
                    #sim_run,BridgeID,MaintenanceRate,MaintenanceRatePolicy,Edge

            if NumBrdg==1:
                [TotalAva,TotalCost,pi,time_spent]=simulate_cmc(QQ)
            elif NumBrdg==2:
                [TotalAva, TotalCost, pi, time_spent] = simulate_cmcTwoRoads(QQ,QQ2Road)
            else:
                [TotalAva, TotalCost, pi, time_spent] = simulate_cmcThreeRoads(QQ,QQ2Road,QQ3Road)
            lst_dic1.append({'subnet':Subnet,'sim_run': sim_run, 'BridgeID': BridgeID,
                 'MaintenanceRate': MaintenanceRate, 'Edge': Edge,
                 'Avail': TotalAva, 'TotalCost': TotalCost})
    pdAlg1BridgeLevel = pdAlg1BridgeLevel.append(lst_dic1)
    #print(TotalAva,TotalCost,time_spent,pi)
    return pdAlg1BridgeLevel

###
def wwork():
    EdgeAll = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 40, 67, 68]
    print(EdgeAll.index(8))
    EdgeListWithBridges = [13, 40]
    EdgeListWithBridgesShortDis = [[13, 40,1]]
    EdgesWithTraffic = [67, 68]

    EdgesWithTrafficProb = [0.6, 0.7]
    EdgesWithTrafficOtherTime = [3, 5]
    BridgeRoadMat = [[1, -1, -1], [0, 0, 0]]  # [[0,0,-1],[1,-1,-1],[0,-1,-1],[1,-1,-1]]
    EdgeNumWithBridges = len(BridgeRoadMat)
    print('numbridge', EdgeNumWithBridges)
    TurningMat = [[0, 0.5, 0.22, 0.28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0.64, 0, 0, 0, 0.36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0.31, 0, 0, 0.19, 0, 0.12, 0.1, 0, 0, 0, 0, 0, 0, 0.28, 0],
                  [0.34, 0, 0.16, 0, 0.11, 0.17, 0, 0.22, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0.25, 0, 0.19, 0, 0.25, 0, 0.31, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0.08, 0.14, 0.12, 0, 0.11, 0.24, 0, 0, 0, 0, 0, 0.31, 0],
                  [0, 0, 0.07, 0, 0, 0.13, 0, 0, 0, 0, 0.1, 0.11, 0, 0.3, 0.29],
                  [0, 0, 0, 0.16, 0.13, 0.21, 0, 0, 0.26, 0.24, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0.44, 0, 0.28, 0.28, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0.6, 0.4, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0.13, 0, 0.25, 0, 0, 0.17, 0, 0, 0.45],
                  [0, 0, 0, 0, 0, 0, 0.17, 0, 0, 0, 0.22, 0, 0, 0, 0.61],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0.24, 0, 0, 0.4, 0.36, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0.28, 0, 0, 0, 0.35, 0.37, 0, 0, 0]
                  ]
    TurningMatMod = [[0, 0.5, 0.22, 0.28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0.64, 0, 0, 0, 0.36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0.31, 0, 0, 0.19, 0, 0.12, 0.1, 0, 0, 0, 0, 0, 0, 0.28, 0],
                     [0.34, 0, 0.16, 0, 0.11, 0.17, 0, 0.22, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0.25, 0, 0.19, 0, 0.25, 0, 0.31, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0.08, 0.14, 0.12, 0, 0.11, 0.24, 0, 0, 0, 0, 0, 0.31, 0],
                     [0, 0, 0.07, 0, 0, 0.13, 0, 0, 0, 0, 0.1, 0.11, 0, 0.3, 0.29],
                     [0, 0, 0, 0.16, 0.13, 0.21, 0, 0, 0.26, 0.24, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0.44, 0, 0.28, 0.28, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0.6, 0.4, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0.13, 0, 0.25, 0, 0, 0.17, 0, 0, 0.45],
                     [0, 0, 0, 0, 0, 0, 0.17, 0, 0, 0, 0.22, 0, 0, 0, 0.61],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0.24, 0, 0, 0.4, 0.36, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0.28, 0, 0, 0, 0.35, 0.37, 0, 0, 0]
                     ]
    TravelingTimeEdge = [1, 8, 3, 1, 3, 1, 5, 2, 4, 9, 5, 12, 12, 2, 3]
    print('ii')
    print(TurningMat)



#############Start Main function
def MainSubsystem():
    #MC simulation
    #    time=1100
    #    warm_up=500
    # CalHighDensityReg with 0.6

    #Q = matrix(QQ, [[-3, 2, 1], [1, -5, 4], [1, 8, -9]])
    resultsPath = 'C:\\Mohsen\\2020-12-20-JanBerlin\\ESREL2022\\20220421Results\\'
    if not os.path.exists(resultsPath): os.makedirs(resultsPath)
    MeasureList=['Avail','TotalCost']
    MeasureList22=['TravelTime','TotalCost']


    #Inputs of Edge Simulation
    pdAlg1BridgeLevel = pd.DataFrame(columns=['sim_run','BridgeID','MaintenanceRate','Edge', 'Avail','TotalCost'])
    NumSimRun=1000
    RM = [[0.2, 0.5], [0.15, 0.4]]
    Q10 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.25, 0.05, RM[0][0]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
    Q11 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.2, 0.05, RM[1][0]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
    Q20 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.55, 0.05, RM[0][1]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
    Q21 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.45, 0.05, RM[1][1]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
    print(RM[0][0],RM[1][0]) # these happen with 0.6 probabilities.

    #Inputs of second alg
    MeasureList=['Avail','TotalCost']
    n_sample=300#for building density
    OverallSample=400#for generation
    PlotDist=0
    pdAlg2InputSample = pd.DataFrame(columns=['sample_run','Edge', 'Avail','TotalCost'])


    #Inputs of third algorithm
    IncomMat1 = ([[0, 0.2, 0.4, 0.4], [0.15, 0.5, 0.15, 0.2], [0.2, 0.3, 0.5, 0], [0.23, 0.1, 0, .67]])
    IncomMat2 = ([[0, 0.2, 0.4, 0.4], [0.15, 0.5, 0.15, 0.2], [0.1, 0.15, 0.75, 0], [0.23, 0.1, 0, .67]])
    print(IncomMat1[0][3])
    pdAlg2output = pd.DataFrame(columns=['sample_run', 'TravelTime','TotalCost'])
    aaa = 2  # 5#2#5
    bbb = 5  # 1#5#1



    #Inputs of conditional prob functions
    CondiProbTableLevels21 = pd.DataFrame(columns=['DepenVar', 'DepenVarLvel','IndepenVar','IndepenVarLvel','CondProb'])
    InputVarLavelNum=2
    CondiProbTableLevels10 = pd.DataFrame(columns=['DepenVar', 'DepenVarLvel','IndepenVar','IndepenVarLvel','IndepenVar1', 'IndepenVar1Lvel','CondProb'])



    ####Functions############################################################################
    pdAlg1BridgeLevel=EdgeSimulation(pdAlg1BridgeLevel, Q10,Q11, Q20,Q21, RM, NumSimRun)
    EdgeNum=2
    #Sampling from density function
    pdAlg2InputSample=Sampling4InputAlg2(resultsPath,pdAlg1BridgeLevel,pdAlg2InputSample,MeasureList,EdgeNum,OverallSample,PlotDist,n_sample)


    #for each row of pdAlg2InputSample, check the incoming transition probability matrix,
    #and then, select one, and then use that, also apply availability in matrix,
    #and then, calculate Kemeny constant
    ##For TotalCost aggregation
    #        term=np.sum(selData['TotalCost'].to_list())
    #        term1=term-(np.sqrt(0.1*term)/(2*beta.rvs(aaa,bbb)))
    pdAlg2output=SimulateAlg2Output(IncomMat1,IncomMat2,pdAlg2output,pdAlg2InputSample,OverallSample,aaa,bbb)

    ####Functions############################################################################
    #discretize
    #Inputs of discretization algorithm
    nDiscretise = 3
    DataFrameForDisLay2outRoad12=pdAlg1BridgeLevel[(pdAlg1BridgeLevel['Edge'] == 0)] # layer 2-out
    DataFrameForDisLay1InputRoad12 = pdAlg2InputSample[(pdAlg2InputSample['Edge'] == 0)] #layer 1-input
    DataFrameForDisLay2outRoad13=pdAlg1BridgeLevel[(pdAlg1BridgeLevel['Edge'] == 1)] # layer 2-out
    DataFrameForDisLay1InputRoad13 = pdAlg2InputSample[(pdAlg2InputSample['Edge'] == 1)] #layer 1-input
    DataFrameForDisLay0out = pdAlg2output#layer0 out DataFrameForDis22

    for MeasureId in range(0, 2, 1):
        Measure=MeasureList[MeasureId]
        Measure22=MeasureList22[MeasureId]
        if MeasureId==0:
            [DataFrameForDisLay0out,DiscLay0out,binsLay0out]=DiscretizeDataFrame(DataFrameForDisLay0out,Measure22,nDiscretise)
            for edge in range(0,2,1):
                if edge==0:
                    [DataFrameForDisLay2outRoad12, DiscLay2outRoad12, binsLay2out] = DiscretizeDataFrame(DataFrameForDisLay2outRoad12,Measure, nDiscretise)
                    ##discretise inputs of alg2
                    DataFrameForDisLay1InputRoad12[Measure] = DataFrameForDisLay1InputRoad12[Measure].astype(float)
                    [DataFrameForDisLay1InputRoad12,DiscLay1InputRoad12]=DiscretizeDataFrameWithBins(DataFrameForDisLay1InputRoad12, Measure, binsLay2out)
                else:
                    [DataFrameForDisLay2outRoad13, DiscLay2outRoad13, binsLay2out] = DiscretizeDataFrame(DataFrameForDisLay2outRoad13,Measure, nDiscretise)
                    ##discretise inputs of alg2
                    DataFrameForDisLay1InputRoad13[Measure] = DataFrameForDisLay1InputRoad13[Measure].astype(float)
                    [DataFrameForDisLay1InputRoad13,DiscLay1InputRoad13]=DiscretizeDataFrameWithBins(DataFrameForDisLay1InputRoad13, Measure, binsLay2out)
        else:
            [DataFrameForDisLay0out,DiscLay0out11,binsLay0out11]=DiscretizeDataFrame(DataFrameForDisLay0out,Measure22,nDiscretise)
            for edge in range(0,2,1):
                if edge==0:
                    [DataFrameForDisLay2outRoad12,DiscLay2out11Road12,binsLay2out11]=DiscretizeDataFrame(DataFrameForDisLay2outRoad12,Measure,nDiscretise)
                    ##discretise inputs of alg2
                    DataFrameForDisLay1InputRoad12[Measure] = DataFrameForDisLay1InputRoad12[Measure].astype(float)
                    [DataFrameForDisLay1InputRoad12,DiscLay1Input11Road12]=DiscretizeDataFrameWithBins(DataFrameForDisLay1InputRoad12, Measure, binsLay2out11)
                else:
                    [DataFrameForDisLay2outRoad13,DiscLay2out11Road13,binsLay2out11]=DiscretizeDataFrame(DataFrameForDisLay2outRoad13,Measure,nDiscretise)
                    ##discretise inputs of alg2
                    DataFrameForDisLay1InputRoad13[Measure] = DataFrameForDisLay1InputRoad13[Measure].astype(float)
                    [DataFrameForDisLay1InputRoad13,DiscLay1Input11Road13]=DiscretizeDataFrameWithBins(DataFrameForDisLay1InputRoad13, Measure, binsLay2out11)



    pd.DataFrame.from_dict(data=DiscLay2outRoad12,orient='index').to_csv(os.path.join(resultsPath + "2022-04-01-DiscLay2outRoad12" + '.csv'))
    pd.DataFrame.from_dict(data=DiscLay1InputRoad12,orient='index').to_csv(os.path.join(resultsPath + "2022-04-01-DiscLay1InputRoad12" + '.csv'))
    pd.DataFrame.from_dict(data=DiscLay2outRoad13,orient='index').to_csv(os.path.join(resultsPath + "2022-04-01-DiscLay2outRoad13" + '.csv'))
    pd.DataFrame.from_dict(data=DiscLay1InputRoad13,orient='index').to_csv(os.path.join(resultsPath + "2022-04-01-DiscLay1InputRoad13" + '.csv'))
    pd.DataFrame.from_dict(data=DiscLay0out,orient='index').to_csv(os.path.join(resultsPath + "2022-04-01-DiscLay0out" + '.csv'))
    pd.DataFrame.from_dict(data=DiscLay2out11Road12,orient='index').to_csv(os.path.join(resultsPath + "2022-04-01-DiscLay2out11Road12" + '.csv'))
    pd.DataFrame.from_dict(data=DiscLay1Input11Road12,orient='index').to_csv(os.path.join(resultsPath + "2022-04-01-DiscLay1Input11Road12" + '.csv'))
    pd.DataFrame.from_dict(data=DiscLay2out11Road13,orient='index').to_csv(os.path.join(resultsPath + "2022-04-01-DiscLay2out11Road13" + '.csv'))
    pd.DataFrame.from_dict(data=DiscLay1Input11Road13,orient='index').to_csv(os.path.join(resultsPath + "2022-04-01-DiscLay1Input11Road13" + '.csv'))
    pd.DataFrame.from_dict(data=DiscLay0out11,orient='index').to_csv(os.path.join(resultsPath + "2022-04-01-DiscLay0out11" + '.csv'))

    DataFrameForDisLay1Input=pd.concat([DataFrameForDisLay1InputRoad12,DataFrameForDisLay1InputRoad13])
    DataFrameForDisLay2out=pd.concat([DataFrameForDisLay2outRoad12,DataFrameForDisLay2outRoad13])
    print(DataFrameForDisLay1Input.columns,DataFrameForDisLay0out.columns)

    ####Functions############################################################################
    #calculate Conditional prob Tables
    #RM[BridgeID][0]
    # for each 'Edge', look at 'MaintenanceRate', 'AvailDisc11',  'TotalCostDisc11'
    CondiProbTableLevels21=ConditionalProbTabLveles21Generate(DataFrameForDisLay2out,CondiProbTableLevels21,nDiscretise,InputVarLavelNum,RM )
    print(CondiProbTableLevels21)
    CondiProbTableLevels10=ConditionalProbTabLveles10Generate(CondiProbTableLevels10,DataFrameForDisLay1Input,DataFrameForDisLay0out,nDiscretise,OverallSample)
    print(CondiProbTableLevels10[['DepenVarLvel','IndepenVarLvel','IndepenVar1Lvel','CondProb']])

    ##Save conditional prob tables into csv dataframe files
    CondiProbTableLevels10.to_csv(os.path.join(resultsPath + "2022-04-01-CondiProbTableLevels10" + '.csv'))
    CondiProbTableLevels21.to_csv(os.path.join(resultsPath + "2022-04-01-CondiProbTableLevels21" + '.csv'))



    print(pdAlg2InputSample[['Avail','TotalCost']])

##################################End of Main Subsystem
#MainSubsystem()







###############################################################################
#############################################################################3
DataFrameForDisLay2out.to_csv(os.path.join(resultsPath+"DataFrameForDisLay2out.csv"))
DataFrameForDisLay2outConCat.to_csv(os.path.join(resultsPath+"DataFrameForDisLay2outConCat.csv"))
DataFrameForDisLay0outConCat.to_csv(os.path.join(resultsPath+"DataFrameForDisLay0outConCat.csv"))
print(DataFrameForDisLay2outConCat.columns)
print(DataFrameForDisLay1Input.columns)

exit()


DataFrameForDisLay2outRoad12 = pdAlg1BridgeLevel[(pdAlg1BridgeLevel['Edge'] == 0)]  # layer 2-out

[DataFrameForDisLay2outRoad12, DiscLay2outRoad12, binsLay2out] = DiscretizeDataFrame(DataFrameForDisLay2outRoad12,Measure, nDiscretise)
DataFrameForDisLay1InputRoad12[Measure] = DataFrameForDisLay1InputRoad12[Measure].astype(float)
[DataFrameForDisLay1InputRoad12,DiscLay1InputRoad12]=DiscretizeDataFrameWithBins(DataFrameForDisLay1InputRoad12, Measure, binsLay2out)


exit()


####working start
for ii in range(8):
    [TotalAva,TotalCost,pi,time_spent]=simulate_cmcThreeRoads(Q20,Q20,Q20)#simulate_cmcTwoRoads(Q20,Q20)#simulate_cmcThreeRoads(Q20,Q10,Q21) #simulate_cmcTwoRoads(Q20,Q20) #simulate_cmc(Q10)
    print(TotalAva)
##########working end
exit()
#print(DataFrameForDis.dtypes)
#print(DataFrameForDis22Input.dtypes)
#print(DataFrameForDisLay2out[['Avail','AvailDisc']])
#print(DataFrameForDisLay2out[['AvailDisc11','TotalCostDisc11']])
#print(DataFrameForDisLay0out[['TravelTime','TravelTimeDisc','TravelTimeDisc11']])
#print(DataFrameForDisLay0out[['TotalCost','TotalCostDisc','TotalCostDisc11']])
#print(DataFrameForDisLay1Input[['Avail','AvailDisc','AvailDisc11','TotalCostDisc11']])
#print(DataFrameForDisLay1Input[['TotalCostDisc11']])
#print(DiscLay2out)
#print(DiscLay0out)
#print(DiscLay1Input)
##print(DataFrameForDisLay2out[['MaintenanceRate','AvailDisc11','TotalCostDisc11']])
##print(DataFrameForDisLay2out.columns)
#print(DataFrameForDisLay1Input[['sample_run','Edge','AvailDisc11','TotalCostDisc11']])
#print(DataFrameForDisLay0out[['sample_run','TravelTimeDisc11','TotalCostDisc11']])




################          ENDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD




































#Groads=create_graph(shapefile_location)
#streets = geopandas.read_file(momepy.datasets.get_path('bubenec'), layer='streets')
#f, ax = plt.subplots(figsize=(10, 10))
#streets.plot(ax=ax)
#ax.set_axis_off()
#print(shapefile.columns)
#print(shapefile["naam"])
#print(shapefile["lengte"])
#print(shapefile[ "wegtype"])
#print(shapefile["meetgeg"])

#shapefilefilter=shapefile[shapefile['naam'].notnull()]
#shapefilefilter1=shapefilefilter[shapefilefilter['naam'].str.startswith(('A'))]
#print(shapefilefilter1)
#plt.figure(figsize=(10, 10))
#shapefilefilter1.plot()
#plt.savefig(os.path.join("C:\\Mohsen\\2020-11-01-DataSetsEindhoven\\01-01-2019\\MohsenFolder\\LinkswithA"+'.png'), format="PNG", dpi=600)

#shapefilefilter1=shapefilefilter[shapefilefilter['naam'].str.startswith(('N'))]
#print(shapefilefilter1)
#plt.figure(figsize=(10, 10))
#shapefilefilter1.plot()
#plt.savefig(os.path.join("C:\\Mohsen\\2020-11-01-DataSetsEindhoven\\01-01-2019\\MohsenFolder\\LinkswithN"+'.png'), format="PNG", dpi=600)

#plt.figure(figsize=(10, 10))
#graph = momepy.gdf_to_nx(shapefilefilter1, approach='primal')
#nx.draw(graph)
#plt.savefig(os.path.join("C:\\Mohsen\\2020-11-01-DataSetsEindhoven\\01-01-2019\\MohsenFolder\\LinkswithNnetwork"+'.png'), format="PNG", dpi=600)


#shapefilefilter1=shapefilefilter[shapefilefilter['naam'].str.startswith(('A','N'))]
#print(shapefilefilter1)
#plt.figure(figsize=(10, 10))
#shapefilefilter1.plot()
#plt.savefig(os.path.join("C:\\Mohsen\\2020-11-01-DataSetsEindhoven\\01-01-2019\\MohsenFolder\\LinkswithAN"+'.png'), format="PNG", dpi=600)

#print(shapefile["naam"])
#print(shapefile["meetgeg"])
#c=fiona.open(shapefile_location)
#df=geopandas.GeoDataFrame.from_features(c[0:300])
#print(df)
#shapefilesub3=shapefile[shapefile.wegtype==3]
#print(shapefilesub3)
#graph11 = momepy.gdf_to_nx(shapefile, approach='primal')
#nx.draw(graph11)


