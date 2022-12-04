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
import more_itertools
from scipy.interpolate import interp1d
from CaseStudyNetwork.Network import *
from scipy.stats import beta
from itertools import product, combinations, combinations_with_replacement


class Roads():
    def __init__(self):
        pass


    ############SimulateEdgeJuly2022

    def edge_simulation_july_2022(self, pdAlg1BridgeLevel, BridgeRoadMat, EdgeList, Subnet, Q10, Q11, Q20, Q21, RM, NumSimRun,
                                  Simulatingtime, warm_up, RehabFullCost, MaintenanceCost, AvailCoeff):
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
                ListMaintenanceRates=[]
                ListMaintenanceRatesCoded =-5
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
                    ListMaintenanceRates.append(MaintenanceRate)

                #ListMaintenanceRatesCoded=set(ListMaintenanceRates)
                if len(set(ListMaintenanceRates))==1 or len(ListMaintenanceRates)==1:
                    if RM[0][0] in set(ListMaintenanceRates) or RM[1][0] in set(ListMaintenanceRates):
                        ListMaintenanceRatesCoded=0
                    elif RM[0][1] in set(ListMaintenanceRates) or RM[1][1] in set(ListMaintenanceRates):
                        ListMaintenanceRatesCoded=1
                else:
                    ListMaintenanceRatesCoded=2

                if NumBrdg==1:
                    [TotalAva,TotalCost,pi,time_spent] = self.simulate_cmc(QQ,Simulatingtime, warm_up,RehabFullCost,MaintenanceCost,AvailCoeff)
                elif NumBrdg==2:
                    [TotalAva, TotalCost, pi, time_spent] = self.simulate_cmcTwoRoads(QQ,QQ2Road,Simulatingtime, warm_up,RehabFullCost,MaintenanceCost,AvailCoeff)
                else:
                    [TotalAva, TotalCost, pi, time_spent] = self.simulate_cmcThreeRoads(QQ,QQ2Road,QQ3Road,Simulatingtime, warm_up,RehabFullCost,MaintenanceCost,AvailCoeff)
                lst_dic1.append({'subnet':Subnet,'sim_run': sim_run, 'BridgeID': BridgeID,
                     'ListMaintenanceRates': ListMaintenanceRates,'ListMaintenanceRatesCoded':ListMaintenanceRatesCoded, 'Edge': Edge,
                     'Avail': TotalAva, 'TotalCost': TotalCost})
        pdAlg1BridgeLevel = pdAlg1BridgeLevel.append(lst_dic1)
        #print(TotalAva,TotalCost,time_spent,pi)
        return pdAlg1BridgeLevel


    def simulate_cmc(self, QQ, time, warm_up,RehabFullCost,MaintenanceCost,AvailCoeff):#,sim_run,BridgeID,MaintenanceRate,MaintenanceRatePolicy,Edge):
        #time=1100
        #warm_up=500
        TotalCost= 0.0
        #RehabFullCost = 200
        #MaintenanceCost = 10
        #Q1 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.25, 0.05, MaintenanceRate, [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
        #Q2 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.55, 0.05, MaintenanceRate, [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
        #RehabFullCost22 = 300
        #MaintenanceCost22 = 20
        #RehabpartialCost = 100
        # RehabFullCost=200
        # MaintenanceCost=10
        #AvailCoeff = [1.0, 0.8, 0.0, 0.6]#AvailCoeff = [1.0, 0.6, 0.0, 0.4]
        QQL = list(QQ)  # In case a matrix is input
        state_space = range(len(QQL))  # Index the state space
        time_spent = {s:0 for s in state_space}  # Set up a dictionary to keep track of time
        clock = 0  # Keep track of the clock
        current_state = 0  # First state
        while clock < time:
            # Sample the transitions
            sojourn_times = [self.sample_from_rate(rate) for rate in QQL[current_state][:current_state]]
            sojourn_times += [np.inf]  # An infinite sojourn to the same state
            sojourn_times += [self.sample_from_rate(rate) for rate in QQL[current_state][current_state + 1:]]

            # Identify the next state
            next_state = min(state_space, key=lambda x: sojourn_times[x])
            #print('next',next_state)
            if next_state==3:
                TotalCost += MaintenanceCost
            elif next_state==2:
                TotalCost += RehabFullCost


            sojourn = sojourn_times[next_state]
            clock += sojourn
            if clock > warm_up:  # Keep track if past warm up time
                time_spent[current_state] += sojourn
            current_state = next_state  # Transition

        pi = [time_spent[state] / sum(time_spent.values()) for state in state_space]  # Calculate probabilities
        TotalAva=sum(x * y for x, y in zip(pi, AvailCoeff))
        return TotalAva,TotalCost,pi,time_spent

    ######SimulateMCTwoRoads
    def simulate_cmcTwoRoads(self, QQ,QQ2Road,time, warm_up,RehabFullCost,MaintenanceCost,AvailCoeff):
        #time=1100
        #warm_up=500
        TotalCost= 0.0
        #RehabFullCost = 200
        #MaintenanceCost = 10
        #AvailCoeff = [1.0, 0.8, 0.0, 0.6]
        QQL = list(QQ)  # In case a matrix is input
        state_space = range(len(QQL))  # Index the state space
        time_spent = {s:0 for s in state_space}  # Set up a dictionary to keep track of time
        QQL2Road = list(QQ2Road)  # In case a matrix is input
        state_space2Road = range(len(QQL2Road))  # Index the state space
        time_spent2Road = {s:0 for s in state_space2Road}  # Set up a dictionary to keep track of time
        clock = 0  # Keep track of the clock
        current_state = 0  # First state
        current_state2Road = 0  # First state
        while clock < time:
            # Sample the transitions
            if clock==0 or nextUpdate==0:
                sojourn_times = [self.sample_from_rate(rate) for rate in QQL[current_state][:current_state]]
                sojourn_times += [np.inf]  # An infinite sojourn to the same state
                sojourn_times += [self.sample_from_rate(rate) for rate in QQL[current_state][current_state + 1:]]
            #print(sojourn_times)
            if clock==0 or nextUpdate==1:
                sojourn_times2Road = [self.sample_from_rate(rate) for rate in QQL2Road[current_state][:current_state]]
                sojourn_times2Road += [np.inf]  # An infinite sojourn to the same state
                sojourn_times2Road += [self.sample_from_rate(rate) for rate in QQL2Road[current_state][current_state + 1:]]
            # Identify the next state
            next_state = min(state_space, key=lambda x: sojourn_times[x])
            next_state2Road = min(state_space2Road, key=lambda x: sojourn_times2Road[x])
            #print(next_state,next_state2Road)
            if next_state==2 or next_state2Road==2:
                next_stateAllAva=2
            else:
                next_stateAllAva=max(next_state,next_state2Road)
            #print('next',next_state)

            MinTime = min(clock+sojourn_times[next_state], clock+sojourn_times2Road[next_state2Road])
            if MinTime==clock+sojourn_times[next_state]:
                next_stateAll=next_state
                sojourn = sojourn_times[next_state]
                nextUpdate=0
            else:
                next_stateAll=next_state2Road
                sojourn = sojourn_times2Road[next_state2Road]
                nextUpdate=1

            if next_stateAll==3:
                TotalCost += MaintenanceCost
            elif next_stateAll==2:
                TotalCost += RehabFullCost


            #sojourn = sojourn_times[next_stateAll]
            clock += sojourn
            if clock > warm_up:  # Keep track if past warm up time
                #time_spent[current_state] += sojourn
                time_spent[next_stateAllAva]+=sojourn
            current_state = next_stateAll#next_state  # Transition
            #print(time_spent, sojourn,sojourn_times,sojourn_times2Road)

        pi = [time_spent[state] / sum(time_spent.values()) for state in state_space]  # Calculate probabilities
        TotalAva=sum(x * y for x, y in zip(pi, AvailCoeff))
        return TotalAva,TotalCost,pi,time_spent

    ######SimulateMCThreeRoads
    def simulate_cmcThreeRoads(self, QQ,QQ2Road,QQ3Road,time, warm_up,RehabFullCost,MaintenanceCost,AvailCoeff):
        #time=1200
        #warm_up=500
        TotalCost= 0.0
        #RehabFullCost = 200
        #MaintenanceCost = 10
        #AvailCoeff = [1.0, 0.8, 0.0, 0.6]
        QQL = list(QQ)  # In case a matrix is input
        state_space = range(len(QQL))  # Index the state space
        time_spent = {s:0 for s in state_space}  # Set up a dictionary to keep track of time
        QQL2Road = list(QQ2Road)  # In case a matrix is input
        state_space2Road = range(len(QQL2Road))  # Index the state space
        QQL3Road = list(QQ3Road)  # In case a matrix is input
        state_space3Road = range(len(QQL3Road))  # Index the state space
        clock = 0  # Keep track of the clock
        current_state = 0  # First state
        current_state2Road = 0  # First state
        current_state3Road = 0  # First state
        while clock < time:
            # Sample the transitions
            if clock==0 or nextUpdate==0:
                sojourn_times = [self.sample_from_rate(rate) for rate in QQL[current_state][:current_state]]
                sojourn_times += [np.inf]  # An infinite sojourn to the same state
                sojourn_times += [self.sample_from_rate(rate) for rate in QQL[current_state][current_state + 1:]]
            #print(sojourn_times)
            if clock==0 or nextUpdate==1:
                sojourn_times2Road = [self.sample_from_rate(rate) for rate in QQL2Road[current_state][:current_state]]
                sojourn_times2Road += [np.inf]  # An infinite sojourn to the same state
                sojourn_times2Road += [self.sample_from_rate(rate) for rate in QQL2Road[current_state][current_state + 1:]]
            if clock==0 or nextUpdate==2:
                sojourn_times3Road = [self.sample_from_rate(rate) for rate in QQL3Road[current_state][:current_state]]
                sojourn_times3Road += [np.inf]  # An infinite sojourn to the same state
                sojourn_times3Road += [self.sample_from_rate(rate) for rate in QQL3Road[current_state][current_state + 1:]]
            # Identify the next state
            next_state = min(state_space, key=lambda x: sojourn_times[x])
            next_state2Road = min(state_space2Road, key=lambda x: sojourn_times2Road[x])
            next_state3Road = min(state_space3Road, key=lambda x: sojourn_times3Road[x])
            #print(next_state,next_state2Road,next_state3Road)
            if next_state==2 or next_state2Road==2 or next_state3Road==2:
                next_stateAllAva=2
            else:
                next_stateAllAva=max(next_state,next_state2Road,next_state3Road)
            #print('next',next_state)

            MinTime = min(clock+sojourn_times[next_state], clock+sojourn_times2Road[next_state2Road],clock+sojourn_times3Road[next_state3Road])
            if MinTime==clock+sojourn_times[next_state]:
                next_stateAll=next_state
                sojourn = sojourn_times[next_state]
                nextUpdate=0
                #current_state = next_state
            elif MinTime==clock+sojourn_times2Road[next_state2Road]:
                next_stateAll=next_state2Road
                sojourn = sojourn_times2Road[next_state2Road]
                nextUpdate=1
                #current_state = next_state2Road
            else:
                next_stateAll=next_state3Road
                sojourn = sojourn_times3Road[next_state3Road]
                nextUpdate=2
                #current_state = next_state3Road


            if next_stateAll==3:
                TotalCost += MaintenanceCost
            elif next_stateAll==2:
                TotalCost += RehabFullCost


            #sojourn = sojourn_times[next_stateAll]
            clock += sojourn
            if clock > warm_up:  # Keep track if past warm up time
                #time_spent[current_state] += sojourn
                time_spent[next_stateAllAva]+=sojourn
            current_state = next_stateAll#next_state  # Transition
            #print(time_spent, sojourn,sojourn_times,sojourn_times2Road)

        pi = [time_spent[state] / sum(time_spent.values()) for state in state_space]  # Calculate probabilities
        TotalAva=sum(x * y for x, y in zip(pi, AvailCoeff))
        return TotalAva,TotalCost,pi,time_spent

    ####NewChangesIN2022March
    def sample_from_rate(self, rate):
        import random
        if rate == 0:
            return np.inf
        return random.expovariate(rate)



    ##########Sampling4InputAlg2 Function

    def sampling_inputs_algorithm2(self, resultsPath, pdAlg1BridgeLevel, pdAlg2InputSample, MeasureList, EdgeList, EdgeNum, Subnet, OverallSample, PlotDist, n_sample):
        for EdgeID in range(0,EdgeNum,1):
            for sample_run in range(OverallSample):
                lst_dic1 = []
                for MeasureId in range(0, 2, 1):
                    #MeasureValue = []
                    Sample=[0]
                    selData = pdAlg1BridgeLevel.loc[(pdAlg1BridgeLevel["Edge"] == EdgeList[EdgeID])]
                    Sample.append(list(selData[MeasureList[MeasureId]].sample(n=n_sample,replace=True, random_state=1)))
                    Sample.append(1.0)
                    Sample = list(more_itertools.collapse(Sample))
                    Sample_sorted = np.sort(Sample)
                    CDFProbval = [0]
                    for ii11 in range(1, n_sample+1, 1):
                        val = list(Sample_sorted).index(Sample_sorted[ii11 - 1]) + 1.0
                        CDFProbval.append(val / (n_sample+1.0))
                    CDFProbval.append(1.0)
                    #print(len(Sample_sorted),len(CDFProbval))
                    CDFfunction = interp1d(Sample_sorted, CDFProbval)
                    CDFfunctionInv = interp1d(CDFProbval, Sample_sorted)
                    # Sample_sorted[np.argwhere()]
                    # print(Measure)#,CDFfunctionInv(CDFfunction(0.3)))
                    # eeee=(np.random.uniform(0,1,100))
                    # print(CDFfunctionInv(np.random.uniform(0,1)))
                    if MeasureId==0:
                        MeasureValue0=[]
                        MeasureValue0.append(CDFfunctionInv(np.random.uniform(0, 1)))
                    else:
                        MeasureValue1 = []
                        MeasureValue1.append(CDFfunctionInv(np.random.uniform(0, 1)))
                        #MeasureValue.append(MeasureValueOne[0])
                        lst_dic1.append({'subnet':Subnet,'sample_run': sample_run,
                             'Edge': EdgeList[EdgeID],'Avail': MeasureValue0[0], 'TotalCost': MeasureValue1[0]})

                pdAlg2InputSample = pdAlg2InputSample.append(lst_dic1)
        #PlotDist=0
        if PlotDist == 1:
            for jjj in range(0, 2, 1):
                Measure = MeasureList[jjj]
                fig, ax = plt.subplots()
                ax2 = ax.twinx()
                ax.hist(pdAlg2InputSample[Measure], bins=100, density=True)
                ax2.hist(pdAlg2InputSample[Measure], cumulative=1, histtype='step', bins=100, color='tab:orange', density=True)
                ax.set_xlim((ax.get_xlim()[0], pdAlg2InputSample[Measure].max()))
                File1 = os.path.abspath(resultsPath + "-Alg2Sample-" + Measure + ".png")
                plt.savefig(File1, format="PNG", dpi=800)

        return pdAlg2InputSample


