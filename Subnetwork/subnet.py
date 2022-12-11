

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
import datetime
from MarkovChain.Markov_chain_new import MarkovChain
from CaseStudyNetwork.Network import *
from scipy.stats import beta
from itertools import product, combinations, combinations_with_replacement
#normalizer = sklearn.preprocessing.Normalization(axis=-1)
from sklearn import preprocessing

class Subnetworks():
    def __init__(self):
        pass

    ############### SimulateAlg2Output function
    # write function with input edge number and returning edge index in matrix
    # write function with input edge number and returning whether edge has bridge or not
    # write fnction  with input edge number and returning whether edge is traffic inflow/outflow
    def simulate_algorithm2_output_generation(self, subnetwork, EdgeAll, EdgeListWithBridges, EdgeListWithBridgesShortDis,
                                              EdgesWithTraffic, EdgesWithTrafficProb, EdgesWithTrafficOtherTime, BridgeRoadMat,
                                              TurningMat, TurningMatMod, TravelingTimeEdge, EdgeNumWithBridges,
                                              pdAlg2output, pdAlg2InputSample, OverallSample):
        # EdgeAll,EdgeListWithBridges,EdgesWithTraffic,EdgesWithTrafficProb,EdgesWithTrafficOtherTime,BridgeRoadMat,TurningMat,TravelingTimeEdge,EdgeNumWithBridges
        # TurningMatMod=TurningMat
        # for index, row in pdAlg2InputSample.iterrows():
        # print(TurningMatMod)
        TurningMatMod11 = copy.deepcopy(TurningMatMod)
        MincostAll = 0
        for edge in EdgeAll:
            if edge in EdgeListWithBridges:
                selDataMinCost = pdAlg2InputSample.loc[(pdAlg2InputSample[
                                                            'Edge'] == edge)]  # &(pdAlg2InputSample['MaintenanceRate'] == RM[edge][inleval[0]])]
                term = np.sum(selDataMinCost['TotalCost'].to_list())
                MincostAll = min(term / OverallSample, MincostAll)

        selDataCostEdgeValAveragedAcrossRuns = []
        for edge in EdgeAll:
            if edge in EdgeListWithBridges:
                selDataCostEdge = pdAlg2InputSample.loc[(pdAlg2InputSample["Edge"] == edge)]
                selDataCostEdgeVal = selDataCostEdge['TotalCost'].to_list()
                selDataCostEdgeValAveragedAcrossRuns.append(sum(selDataCostEdgeVal)/len(selDataCostEdgeVal))

        selDataCostEdgeValAveragedAcrossRunsMin = min(selDataCostEdgeValAveragedAcrossRuns)

        for sample_run in range(OverallSample):  # OverallSample
            TurningMatMod = TurningMatMod11
            for edge in EdgeAll:
                if not edge in EdgesWithTraffic:
                    Indx = EdgeAll.index(edge)
                    TurningMatMod[Indx][Indx] = (TravelingTimeEdge[Indx] - 1) / (TravelingTimeEdge[Indx])
                    # print(TurningMatMod[Indx], edge, TravelingTimeEdge[Indx])
                    for edgprime in EdgeAll:
                        if edge != edgprime:
                            Indxprime = EdgeAll.index(edgprime)
                            TurningMatMod[Indx][Indxprime] = (1 - TurningMatMod[Indx][Indx]) * (
                            TurningMat[Indx][Indxprime])
                    # print(TurningMatMod[Indx],edge,TravelingTimeEdge[Indx])
                else:
                    IndxTr = EdgesWithTraffic.index(edge)
                    Indx = EdgeAll.index(edge)
                    if np.random.random() < EdgesWithTrafficProb[IndxTr]:
                        TravelTimeSelf = TravelingTimeEdge[Indx]
                    else:
                        TravelTimeSelf = EdgesWithTrafficOtherTime[IndxTr]
                    TurningMatMod[Indx][Indx] = (TravelTimeSelf - 1) / (TravelTimeSelf)
                    for edgprime in EdgeAll:
                        if edge != edgprime:
                            Indxprime = EdgeAll.index(edgprime)
                            TurningMatMod[Indx][Indxprime] = (1 - TurningMatMod[Indx][Indx]) * (
                            TurningMat[Indx][Indxprime])

            # np.set_printoptions(threshold=sys.maxsize)
            # print('hereeeee')
            # print(TurningMatMod)
            # if row["sample_run"] == sample_run:
            lst_dic1 = []
            selData = pdAlg2InputSample.loc[(pdAlg2InputSample["sample_run"] == sample_run)]

            IncomMatEdit = []
            # IncomMatEdit =  IncomMat
            for edge in EdgeAll:
                if not edge in EdgesWithTraffic:
                    Indx = EdgeAll.index(edge)
                    TurningMatMod[Indx][Indx] = (TravelingTimeEdge[Indx] - 1) / (TravelingTimeEdge[Indx])
                    for edgprime in EdgeAll:
                        if edge != edgprime:
                            Indxprime = EdgeAll.index(edgprime)
                            TurningMatMod[Indx][Indxprime] = (1 - TurningMatMod[Indx][Indx]) * (
                            TurningMat[Indx][Indxprime])
                else:
                    IndxTr = EdgesWithTraffic.index(edge)
                    Indx = EdgeAll.index(edge)
                    if np.random.random() < EdgesWithTrafficProb[IndxTr]:
                        TravelTimeSelf = TravelingTimeEdge[Indx]
                    else:
                        TravelTimeSelf = EdgesWithTrafficOtherTime[IndxTr]
                    TurningMatMod[Indx][Indx] = (TravelTimeSelf - 1) / (TravelTimeSelf)
                    for edgprime in EdgeAll:
                        if edge != edgprime:
                            Indxprime = EdgeAll.index(edgprime)
                            TurningMatMod[Indx][Indxprime] = (1 - TurningMatMod[Indx][Indx]) * (
                            TurningMat[Indx][Indxprime])

            # for jjj in range(0,4,1):
            for edge in EdgeAll:
                if edge in EdgeListWithBridges:
                    # if jjj<=1:
                    Indx = EdgeAll.index(edge)
                    for index11, row11 in selData.iterrows():
                        # for EdgeID in range(0, 2, 1):
                        if row11["Edge"] == edge and row11["Avail"] < 1.0:
                            rowToAdd = []
                            if TurningMatMod[Indx][Indx] == 0:
                                for iii in range(0, len(EdgeAll), 1):
                                    if iii == Indx:
                                        # IncomMatEdit[EdgeID][EdgeID] = \
                                        rowToAdd.append(1.0 - row11["Avail"])
                                    elif TurningMatMod[Indx][iii] > 0:
                                        # IncomMatEdit[EdgeID][iii] = \
                                        rowToAdd.append(TurningMatMod[Indx][iii] * row11["Avail"])
                                    else:
                                        rowToAdd.append(TurningMatMod[Indx][iii])
                            else:
                                for iii in range(0, len(EdgeAll), 1):
                                    if iii == Indx:
                                        # IncomMatEdit[EdgeID][EdgeID]=\
                                        rowToAdd.append(
                                            TurningMatMod[Indx][Indx] + (1.0 - TurningMatMod[Indx][Indx]) * (
                                                        1.0 - row11["Avail"]))

                                    elif TurningMatMod[Indx][iii] > 0:
                                        # IncomMatEdit[EdgeID][iii] =
                                        rowToAdd.append(TurningMatMod[Indx][iii] * row11["Avail"])
                                    else:
                                        rowToAdd.append(TurningMatMod[Indx][iii])

                            # print(len(EdgeAll),'jiiiiii',np.array(rowToAdd))
                            IncomMatEdit.append(np.array(rowToAdd).ravel())
                else:
                    Indx = EdgeAll.index(edge)
                    IncomMatEdit.append(np.array(TurningMatMod[Indx]).ravel())
                    # IncomMatEdit.append(np.array(IncomMat[edge]).ravel())

            # print(IncomMatEdit)
            ##Here the matrix is ready
            IncomMatEdit11 = np.asarray(IncomMatEdit)
            # IncomMatEditArray=np.concatenate(IncomMatEdit).ravel()#np.array(IncomMatEdit)#.reshape(4,4)
            # print(type(IncomMatEdit))
            # np.savetxt("IncomMat.csv",IncomMatEdit11,delimiter=',')
            IncomMatEdit = preprocessing.normalize(IncomMatEdit11, norm="l1")
            # print((IncomMatEdit))
            MC = MarkovChain(IncomMatEdit, verbose=True)
            #Here Mohsen eee. print(MC.K)
            # for index11, row11 in selData.iterrows():
            #    for EdgeID in range(0, 2, 1):
            #        if row11["Edge"] == EdgeID:
            term = np.sum(selData['TotalCost'].to_list())
            # term1=term-(np.sqrt(0.1*term)/(2*beta.rvs(aaa,bbb)))
            # term1 = term +(0.1 * term*2 * beta.rvs(aaa, bbb))
            # print('cost',term1)
            # print(selData)
            EdgewithBridgeHighMedinacost = []
            Mediancost = np.median(selData['TotalCost'].to_list())
            # TODO BigAssumptionMohsen
            for edge in EdgeAll:
                if edge in EdgeListWithBridges:
                    selDataCostEdge = pdAlg2InputSample.loc[
                        (pdAlg2InputSample["sample_run"] == sample_run) & (pdAlg2InputSample["Edge"] == edge)]
                    selDataCostEdgeVal = selDataCostEdge['TotalCost'].to_list()
                    if selDataCostEdgeVal > Mediancost or len(EdgeListWithBridges) <= 3:
                        EdgewithBridgeHighMedinacost.append(edge)

            TotalshortDist = 0
            for eedge in range(0, len(EdgeListWithBridgesShortDis), 1):
                # for brdg in range(0, 2, 1):
                # print(eedge,'heeer')
                br1edge = EdgeListWithBridgesShortDis[eedge][0]
                br2edge = EdgeListWithBridgesShortDis[eedge][1]
                if br1edge in EdgewithBridgeHighMedinacost and br2edge in EdgewithBridgeHighMedinacost:
                    TotalshortDist += EdgeListWithBridgesShortDis[eedge][2]

            # term1  exponential truncated with MincostAll and TotalshortDist/len(EdgeAll)
            # print(EdgewithBridgeHighMedinacost)
            # truncexpon has three parameters, b or shape paramter, then, loc/location paramater, and then a scale paramter
            # the support of the distribution is [x1=loc, x2=shape*scale + loc]
            # scale is mean, so, scale= len(EdgeAll) / (TotalshortDist);
            # thus, scale * shape =selDataCostEdgeValAveragedAcrossRunsMin
            #thus, shape=b= selDataCostEdgeValAveragedAcrossRunsMin * (TotalshortDist) / len(EdgeAll)
            #lowerexp, upperexp, scaleexp = 0, 1, len(EdgeAll) / (TotalshortDist)
            lowerexp = 0
            # term2 = truncexpon.rvs(b=(upperexp - lowerexp) / scaleexp, loc=lowerexp, scale=selDataCostEdgeValAveragedAcrossRunsMin)
            term2 = truncexpon.rvs(b=selDataCostEdgeValAveragedAcrossRunsMin * (TotalshortDist) / len(EdgeAll), loc=lowerexp, scale=len(EdgeAll) / (TotalshortDist))
            term1 = term - term2
            lst_dic1.append(
                {'subnetwork': subnetwork, 'sample_run': sample_run, 'TravelTime': MC.K, 'TotalCost': term1})
            #pdAlg2output = pdAlg2output.append(lst_dic1)
            pdAlg2output = pd.concat([pdAlg2output, pd.DataFrame.from_dict(lst_dic1, orient='columns')])

        return pdAlg2output
        # print(pdAlg2output)

