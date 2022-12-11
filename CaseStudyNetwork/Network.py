import os, sys, sklearn

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.0.3"
os.environ["PATH"] = r"C:\Program Files\R\R-4.0.3\bin\x64" + ";" + os.environ["PATH"]
from rpy2.robjects import numpy2ri, pandas2ri

numpy2ri.activate()
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
os.getcwd()
import copy
import networkx as nx
import scipy as sp
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from matplotlib import rc
# from Functions11 import Functions
from scipy.spatial import distance as dist
from random import seed
# from Network import *
from scipy.stats import beta
from itertools import product, combinations, combinations_with_replacement


###############CharacterizeSubnet
class Network:
    def __init__(self):
        pass

    def TwoSubnetworkShortdistanceEdgeswithBridges(self, subnetwork1, subnetwork2):
        EdgeListWithBridges1 = self.EdgesWithBridgesofSubnetwork(subnetwork1)
        EdgeListWithBridges2 = self.EdgesWithBridgesofSubnetwork(subnetwork2)
        sizeSubntwork1 = self.sizeofsbnetwork(subnetwork1)
        sizeSubntwork2 = self.sizeofsbnetwork(subnetwork2)

        EdgeListWithBridgesShortDis = [[13, 22, 5], [13, 24, 6], [13, 37, 4], [13, 49, 5],
                                       [40, 22, 4], [40, 24, 5], [40, 37, 3], [40, 49, 4],
                                       [50, 22, 5], [50, 24, 6], [50, 37, 4], [50, 49, 4],
                                       [57, 22, 6], [57, 24, 7], [57, 37, 5], [57, 49, 5],
                                       [43, 22, 4], [43, 24, 5], [43, 37, 3], [43, 49, 2],
                                       [58, 22, 5], [58, 24, 5], [58, 37, 6], [58, 49, 2],
                                       [63, 22, 8], [63, 24, 8], [63, 37, 6], [63, 49, 5],
                                       [13, 50, 1], [13, 57, 2], [40, 50, 2], [40, 57, 2],
                                       [13, 43, 3], [13, 58, 5], [13, 63, 3], [40, 43, 2], [40, 58, 4], [40, 63, 4],
                                       [50, 43, 3], [50, 58, 4], [50, 63, 2], [57, 43, 4], [57, 58, 4], [57, 63, 2]
                                       ]

        TotalshortDist = 0.0
        for eedge in range(0, len(EdgeListWithBridgesShortDis), 1):
            br1edge = EdgeListWithBridgesShortDis[eedge][0]
            br2edge = EdgeListWithBridgesShortDis[eedge][1]
            if br1edge in EdgeListWithBridges1 and br2edge in EdgeListWithBridges2:
                TotalshortDist += EdgeListWithBridgesShortDis[eedge][2]

        final_term1 = TotalshortDist / (sizeSubntwork1 + sizeSubntwork2)

        return final_term1

    def sizeofsbnetwork(sel, subnetwork):
        EdgeAll = []
        if subnetwork == 1:
            EdgeAll.extend([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 40, 67, 68])
        elif subnetwork == 2:
            EdgeAll.extend([6, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 37, 48, 49, 73, 74])
        elif subnetwork == 3:
            EdgeAll.extend([50, 57, 64, 66, 69])
        elif subnetwork == 4:
            EdgeAll.extend(
                [17, 25, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 58,
                 59, 60, 61, 62, 63, 65, 70, 71, 72])

        size = len(EdgeAll)

        return size

    def LeftOverRoadsTravelTime(self):
        TravelTimeGenerated = 0.0
        for leftoverroads_id in range(1, 9, 1):
            if leftoverroads_id == 1:
                TravelTime = [2, 3]
                Probs = [0.6, 0.4]
            elif leftoverroads_id == 2:
                TravelTime = [3, 5]
                Probs = [0.7, 0.3]
            elif leftoverroads_id == 3:
                TravelTime = [3, 6]
                Probs = [0.5, 0.5]
            elif leftoverroads_id == 4:
                TravelTime = [3, 4]
                Probs = [0.6, 0.4]
            elif leftoverroads_id == 5:
                TravelTime = [5, 8]
                Probs = [0.4, 0.6]
            elif leftoverroads_id == 6:
                TravelTime = [7, 9]
                Probs = [0.8, 0.2]
            elif leftoverroads_id == 7:
                TravelTime = [2, 4]
                Probs = [0.5, 0.5]
            else:
                TravelTime = [2, 3]
                Probs = [0.45, 0.55]

            TravelTimeGenerated += np.random.choice(TravelTime, 1, p=Probs)

        return TravelTimeGenerated[0]

    def EdgesWithBridgesofSubnetwork(self, subnetwork):
        EdgeListWithBridges = []
        if subnetwork == 1:
            EdgeListWithBridges.extend([13, 40])
        elif subnetwork == 2:
            EdgeListWithBridges.extend([22, 24, 37, 49])
        elif subnetwork == 3:
            EdgeListWithBridges.extend([50, 57])
        elif subnetwork == 4:
            EdgeListWithBridges.extend([43, 58, 63])

        return EdgeListWithBridges

    def CharacterizeSubnet(self, subnet):
        if subnet == 1:
            EdgeAll = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 40, 67, 68]
            #print(EdgeAll.index(8))
            EdgeListWithBridges = [13, 40]
            EdgeListWithBridgesShortDis = [[13, 40, 1]]
            EdgesWithTraffic = [67, 68]

            EdgesWithTrafficProb = [0.6, 0.7]
            EdgesWithTrafficOtherTime = [3, 5]
            BridgeRoadMat = [[1, -1, -1], [0, 0, 0]]  # [[0,0,-1],[1,-1,-1],[0,-1,-1],[1,-1,-1]]
            EdgeNumWithBridges = len(BridgeRoadMat)
            #print('numbridge', EdgeNumWithBridges)
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
        if subnet == 2:
            # TurningMatMod=TurningMat
            EdgeAll = [6, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 37, 48, 49, 73, 74]
            EdgeListWithBridges = [22, 24, 37, 49]
            EdgeListWithBridgesShortDis = [[22, 24, 1], [22, 37, 3], [24, 37, 4], [22, 49, 5],
                                           [24, 49, 4], [37, 49, 8]]
            EdgesWithTraffic = [73, 74]
            EdgesWithTrafficProb = [0.5, 0.45]
            EdgesWithTrafficOtherTime = [4, 3]
            BridgeRoadMat = [[0, -1, -1], [1, 1, -1], [0, -1, -1],
                             [0, 0, -1]]  # [[0,0,-1],[1,-1,-1],[0,-1,-1],[1,-1,-1]]

            EdgeNumWithBridges = len(BridgeRoadMat)
            TurningMat = [[0, 0.49, 0.51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0.24, 0, 0.26, 0, 0, 0.2, 0, 0.17, 0.13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0.25, 0.25, 0, 0, 0, 0, 0, 0.14, 0, 0.2, 0.16, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0.3, 0.35, 0.35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0.15, 0, 0.19, 0, 0, 0.22, 0.25, 0.19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0.19, 0, 0.22, 0, 0, 0.21, 0, 0, 0, 0, 0, 0, 0.38, 0, 0, 0],
                          [0, 0.11, 0.11, 0, 0, 0.23, 0, 0, 0.15, 0.22, 0.18, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0.11, 0, 0, 0, 0.21, 0.23, 0.18, 0, 0, 0, 0, 0, 0, 0, 0.27, 0, 0, 0],
                          [0, 0, 0.15, 0, 0, 0, 0, 0.23, 0, 0, 0.25, 0.37, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0.15, 0, 0, 0, 0, 0.25, 0, 0.34, 0, 0, 0, 0, 0, 0.26, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.68, 0, 0, 0.32, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27, 0, 0.46, 0.27, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.57, 0, 0.43, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.22, 0.28, 0, 0, 0.32, 0, 0.18],
                          [0, 0, 0, 0, 0, 0, 0.47, 0, 0.3, 0, 0.23, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.37, 0, 0, 0.25, 0.38],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.36, 0, 0.64, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.4, 0, 0, 0, 0, 0, 0]
                          ]
            TurningMatMod = [[0, 0.49, 0.51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0.24, 0, 0.26, 0, 0, 0.2, 0, 0.17, 0.13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0.25, 0.25, 0, 0, 0, 0, 0, 0.14, 0, 0.2, 0.16, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0.3, 0.35, 0.35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0.15, 0, 0.19, 0, 0, 0.22, 0.25, 0.19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0.19, 0, 0.22, 0, 0, 0.21, 0, 0, 0, 0, 0, 0, 0.38, 0, 0, 0],
                             [0, 0.11, 0.11, 0, 0, 0.23, 0, 0, 0.15, 0.22, 0.18, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0.11, 0, 0, 0, 0.21, 0.23, 0.18, 0, 0, 0, 0, 0, 0, 0, 0.27, 0, 0, 0],
                             [0, 0, 0.15, 0, 0, 0, 0, 0.23, 0, 0, 0.25, 0.37, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0.15, 0, 0, 0, 0, 0.25, 0, 0.34, 0, 0, 0, 0, 0, 0.26, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.68, 0, 0, 0.32, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27, 0, 0.46, 0.27, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.57, 0, 0.43, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.22, 0.28, 0, 0, 0.32, 0, 0.18],
                             [0, 0, 0, 0, 0, 0, 0.47, 0, 0.3, 0, 0.23, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.37, 0, 0, 0.25, 0.38],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.36, 0, 0.64, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.4, 0, 0, 0, 0, 0, 0]
                             ]
            TravelingTimeEdge = [12, 3, 4, 4, 3, 4, 7, 6, 18, 11, 7, 4, 9, 4, 3, 9, 8, 2, 2]
        if subnet == 3:
            # TurningMatMod=TurningMat
            EdgeAll = [50, 57, 64, 66, 69]
            EdgeListWithBridges = [50, 57]
            EdgeListWithBridgesShortDis = [[50, 57, 1]]
            EdgesWithTraffic = [69]
            EdgesWithTrafficProb = [0.5]
            EdgesWithTrafficOtherTime = [6]
            BridgeRoadMat = [[0, -1, -1], [0, 0, 0]]  # [[0,0,-1],[1,-1,-1],[0,-1,-1],[1,-1,-1]]
            EdgeNumWithBridges = len(BridgeRoadMat)
            TurningMat = [[0, 0.24, 0, 0, 0.76],
                          [0.5, 0, 0.5, 0, 0],
                          [0, 0.25, 0, 0.75, 0],
                          [0, 0, 1, 0, 0],
                          [1, 0, 0, 0, 0]
                          ]
            TurningMatMod = [[0, 0.24, 0, 0, 0.76],
                             [0.5, 0, 0.5, 0, 0],
                             [0, 0.25, 0, 0.75, 0],
                             [0, 0, 1, 0, 0],
                             [1, 0, 0, 0, 0]
                             ]
            TravelingTimeEdge = [7, 8, 6, 6, 3]
        if subnet == 4:
            EdgeAll = [17, 25, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55,
                       56, 58, 59, 60, 61, 62, 63, 65, 70, 71, 72]
            EdgeListWithBridges = [43, 58, 63]
            EdgeListWithBridgesShortDis = [[43, 58, 3], [43, 63, 6], [58, 63, 3]]
            EdgesWithTraffic = [70, 71, 72]
            EdgesWithTrafficProb = [0.6, 0.4, 0.8]
            EdgesWithTrafficOtherTime = [4, 8, 9]
            BridgeRoadMat = [[1, 1, 1], [1, 1, -1], [0, -1, -1]]  # [[0,0,-1],[1,-1,-1],[0,-1,-1],[1,-1,-1]]
            EdgeNumWithBridges = len(BridgeRoadMat)
            # till here
            # pd.set_option('mode.use_inf_as_na',True)
            TurningMatPandas = pd.read_csv('Subnetwork4-turningprob.csv',
                                           header=None)  # read_excel('Subnetwork4-turningprob.xlsx')
            # TurningMatPandas=TurningMatPandas.mask(np.isinf(TurningMatPandas))#.replace(np.nan,0)#fillna(0)
            # TurningMatPandas.replace(np.nan,0)
            TurningMatPandas.to_csv('Subnetwork4-turningprob1.csv')
            TurningMat = [
                [0, 0, 0, 0, 0, 0, 0, 0.28, 0, 0.36, 0.36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0],
                [0, 0, 0.49, 0.51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0],
                [0, 0.16, 0, 0.23, 0.39, 0.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0],
                [0, 0.28, 0.38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0],
                [0, 0, 0.32, 0, 0, 0.18, 0.20, 0.18, 0.12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0.25, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0.18, 0, 0.14, 0.18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.3, 0, 0, 0.28, 0.16, 0.26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0],
                [0.136125654450262, 0, 0, 0, 0.194107452339688, 0, 0.19930675909878698, 0, 0.106585788561525,
                 0.18193717277486898, 0.18193717277486898, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0],
                [0, 0, 0, 0, 0.16892911010558098, 0, 0.173453996983409, 0.15761689291101097, 0, 0, 0,
                 0.18451612903225798, 0, 0.15419354838709698, 0.161290322580645, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0],
                [0.22780131035358603, 0, 0, 0, 0, 0, 0.23883696780893, 0.22889650896105498, 0, 0, 0.30446521287642797,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.178060737250907, 0, 0, 0, 0, 0, 0, 0.17891679848769002, 0, 0.23798502382573197, 0,
                 0.19469026548672602, 0.210347174948945, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0.119785973577318, 0, 0.191328286304198, 0, 0.212663454920853,
                 0.232754859227472, 0.243467425970159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.197583511016347, 0.20326936744847196, 0, 0, 0, 0, 0, 0, 0, 0,
                 0.17729056172243104, 0, 0, 0.21448992065526298, 0.20736663915748701, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0.112431444241316, 0, 0, 0.093323216995448, 0, 0, 0.21699544764795103, 0, 0,
                 0.18968133535660103, 0.17550274223034698, 0.21206581352833603, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0.094907407407407, 0, 0, 0.220679012345679, 0, 0.18441358024691396, 0,
                 0.15508885298869102, 0, 0.17770597738287602, 0.16720516962843301, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0.10353535353535401, 0, 0, 0, 0, 0, 0, 0, 0.20117845117845104, 0.18463810930576102, 0,
                 0.195286195286195, 0.162481536189069, 0.15288035450517, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0],
                [0, 0, 0, 0.24468085106383, 0, 0.16769721176741698, 0, 0, 0, 0, 0, 0, 0, 0.32585067977571197, 0,
                 0.261771257393041, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.212580221150017, 0.16326160984321303, 0, 0,
                 0.176016423112214, 0.124675885911841, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.323465859982714],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.18882175226586104, 0.145015105740181, 0, 0.166163141993958,
                 0, 0, 0.170212765957447, 0.16337386018237102, 0.166413373860182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.10105649977032598, 0, 0, 0, 0.233563936079651, 0,
                 0, 0, 0.32155779106779897, 0, 0, 0, 0, 0, 0, 0, 0, 0.343821773082223],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.18349168646080802, 0, 0, 0, 0, 0, 0.16146645865834602, 0, 0,
                 0.16770670826833098, 0.170826833073323, 0.160926365795724, 0.15558194774346804, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.159230769230769, 0.246792130025663,
                 0.172307692307692, 0, 0.168461538461538, 0, 0, 0.253207869974337, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.160216718266254, 0, 0.17337461300309603,
                 0.16640866873065, 0, 0, 0, 0, 0.203453453453453, 0.153903903903904, 0.142642642642643, 0, 0, 0, 0, 0,
                 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.24342406018842605, 0, 0, 0, 0, 0, 0, 0, 0.176462749133358, 0, 0,
                 0, 0.206398394075623, 0, 0, 0, 0, 0, 0, 0, 0.373714796602593, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.192164179104478, 0, 0, 0, 0, 0, 0, 0, 0.13930348258706501, 0, 0,
                 0.16853233830845799, 0, 0, 0, 0.221143473570658, 0, 0.278856526429342, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.286494538232373, 0, 0.213505461767627, 0, 0,
                 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.17833876221498396, 0, 0, 0.5, 0,
                 0.166938110749186, 0.154723127035831, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.161029411764706, 0,
                 0.251681075888569, 0, 0.199264705882353, 0, 0.139705882352941, 0.24831892411143103, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.157553956834532, 0, 0, 0,
                 0.194964028776978, 0.147482014388489, 0, 0, 0.127470355731225, 0.07559288537549401, 0,
                 0.29693675889328097, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.40318140104007294, 0, 0,
                 0.315466363409218, 0, 0, 0.281352235550709, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.144595149735977,
                 0.281590413943355, 0, 0.11643714689265501, 0, 0.457377289428012, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.181124880838894,
                 0, 0.24594852240228798, 0, 0, 0.5729265967588181, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0.31613976705490804, 0, 0.42928452579034904, 0.254575707154742, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.43264503441494606, 0, 0.5673549655850539, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]

            # TurningMat=TurningMatPandas.values.tolist()
            TurningMatMod = TurningMat
            TravelingTimeEdge = [15, 4, 8, 2, 4, 3, 6, 7, 2, 2, 3, 4, 7, 1, 3, 1, 7, 9, 6, 2, 4, 4, 2, 5, 3, 2, 4, 4, 4,
                                 1, 4, 4, 3, 5, 7]

        #print(type(TurningMatMod))

        # print(TurningMatMod)
        # print(BridgeRoadMat[1]) #print(BridgeRoadMat[1][1]) #print(len(BridgeRoadMat))
        return EdgeAll, EdgeListWithBridges, EdgeListWithBridgesShortDis, EdgesWithTraffic, EdgesWithTrafficProb, EdgesWithTrafficOtherTime, BridgeRoadMat, TurningMat, TurningMatMod, TravelingTimeEdge, EdgeNumWithBridges
