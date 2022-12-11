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
import more_itertools
from scipy.interpolate import interp1d
from CaseStudyNetwork.Network import *
from scipy.stats import beta
from itertools import product, combinations, combinations_with_replacement


class SystemAll():
    def __init__(self):
        pass

    ##########Sampling4InputAlg3 Function
    def simulate_algorithm3_input_generation(self, resultsPath, pdAlg2SubnetLevel, pdAlg3InputSample, MeasureList, Subnet, OverallSample,
                                             PlotDist, n_sample):
        for sample_run in range(OverallSample):
            lst_dic1 = []
            for MeasureId in range(0, 2, 1):
                # MeasureValue = []
                Sample = [0]
                selData = pdAlg2SubnetLevel.loc[(pdAlg2SubnetLevel["subnetwork"] == Subnet)]
                Sample.append(list(selData[MeasureList[MeasureId]].sample(n=n_sample, replace=True, random_state=1)))
                Sample.append(1.0)
                Sample = list(more_itertools.collapse(Sample))
                Sample_sorted = np.sort(Sample)
                CDFProbval = [0]
                for ii11 in range(1, n_sample + 1, 1):
                    val = list(Sample_sorted).index(Sample_sorted[ii11 - 1]) + 1.0
                    CDFProbval.append(val / (n_sample + 1.0))
                CDFProbval.append(1.0)
                # print(len(Sample_sorted),len(CDFProbval))
                CDFfunction = interp1d(Sample_sorted, CDFProbval)
                CDFfunctionInv = interp1d(CDFProbval, Sample_sorted)
                # Sample_sorted[np.argwhere()]
                # print(Measure)#,CDFfunctionInv(CDFfunction(0.3)))
                # eeee=(np.random.uniform(0,1,100))
                # print(CDFfunctionInv(np.random.uniform(0,1)))
                if MeasureId == 0:
                    MeasureValue0 = []
                    MeasureValue0.append(CDFfunctionInv(np.random.uniform(0, 1)))
                else:
                    MeasureValue1 = []
                    MeasureValue1.append(CDFfunctionInv(np.random.uniform(0, 1)))
                    # MeasureValue.append(MeasureValueOne[0])
                    lst_dic1.append({'subnetwork': Subnet, 'sample_run': sample_run,
                                     'TravelTime': float(MeasureValue0[0]), 'TotalCost': float(MeasureValue1[0])})

            # pdAlg3InputSample = pdAlg3InputSample.append(lst_dic1)
            pdAlg3InputSample = pd.concat([pdAlg3InputSample, pd.DataFrame(lst_dic1)])
        # PlotDist=0
        if PlotDist == 1:
            for jjj in range(0, 2, 1):
                Measure = MeasureList[jjj]
                fig, ax = plt.subplots()
                ax2 = ax.twinx()
                ax.hist(pdAlg3InputSample[Measure], bins=100, density=True)
                ax2.hist(pdAlg3InputSample[Measure], cumulative=1, histtype='step', bins=100, color='tab:orange',
                         density=True)
                ax.set_xlim((ax.get_xlim()[0], pdAlg3InputSample[Measure].max()))
                File1 = os.path.abspath(resultsPath + "-Alg2Sample-" + Measure + ".png")
                plt.savefig(File1, format="PNG", dpi=800)

        return pdAlg3InputSample

    ##########SimulateCostAlg3 Function
    # Date: 20th November 2022,edit the function below, add loop for runs, edits...
    def simulate_algorithm3_output_generation(self, network_instance, pdAlg3InputSampleConCat, pdAlg3OutputConCat, OverallSample):
        selDataCostsubnetValAveragedAcrossRuns = []
        for sample_run in range(OverallSample):
            for Subnet in range(1, 5, 1):
                selDataCostEdge = pdAlg3InputSampleConCat.loc[(pdAlg3InputSampleConCat["subnetwork"] == Subnet)]
                selDataCostEdgeVal = selDataCostEdge['TotalCost'].to_list()
                selDataCostsubnetValAveragedAcrossRuns.append(sum(selDataCostEdgeVal) / len(selDataCostEdgeVal))

        selDataCostEdgeValAveragedAcrossRunsMin = min(selDataCostsubnetValAveragedAcrossRuns)
        for sample_run in range(OverallSample):  # OverallSample
            lst_dic1 = []
            selData = pdAlg3InputSampleConCat.loc[(pdAlg3InputSampleConCat["sample_run"] == sample_run)]

            # TotalCost
            term = np.sum(selData['TotalCost'].to_list())
            Mediancost = np.median(selData['TotalCost'].to_list())
            SubnetworkHighMedinacost = []
            for Subnet in range(1, 5, 1):
                selDataCostSubnet = pdAlg3InputSampleConCat.loc[
                    (pdAlg3InputSampleConCat["sample_run"] == sample_run) & (
                                pdAlg3InputSampleConCat["subnetwork"] == Subnet)]
                selDataCostSubnetVal = selDataCostSubnet['TotalCost'].to_list()
                if selDataCostSubnetVal > Mediancost:
                    SubnetworkHighMedinacost.append(Subnet)
            final_term = 0.0
            for Subnet1 in SubnetworkHighMedinacost:
                for Subnet2 in SubnetworkHighMedinacost:
                    if Subnet1 != Subnet2:
                        final_term1 = network_instance.TwoSubnetworkShortdistanceEdgeswithBridges(Subnet1, Subnet2)
                        final_term += final_term1
            lowerexp = 0
            term2 = truncexpon.rvs(b=selDataCostEdgeValAveragedAcrossRunsMin * final_term, loc=lowerexp,
                                   scale=1 / final_term)
            term1 = term - term2

            # TravelTime
            TotalTravelTimeLeftOver = network_instance.LeftOverRoadsTravelTime()
            term = np.sum(selData['TravelTime'].to_list())
            TotalTravelTime = term + TotalTravelTimeLeftOver - 4 + 1

            lst_dic1.append({'sample_run': sample_run, 'TravelTime': TotalTravelTime, 'TotalCost': term1})
            #pdAlg3OutputConCat = pdAlg3OutputConCat.append(lst_dic1)
            pdAlg3OutputConCat = pd.concat([pdAlg3OutputConCat, pd.DataFrame(lst_dic1)])

        return pdAlg3OutputConCat
