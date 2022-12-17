import os,sys,sklearn
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.0.3"
os.environ["PATH"]   = r"C:\Program Files\R\R-4.0.3\bin\x64" + ";" + os.environ["PATH"]
from scipy.stats import truncexpon
from rpy2.robjects import numpy2ri, pandas2ri
import copy
from numpy import random
numpy2ri.activate()
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
os.getcwd()
from MarkovChain.Markov_chain_new import MarkovChain
from CaseStudyNetwork.Network import *
from sklearn import preprocessing

class Subnetworks():
    def __init__(self):
        pass

    ############### Plot results
    def plot_subnet_results(self, pdAlg2output, MeasureList, resultsPath, subnet):
            for jjj in range(0, 2, 1):
                Measure = MeasureList[jjj]
                fig, ax = plt.subplots()
                ax2 = ax.twinx()
                ax.hist(pdAlg2output[Measure], bins=100, density=True)
                ax2.hist(pdAlg2output[Measure], cumulative=1, histtype='step', bins=100, color='tab:orange',
                         density=True)
                ax.set_xlim((ax.get_xlim()[0], pdAlg2output[Measure].max()))
                File1 = os.path.abspath(resultsPath + "-Alg2Sample-" + str(subnet) + "subnet" + Measure + ".png")
                plt.savefig(File1, format="PNG", dpi=800)


    ############### SimulateAlg2Output function
    def simulate_algorithm2_output_generation(self, subnetwork, EdgeAll, EdgeListWithBridges, EdgeListWithBridgesShortDis,
                                              EdgesWithTraffic, EdgesWithTrafficProb, EdgesWithTrafficOtherTime, BridgeRoadMat,
                                              TurningMat, TurningMatMod, TravelingTimeEdge, EdgeNumWithBridges,
                                              pdAlg2output, pdAlg2InputSample, OverallSample):
        TurningMatMod11 = copy.deepcopy(TurningMatMod)
        MincostAll = 0
        for edge in EdgeAll:
            if edge in EdgeListWithBridges:
                selDataMinCost = pdAlg2InputSample.loc[(pdAlg2InputSample[
                                                            'Edge'] == edge)]
                term = np.sum(selDataMinCost['TotalCost'].to_list())
                MincostAll = min(term / OverallSample, MincostAll)

        selDataCostEdgeValAveragedAcrossRuns = []
        for edge in EdgeAll:
            if edge in EdgeListWithBridges:
                selDataCostEdge = pdAlg2InputSample.loc[(pdAlg2InputSample["Edge"] == edge)]
                selDataCostEdgeVal = selDataCostEdge['TotalCost'].to_list()
                #selDataCostEdgeVal = [x for x in selDataCostEdgeVal00 if x is not np.NaN]
                #selDataCostEdgeVal.remove(np.nan)
                #print('hop:', type(selDataCostEdgeVal), type(selDataCostEdgeVal)) #, sum(selDataCostEdgeVal)/len(selDataCostEdgeVal))
                selDataCostEdgeValAveragedAcrossRuns.append(np.mean(selDataCostEdgeVal))
                #sum(selDataCostEdgeVal)/len(selDataCostEdgeVal))

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

            lst_dic1 = []
            selData = pdAlg2InputSample.loc[(pdAlg2InputSample["sample_run"] == sample_run)]

            IncomMatEdit = []
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

            for edge in EdgeAll:
                if edge in EdgeListWithBridges:
                    Indx = EdgeAll.index(edge)
                    for index11, row11 in selData.iterrows():
                        # print(row11["Avail"], type(float(row11["Avail"])))
                        # for EdgeID in range(0, 2, 1):
                        if row11["Edge"] == edge and float(row11["Avail"]) < 1.0:
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


            IncomMatEdit11 = np.asarray(IncomMatEdit)

            IncomMatEdit = preprocessing.normalize(IncomMatEdit11, norm="l1")
            MC = MarkovChain(IncomMatEdit, verbose=True)
            #Here Mohsen eee. print(MC.K)
            term = np.sum(selData['TotalCost'].to_list())
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
                br1edge = EdgeListWithBridgesShortDis[eedge][0]
                br2edge = EdgeListWithBridgesShortDis[eedge][1]
                if br1edge in EdgewithBridgeHighMedinacost and br2edge in EdgewithBridgeHighMedinacost:
                    TotalshortDist += EdgeListWithBridgesShortDis[eedge][2]

            lowerexp = 0
            if TotalshortDist > 0:
                term2 = truncexpon.rvs(b=selDataCostEdgeValAveragedAcrossRunsMin * TotalshortDist / len(EdgeAll), loc=lowerexp, scale=len(EdgeAll) / TotalshortDist)
            else:
                term2 = 0
                print('ShorDist is zero!')
            term1 = term - term2
            lst_dic1.append(
                {'subnetwork': subnetwork, 'sample_run': sample_run, 'TravelTime': MC.K, 'TotalCost': term1})
            #pdAlg2output = pdAlg2output.append(lst_dic1)
            pdAlg2output = pd.concat([pdAlg2output, pd.DataFrame.from_dict(lst_dic1, orient='columns')])

        return pdAlg2output

