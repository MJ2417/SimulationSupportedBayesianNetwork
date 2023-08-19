##Bridges, California State Highway System, 2015
##https://purl.stanford.edu/td948by1459
##Traffic Volume, California, 2014
##https://searchworks.stanford.edu/view/fx041qj6799
##

# os.environ["PATH"] += os.pathsep+ r"C:\Program Files\Graphviz\bin"

import os, sys
import warnings

# warnings.filterwarnings('ignore')

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.1"
os.environ["PATH"] = r"C:\Program Files\R\R-4.3.1\bin\x64" + ";" + os.environ["PATH"]
from rpy2.robjects import numpy2ri, pandas2ri
from datetime import datetime

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
from CaseStudyNetwork.Input_parameters import *

network_instance = Network()
roads_instance = Roads()
subnetwork_instance = Subnetworks()
system_instance = SystemAll()
discretization_instance = Discretization()
conditional_tables_instance = ConditionalTables()

for Subnet in range(1, 5, 1):
    '''
    define dataframes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    df_algorithm1_bridge_level = pd.DataFrame(
        columns=['subnet', 'sim_run', 'NumBrdg', 'ListMaintenanceRates', 'ListMaintenanceRatesCoded', 'Edge', 'Avail',
                 'TotalCost', 'bridge1Rate', 'bridge2Rate', 'bridge3Rate'])
    df_algorithm2_input_sample = pd.DataFrame(columns=['subnet', 'sample_run', 'Edge', 'Avail', 'TotalCost'])
    df_algorithm2_output = pd.DataFrame(columns=['subnetwork', 'sample_run', 'TravelTime', 'TotalCost'])
    df_algorithm3_input_sample = pd.DataFrame(columns=['subnetwork', 'sample_run', 'TravelTime', 'TotalCost'])
    conditional_prob_table_road_level = pd.DataFrame(
        columns=['DepenVar', 'DepenVarLvel', 'IndepenVar', 'IndepenVarLvel', 'CondProb'])
    conditional_prob_table_road_to_subnet_level = pd.DataFrame(
        columns=['DepenVar', 'DepenVarLvel', 'IndepenVar', 'IndepenVarLvel',
                 'IndepenVar1', 'IndepenVar1Lvel', 'IndepenVar2', 'IndepenVar2Lvel',
                 'IndepenVar3', 'IndepenVar3Lvel', 'CondProb'])
    if Subnet == 1:
        df_algorithm1_bridge_level_concatenated = pd.DataFrame({}, columns=df_algorithm1_bridge_level.columns.values,
                                                               index=None)
        df_algorithm2_input_sample_concatenated = pd.DataFrame({}, columns=df_algorithm2_input_sample.columns.values,
                                                               index=None)
        df_algorithm2_output_concatenated = pd.DataFrame({}, columns=df_algorithm2_output.columns.values, index=None)
        df_algorithm2_output_for_discretization_concat = pd.DataFrame({},
                                                                      columns=df_algorithm2_output.columns.values,
                                                                      index=None)
        df_algorithm3_input_sample_concatenated = pd.DataFrame({}, columns=df_algorithm3_input_sample.columns.values,
                                                               index=None)
        df_algorithm3_input_sample_concatenated_discretized = pd.DataFrame(
            columns=['subnetwork', 'sample_run', 'TravelTime', 'TotalCost', 'TravelTimeDisc11', 'TotalCostDisc11'])
        conditional_prob_table_road_level_concate = pd.DataFrame({},
                                                                 columns=conditional_prob_table_road_level.columns.values,
                                                                 index=None)
        columns = ['Subnet']
        columns.extend(conditional_prob_table_road_to_subnet_level.columns.values)
        conditional_prob_table_road_to_subnet_level_concate = pd.DataFrame({}, columns=columns, index=None)

    '''
    define dataframes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    print('Progressing with Algorithm 1&2 of subnetwork', Subnet, datetime.now())
    [EdgeAll, edge_with_bridges_list, EdgeListWithBridgesShortDis, EdgesWithTraffic, EdgesWithTrafficProb,
     EdgesWithTrafficOtherTime, BridgeRoadMat, TurningMat, TurningMatMod, TravelingTimeEdge,
     EdgeNumWithBridges] = network_instance.CharacterizeSubnet(Subnet)

    '''
    Algorithms 1 & 2 
    '''
    df_algorithm1_bridge_level = roads_instance.edge_simulation_july_2022(df_algorithm1_bridge_level, BridgeRoadMat,
                                                                          edge_with_bridges_list,
                                                                          Subnet, Q10, Q11, Q20, Q21, RM, NumSimRun,
                                                                          Simulatingtime, warm_up, RehabFullCost,
                                                                          MaintenanceCost,
                                                                          AvailCoeff)
    df_algorithm2_input_sample = roads_instance.sampling_inputs_algorithm2(resultsPath, df_algorithm1_bridge_level,
                                                                           df_algorithm2_input_sample,
                                                                           measure_list, edge_with_bridges_list,
                                                                           EdgeNumWithBridges, Subnet,
                                                                           OverallSample, PlotDist, n_sample)

    # Inputs of third algorithm
    df_algorithm2_output = subnetwork_instance.simulate_algorithm2_output_generation(Subnet, EdgeAll,
                                                                                     edge_with_bridges_list,
                                                                                     EdgeListWithBridgesShortDis,
                                                                                     EdgesWithTraffic,
                                                                                     EdgesWithTrafficProb,
                                                                                     EdgesWithTrafficOtherTime,
                                                                                     BridgeRoadMat,
                                                                                     TurningMat, TurningMatMod,
                                                                                     TravelingTimeEdge,
                                                                                     EdgeNumWithBridges,
                                                                                     df_algorithm2_output,
                                                                                     df_algorithm2_input_sample,
                                                                                     OverallSample)

    '''
    Discretizations 1 & 2 
    '''
    indexLoop = 0
    np.set_printoptions(threshold=sys.maxsize)
    df_algorithm2_output_for_discretization = df_algorithm2_output
    for edge_with_bridge_instance in edge_with_bridges_list:
        df_algorithm1_bridge_level_for_discretization = df_algorithm1_bridge_level[
            (df_algorithm1_bridge_level['Edge'] ==
             edge_with_bridge_instance)]
        df_algorithm2_input_sample_for_discretization = df_algorithm2_input_sample[
            (df_algorithm2_input_sample['Edge'] ==
             edge_with_bridge_instance)]  # layer 1-input

        if indexLoop == 0 and Subnet == 1:
            df_algorithm1_bridge_level_for_discretization_concatenated = pd.DataFrame()
            df_algorithm2_input_sample_for_discretization_concate = pd.DataFrame()
            df_algorithm1_bridge_level_for_discretization_concatenated = pd.DataFrame({},
                                                                                      columns=df_algorithm1_bridge_level_for_discretization.columns.values,
                                                                                      index=None)  # copy.deepcopy(DataFrameForDisLay2out)
            df_algorithm2_input_sample_for_discretization_concate = pd.DataFrame({},
                                                                                 columns=df_algorithm2_input_sample_for_discretization.columns.values,
                                                                                 index=None)  # copy.deepcopy(DataFrameForDisLay1Input)
        indexLoop += 1

        for MeasureId in range(0, 2, 1):
            Measure = measure_list[MeasureId]
            [df_algorithm1_bridge_level_for_discretization, DiscLay2outRoad12,
             binsLay2out] = discretization_instance.discretize_data_frame(
                df_algorithm1_bridge_level_for_discretization, Measure, n_discretization)
            df_algorithm2_input_sample_for_discretization[Measure] = df_algorithm2_input_sample_for_discretization[
                Measure].astype(float)
            [df_algorithm2_input_sample_for_discretization,
             df_algorithm2_bins] = discretization_instance.discretize_data_frame_with_bins(
                df_algorithm2_input_sample_for_discretization, Measure, binsLay2out)
        df_algorithm1_bridge_level_for_discretization_concatenated = pd.concat(
            [df_algorithm1_bridge_level_for_discretization_concatenated,
             df_algorithm1_bridge_level_for_discretization])
        df_algorithm2_input_sample_for_discretization_concate = pd.concat(
            [df_algorithm2_input_sample_for_discretization_concate, df_algorithm2_input_sample_for_discretization])

    df_algorithm1_bridge_level_concatenated = pd.concat([df_algorithm1_bridge_level_concatenated,
                                                         df_algorithm1_bridge_level])
    df_algorithm2_input_sample_concatenated = pd.concat([df_algorithm2_input_sample_concatenated,
                                                         df_algorithm2_input_sample])
    df_algorithm2_output_concatenated = pd.concat([df_algorithm2_output_concatenated, df_algorithm2_output])

    '''
    Algorithm 3-input sampling & discretization
    '''
    df_algorithm3_input_sample = system_instance.simulate_algorithm3_input_generation(resultsPath,
                                                                                      df_algorithm2_output_concatenated,
                                                                                      df_algorithm3_input_sample,
                                                                                      MeasureList22, Subnet,
                                                                                      OverallSample, PlotDist,
                                                                                      n_sample)

    for MeasureId in range(0, 2, 1):
        Measure22 = MeasureList22[MeasureId]
        [df_algorithm2_output_for_discretization, DiscLay0out,
         binsLay0out] = discretization_instance.discretize_data_frame_algorithm2_and_3(
            df_algorithm2_output_for_discretization, Measure22, n_discretization)

        # Here Mohsen
        [df_algorithm3_input_sample, DiscLay3In] = discretization_instance. \
            discretize_data_frame_with_bins(
            df_algorithm3_input_sample, Measure22, binsLay0out)

    df_algorithm3_input_sample_concatenated = pd.concat([df_algorithm3_input_sample_concatenated,
                                                         df_algorithm3_input_sample])
    df_algorithm3_input_sample_concatenated_discretized = pd.concat(
        [df_algorithm3_input_sample_concatenated_discretized,
         df_algorithm3_input_sample])
    df_algorithm2_output_for_discretization_concat = pd.concat(
        [df_algorithm2_output_for_discretization_concat, df_algorithm2_output_for_discretization])

    '''
    Conditional probability tables -- algorithms 1&2
    '''
    conditional_prob_table_road_level = conditional_tables_instance.ConditionalProbTabLveles21GenerateJuly2022(
        df_algorithm1_bridge_level_for_discretization_concatenated, conditional_prob_table_road_level,
        EdgeNumWithBridges, edge_with_bridges_list, OverallSample)
    # here#DataFrameForDisLay2outConCat.to_csv(os.path.join(resultsPath+"DataFrameForDisLay2outConCat"+".csv"))
    conditional_prob_table_road_level_concate = pd.concat(
        [conditional_prob_table_road_level_concate, conditional_prob_table_road_level])
    conditional_prob_table_road_to_subnet_level = conditional_tables_instance.ConditionalProbTabLveles10GenerateJuly2022(
        conditional_prob_table_road_to_subnet_level, df_algorithm2_input_sample_for_discretization_concate,
        df_algorithm2_output_for_discretization, conditional_prob_table_road_level_concate,
        n_discretization,
        OverallSample,
        EdgeNumWithBridges, edge_with_bridges_list, Subnet)

    conditional_prob_table_road_to_subnet_level.insert(0, 'Subnet', Subnet)

    conditional_prob_table_road_to_subnet_level_concate = pd.concat(
        [conditional_prob_table_road_to_subnet_level_concate, conditional_prob_table_road_to_subnet_level])

    subnetwork_instance.plot_subnet_results(df_algorithm2_output, MeasureList22, resultsPath, Subnet)
    if Subnet == 4:
        '''
        write results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        '''
        df_algorithm2_output_concatenated.to_csv(os.path.join(resultsPath + "df_algorithm2_output_concatenated.csv"))
        df_algorithm2_input_sample_concatenated.to_csv(os.path.join(resultsPath +
                                                                    "df_algorithm2_input_sample_concatenated.csv"))
        df_algorithm1_bridge_level_concatenated.to_csv(os.path.join(resultsPath +
                                                                    "df_algorithm1_bridge_level_concatenated.csv"))
        df_algorithm1_bridge_level_for_discretization_concatenated.to_csv(
            os.path.join(resultsPath + "df_algorithm1_bridge_level_for_discretization_concatenated.csv"))

        df_algorithm1_bridge_level_for_discretization_concatenated_grouped0 = \
            df_algorithm1_bridge_level_for_discretization_concatenated.groupby(
                [ "ListMaintenanceRatesCoded",
                 "Edge", "AvailDisc", "AvailDisc11"], group_keys=False).size().to_frame()

        df_algorithm1_bridge_level_for_discretization_concatenated_grouped1 = \
            df_algorithm1_bridge_level_for_discretization_concatenated.groupby(
                ["ListMaintenanceRatesCoded",
                 "Edge", "TotalCostDisc", "TotalCostDisc11"], group_keys=False).size().to_frame()
        df_algorithm1_bridge_level_for_discretization_concatenated_grouped0.describe()
        df_algorithm1_bridge_level_for_discretization_concatenated_grouped0.to_csv(
            os.path.join(resultsPath + "df_algorithm1_bridge_level_for_discretization_concatenated_grouped0.csv"))
        df_algorithm1_bridge_level_for_discretization_concatenated_grouped1.to_csv(
            os.path.join(resultsPath + "df_algorithm1_bridge_level_for_discretization_concatenated_grouped1.csv"))

        df_algorithm2_output_for_discretization_concat.to_csv(
            os.path.join(resultsPath + "df_algorithm2_output_for_discretization_concat.csv"))
        df_algorithm2_output_for_discretization_concat_group0 = \
            df_algorithm2_output_for_discretization_concat.groupby(
                [ "subnetwork", "TravelTimeDisc",
                  "TravelTimeDisc11"], group_keys=False).size().to_frame()
        df_algorithm2_output_for_discretization_concat_group0.to_csv(
            os.path.join(resultsPath + "df_algorithm2_output_for_discretization_concat_group0.csv"))
        df_algorithm2_output_for_discretization_concat_group1 = \
            df_algorithm2_output_for_discretization_concat.groupby(
                [ "subnetwork", "TotalCostDisc",
                  "TotalCostDisc11"], group_keys=False).size().to_frame()
        df_algorithm2_output_for_discretization_concat_group1.to_csv(
            os.path.join(resultsPath + "df_algorithm2_output_for_discretization_concat_group1.csv"))
        df_algorithm2_input_sample_for_discretization_concate.to_csv(
            os.path.join(resultsPath + "df_algorithm2_input_sample_for_discretization_concate01" + '.csv'))
        conditional_prob_table_road_level_concate.to_csv(
            os.path.join(resultsPath + "conditional_prob_table_road_level_concate01" + '.csv'))
        conditional_prob_table_road_to_subnet_level_concate.to_csv(
            os.path.join(resultsPath + "conditional_prob_table_road_to_subnet_level_concate01" + '.csv'))

        df_algorithm3_input_sample_concatenated.to_csv(os.path.join(resultsPath +
                                                                    "df_algorithm3_input_sample_concatenated.csv"))
        df_algorithm3_input_sample_concatenated_discretized. \
            to_csv(os.path.join(resultsPath +
                                "df_algorithm3_input_sample_concatenated_discretized.csv"))

        '''
        write results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
        '''

print('Progressing with Algorithm 3')
df_algorithm3_output = pd.DataFrame(columns=['sample_run', 'TravelTime', 'TotalCost'])
df_algorithm3_output = system_instance.simulate_algorithm3_output_generation(network_instance,
                                                                             df_algorithm3_input_sample_concatenated,
                                                                             df_algorithm3_output, OverallSample)
df_algorithm3_output.to_csv(os.path.join(resultsPath + "df_algorithm3_output.csv"))

for MeasureId in range(0, 2, 1):
    Measure = MeasureList22[MeasureId]

    [df_algorithm3_output, DiscLay3out, binsLay2out] = discretization_instance.discretize_data_frame_algorithm2_and_3(
        df_algorithm3_output, Measure, n_discretization)

df_algorithm3_input_sample_concatenated.to_csv(os.path.join(resultsPath +
                                                            "df_algorithm3_input_sample_concatenated-discretized.csv"))
df_algorithm3_output.to_csv(os.path.join(resultsPath + "df_algorithm3_output_after_discretization.csv"))
df_algorithm3_output_group0 = df_algorithm3_output.groupby(
        ["TravelTimeDisc", "TravelTimeDisc11"], group_keys=False).size().to_frame()
df_algorithm3_output_group0.to_csv(os.path.join(resultsPath + "df_algorithm3_output_group0.csv"))
df_algorithm3_output_group1 = df_algorithm3_output.groupby(
        ["TotalCostDisc",	"TotalCostDisc11"], group_keys=False).size().to_frame()
df_algorithm3_output_group1.to_csv(os.path.join(resultsPath + "df_algorithm3_output_group1.csv"))

conditional_prob_table_system_level = pd.DataFrame(
    columns=['DepenVar', 'DepenVarLvel', 'IndepenVar', 'IndepenVarLvel',
             'IndepenVar1', 'IndepenVar1Lvel', 'IndepenVar2', 'IndepenVar2Lvel',
             'IndepenVar3', 'IndepenVar3Lvel', 'IndepenVar4', 'IndepenVar4Lvel', 'CondProb'])
conditional_prob_table_system_level = conditional_tables_instance.ConditionalProbTabSystemLevelJuly2022(
    conditional_prob_table_system_level, df_algorithm3_input_sample_concatenated_discretized,
    df_algorithm3_output, conditional_prob_table_road_to_subnet_level_concate, n_discretization, OverallSample, 4)
conditional_prob_table_system_level.to_csv(os.path.join(resultsPath + "conditional_prob_table_system_level.csv"))

# TODO discretize and conditional probabiltiues for algorithm 3 .. TODO R function for BN
#  pdAlg3output=SimulateAlg2Output(Subnet, EdgeAll,EdgeListWithBridges,EdgeListWithBridgesShortDis,EdgesWithTraffic,
#  EdgesWithTrafficProb,EdgesWithTrafficOtherTime,BridgeRoadMat, TurningMat,TurningMatMod,TravelingTimeEdge,
#  EdgeNumWithBridges, pdAlg2output,pdAlg2InputSample,OverallSample)


# Search for BigAssumptionMohsen this is an important about the complicated equation
# of subnetwork simulation here we examine the set AA in which those edges with higher than median costs, and
# there is an or condition edded in order to handle situations in which the number of edges/roads
# wwith bridges are less than 2-3, then that set AA could be empty or with only one element, and
# that makes calculation of pair of edges impossible!
exit()
