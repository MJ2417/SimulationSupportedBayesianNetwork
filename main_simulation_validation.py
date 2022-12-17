##Bridges, California State Highway System, 2015
##https://purl.stanford.edu/td948by1459
##Traffic Volume, California, 2014
##https://searchworks.stanford.edu/view/fx041qj6799
##

# os.environ["PATH"] += os.pathsep+ r"C:\Program Files\Graphviz\bin"

import os, sys
import warnings

# warnings.filterwarnings('ignore')

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.2.1"
os.environ["PATH"] = r"C:\Program Files\R\R-4.2.1\bin\x64" + ";" + os.environ["PATH"]
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
resultsPath = 'C:\\Users\\Mohsen\\Documents\\PythonProjects\\Opac\\OutputResultsValidation\\'
for Subnet in range(1, 5, 1):
    '''
    define dataframes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    df_algorithm1_bridge_level = pd.DataFrame(
        columns=['subnet', 'sim_run', 'BridgeID', 'ListMaintenanceRates', 'ListMaintenanceRatesCoded', 'Edge', 'Avail',
                 'TotalCost'])
    df_algorithm2_input_sample = pd.DataFrame(columns=['subnet', 'sample_run', 'Edge', 'Avail', 'TotalCost'])
    df_algorithm2_output = pd.DataFrame(columns=['subnetwork', 'sample_run', 'TravelTime', 'TotalCost'])
    if Subnet == 1:
        df_algorithm1_bridge_level_concatenated = pd.DataFrame({}, columns=df_algorithm1_bridge_level.columns.values,
                                                               index=None)
        df_algorithm2_input_sample_concatenated = pd.DataFrame({}, columns=df_algorithm2_input_sample.columns.values,
                                                               index=None)
        df_algorithm2_output_concatenated = pd.DataFrame({}, columns=df_algorithm2_output.columns.values, index=None)

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
    df_algorithm2_input_sample = roads_instance.inputs_algorithm2_simulation_validation(resultsPath,
                                                                                        df_algorithm1_bridge_level,
                                                                                        df_algorithm2_input_sample,
                                                                                        measure_list,
                                                                                        edge_with_bridges_list,
                                                                                        EdgeNumWithBridges, Subnet,
                                                                                        OverallSample, PlotDist)

    # Inputs of third algorithm
    print(len(df_algorithm2_input_sample))
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

    df_algorithm2_output_concatenated = pd.concat([df_algorithm2_output_concatenated, df_algorithm2_output])
    df_algorithm1_bridge_level_concatenated = pd.concat([df_algorithm1_bridge_level_concatenated,
                                                         df_algorithm1_bridge_level])

    subnetwork_instance.plot_subnet_results(df_algorithm2_output, MeasureList22, resultsPath, Subnet)
    if Subnet == 4:
        '''
        write results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        '''
        df_algorithm1_bridge_level_concatenated.to_csv(os.path.join(resultsPath +
                                                                    "df_algorithm1_bridge_level_concatenated.csv"))
        df_algorithm2_output_concatenated.to_csv(os.path.join(resultsPath + "df_algorithm2_output_concatenated.csv"))
        '''
        write results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
        '''

exit()
