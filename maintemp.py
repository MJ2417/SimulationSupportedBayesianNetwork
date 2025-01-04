

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


# df_algorithm3_output.to_csv(os.path.join(resultsPath + "df_algorithm3_output_after_discretization.csv"))
#
# df_algorithm3_input_sample_concatenated_discretized.to_csv(os.path.join(resultsPath +"df_algorithm3_input_sample_concatenated_discretized.csv"))
# conditional_prob_table_road_to_subnet_level_concate.to_csv(
#     os.path.join(resultsPath + "conditional_prob_table_road_to_subnet_level_concate01" + '.csv'))


df_algorithm3_output = pd.read_csv(os.path.join(resultsPath + "df_algorithm3_output_after_discretization.csv"))
df_algorithm3_input_sample_concatenated_discretized = pd.read_csv(os.path.join(resultsPath + "df_algorithm3_input_sample_concatenated_discretized.csv"))
conditional_prob_table_road_to_subnet_level_concate = pd.read_csv(os.path.join(resultsPath + "conditional_prob_table_road_to_subnet_level_concate01.csv"))


conditional_prob_table_system_level = pd.DataFrame(
    columns=['DepenVar', 'DepenVarLvel', 'IndepenVar', 'IndepenVarLvel',
             'IndepenVar1', 'IndepenVar1Lvel', 'IndepenVar2', 'IndepenVar2Lvel',
             'IndepenVar3', 'IndepenVar3Lvel', 'IndepenVar4', 'IndepenVar4Lvel', 'CondProb'])
conditional_prob_table_system_level = conditional_tables_instance.ConditionalProbTabSystemLevelJuly2022(
    conditional_prob_table_system_level, df_algorithm3_input_sample_concatenated_discretized,
    df_algorithm3_output, conditional_prob_table_road_to_subnet_level_concate, n_discretization, OverallSample, 4)
conditional_prob_table_system_level.to_csv(os.path.join(resultsPath + "conditional_prob_table_system_level.csv"))
