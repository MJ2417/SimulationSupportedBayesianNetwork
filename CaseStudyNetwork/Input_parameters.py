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


### For simulation validation, NumSimRun > OverallSample
NumSimRun = 2000  # road simulation
n_sample = 1500  # for building density algorithms 2 &3
OverallSample = 1500  # for generation for input/output sampling of algorithms 1,2,3
PlotDist = 0
measure_list = ['Avail', 'TotalCost']
MeasureList22 = ['TravelTime', 'TotalCost']
# Subnet=4
n_discretization = 5
