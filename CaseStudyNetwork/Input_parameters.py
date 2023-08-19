resultsPath = 'C:\\Users\\Mohsen\\Documents\\PythonProjects\\Opac\\OutputResults29June2023V56disc8\\'
RM = [[0.2, 0.5], [0.15, 0.4]]
Q10 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.25, 0.05, RM[0][0]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
Q11 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.2, 0.05, RM[1][0]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
Q20 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.55, 0.05, RM[0][1]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
Q21 = ([[-1 / 30, 1 / 30, 0, 0], [0, -0.45, 0.05, RM[1][1]], [0.1, 0.2, -0.3, 0], [1, 0, 0, -1]])
Simulatingtime = 1200
warm_up = 500
RehabFullCost = 200
MaintenanceCost = 10
AvailCoeff = [1.0, 0.6, 0.0, 0.3] #for OutputResults29June2023V56disc8
#AvailCoeff = [1.0, 0.8, 0.0, 0.6]  # for OutputResults29June2023V55disc8
# for OutputResults29June2023V05disc6, n_sample = 2000 is changed to n_sample = 1500, and OverallSample = 2000 to 1500,

### For simulation validation, NumSimRun > OverallSample
NumSimRun = 1500#2000  # road simulation
n_sample = 500#1500  # for building density algorithms 2 &3
OverallSample = 500#1500  # for generation for input/output sampling of algorithms 1,2,3
PlotDist = 0
measure_list = ['Avail', 'TotalCost']
MeasureList22 = ['TravelTime', 'TotalCost']
# Subnet=4
n_discretization = 8
