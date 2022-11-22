# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:01:19 2017

@author: Joost Berkhout

Description: this module contains different Markov chain functions.
"""

from __future__ import division
import numpy as np
from numpy import mean
import scipy.linalg as la
import numpy.linalg as npla
import csv
import math
from math import pi
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import pydotplus
import matplotlib.image as mpimg
from tarjan import tarjan
import scipy.io
from scipy import sparse

# input functions:
# ----------------
def load_x_and_y_coord(name_csv_file):
    r"""This function loads x and y coordinates in from a csv file. The first
    row contains the labels and is skipped."""

    # init
    vIndex = []
    vX = []
    vY = []

    with open(name_csv_file, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        reader.next() # skip labels
        for row in reader:
            vIndex = vIndex + [int(row[0])-1]
            vX = vX + [float(row[1])]
            vY = vY + [float(row[2])]

    # create dictionary with the coordinates
    coord_dict = dict(zip(vIndex,zip(vX, vY)))

    return vX, vY, coord_dict

def load_csv_comp_Gaussian(name_csv_file):
    r"""Loads x and y coordinates in from a csv file and calculates the
    Gaussian similarity function for all pairs to obtain a similarity matrix.
    """

    # load coordinates
    vX, vY, coord_dict = load_x_and_y_coord(name_csv_file)

    return Guassian_similarity_matrix(vX, vY)

def Guassian_similarity_matrix(dataset, scale_fact=3):
    r"""Calculates the Gaussian similarity function for a dataset. Every row
    in dataset, consists of an x and y coordinate, respectively.
    """

    # init
    n = len(dataset)  # number (n) of datapoint

    # generate mP using Gaussian similarity function:
    # -----------------------------------------------
    # 1) find sigma first
    mu = mean(dataset, 0)  # calc. mean of datapoints
    sigma = 0
    for data in dataset:
        sigma += la.norm(data - mu)**2
    sigma = sigma/(n-1)

    # 2) using sigma fill in the P matrix
    # for idea: see http://stackoverflow.com/questions/22720864/ ...
    # ... efficiently-calculating-a-euclidean-distance-matrix-using-numpy
    complex_repr = np.array([[complex(d[0], d[1]) for d in dataset]])
    eucl_dist_matrix = abs(complex_repr.T-complex_repr)
    S = np.exp((-1)*eucl_dist_matrix**2/(2*sigma/scale_fact))

    return sigma, S


def plot_x_y_data(vX, vY, b_labels = False, name_file='No file name given'):
    r"""This function plots the x and y coordinates as given in vX and vY."""

    # initialize plot
    plt.figure(name_file)
    plt.clf() # remove previous plot
    lColourScheme = ['b.', 'r.', 'g.', 'k.', 'y.','b*', 'r*', 'g*', 'k*', 'y*'
                     ,'b+', 'r+', 'g+', 'k+', 'y+']
    font = {'family' : 'arial',
            'weight' : 'normal',
            'size'   : 18}
    plt.rc('font', **font)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(name_file)
    plt.plot(vX, vY, 'bo')
    plt.axis('equal')

    if b_labels:
        # label all nodes
        n = len(vX)
        axes = plt.gca()
        for i in range(n):
            axes.annotate(i, (vX[i]+0.3,vY[i]), fontsize = 11)

def plot_clusters(vX, vY, EC, vTrans, method_name=None, fig_nr=2):
    if method_name is None:
        method_name = 'No name for the clustering method was given.'

    # initialize plot
    plt.figure(fig_nr)
    plt.clf() # remove previous plot
    lColourScheme = ['b.', 'r.', 'g.', 'k.', 'y.','b*', 'r*', 'g*', 'k*', 'y*'
                     ,'b+', 'r+', 'g+', 'k+', 'y+']
    n = len(lColourScheme)  # number of colourschemes
    font = {'family' : 'arial',
            'weight' : 'normal',
            'size'   : 18}
    plt.rc('font', **font)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(method_name)
    plt.axis('equal')

    # plot ergodic classes
    numb_EC = len(EC)
    for i in range(numb_EC):
        # plot i-th ergodic class
        vXPart = [vX[j] for j in EC[i]]
        vYPart = [vY[j] for j in EC[i]]
        plt.plot(vXPart, vYPart, lColourScheme[i%n],
                 label='EC {i}'.format(i = i+1) )

    # plot transient states
    n_tr = len(vTrans)  # number of s.c.c. consisting of transient states
    for i in range(n_tr):  # for transient s.c.c. i
        vXPart = [vX[j] for j in vTrans[i]]
        vYPart = [vY[j] for j in vTrans[i]]
        plt.plot(vXPart,
                 vYPart,
                 lColourScheme[(i+numb_EC)%n],
                 label='Tr. st. of s.c.c. {i}'.format(i = i + 1))
    plt.legend()

    if n < numb_EC + n_tr:
        print('Warning: Colourschemes have been reused.'
              'Generate more colourschemes in plot.cluster()'
              ' to avoid this message.')
        print('Number of colors needed = {i}'.format(i = numb_EC + n_tr))
        print('Number of colourschemes = {i}'.format(i = n))

#    # label all nodes
#    axes = plt.gca()
#    n = len(vX)
#    for i in range(n):
#        axes.annotate(i, (vX[i],vY[i]), fontsize = 10)

def create_graph(A):
    r"""Based on a given A, this function returns a graph in the common python
    structure: dictionaries. Every (i,j): A(i,j)>0 is an edge by assumption."""

    graph = {}
    for i in range(len(A)):
        graph[i] = np.where(A[i] > 0)[0].tolist()

    return graph

def det_MC_structure(P):
    r"""This function determines the Markov chain (MC) structure, i.e., it
    splits the MC described by A into ergodic classes and transient states."""

    # init
    scc = tarjan(create_graph(P))  # strongly connected components (scc)
    n = len(scc)  # number of scc
    erg_classes = []  # a list to keep track of the ergodic classes
    tr_states = []  # a list to keep track of transient states
    prec = 10**(-10)  # precision used

    for i in range(n):  # evaluate scc i
        scc_i = scc[i]  # scc i
        size_scc_i = len(scc_i)  # number of states in scc i
        if abs(np.sum(P[np.ix_(scc_i,scc_i)]) - size_scc_i) < prec:
            # no 'outgoing' probabilities: scc i is an ergodic class
            erg_classes.append(scc_i)
        else:  # scc i is a transient connected component
            tr_states.append(scc_i)

    return erg_classes, tr_states

def create_edge_list(A):
    r"""Based on a given A, this function returns a list of adges for all edges
    in A which are non-zero."""
    edge_list = []
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i][j] > 0:
                edge_list.append((i+1,j+1))

    return edge_list

def plot_graph(A, graph_name=None):
    r"""Using Graphviz, this function plots the graph corresponding to A."""

    # init
    n = len(A)  # number of nodes
    graph_dict = create_graph(A)  # create graph dictionary
    if graph_name is None:  # check for graph name
        graph_name = 'no_file_name_given.pdf'

    # initialize graph
    graph = pydotplus.Dot(rankdir='LR',
                          graph_type='digraph',
                          layout='circo',
                          pad="0.1",
                          nodesep="0.5",
                          ranksep="2")

    # create node-objects in vector and add nodes to graph
    nodes = []
    for i in range(n):
        # create node i
        node_i = pydotplus.Node(str(i), shape='circle', fixedsize='true')
        nodes.append(node_i)

        # add node to graph
        graph.add_node(node_i)

    # add edges to graph
    for i in range(n):
        # add edges (i,j) for all j
        for j in graph_dict[i]:
            graph.add_edge(pydotplus.Edge(nodes[i], nodes[j],
                                          penwidth = 0.1+2*A[i][j]))

    # save and plot graph
    graph.write_pdf(graph_name+'.pdf')
    graph.write_jpg(graph_name+'.jpg')
    img = mpimg.imread(graph_name+'.jpg')
    plt.figure(graph_name)
    plt.imshow(img)

def PointsInCircum(r,n):
    r"""This function returns n coordinates spread along the circle with
    center at (0,0) with circumvent r. """
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in xrange(0,n+1)]

def fixed_plot_color_graph(A, erg_classes, tr_states, graph_name = None):
    r"""Using Graphviz, this function plots the graph corresponding to A. In
    particular, nodes from the same list in erg_classes and tr_states are
    colored in the same color. Transient states will be plotted as triangles
    while ergodic states will be plotted by circles. This is a copy of
    plot_color_graph where the node positions are fixed. """

    if graph_name is None:  # check for graph name
        graph_name = 'no_file_name_given.pdf'

    # init
    erg_classes = list(reversed(erg_classes))  # so that erg. classes get
                                               # the same colors
    n = len(A)  # number of nodes
    n_EC = len(erg_classes)  # number of ergodic classes (EC)
    n_Tr = len(tr_states)  # number of scc of transient (Tr) states
    vColors = ['lightblue2', 'seagreen3', 'indianred1', 'mediumslateblue',
               'limegreen', 'darkseagreen', 'palegoldenrod', 'saddlebrown',
               'tomato', 'red', 'grey52', 'dimgrey', 'purple', 'salmon',
               'yellowgreen', 'cornflowerblue',  'mediumvioletred', 'yellow']  # load some colors
    numb_colors = len(vColors)  # number of loaded colors
    if n_EC + n_Tr > numb_colors: raise TypeError(('Please give more colors.'
                                                   '(' + str(n_EC + n_Tr) +
                                                   ' colors required)'))
    graph_dict = create_graph(A)  # create graph dictionary
    nodes_coord = PointsInCircum(5,n)  # coordinates of the nodes aligned
                                         # on a circle (default values: 2.5)

    # initialize graph
    graph = pydotplus.Dot(rankdir='LR',
                          graph_type='digraph',
                          layout='neato',
                          pad="0.1",
                          nodesep="0.25",
                          ranksep="equally")

    # create node-objects for all ergodic classes and add nodes to graph
    nodes = range(n)  # to save the references to all nodes-objects
    for i in reversed(range(n_EC)):  # for EC i
        for j in erg_classes[i]:
            # create node j
            x, y = nodes_coord[j]  # coordinates of the nodes
            node_pos = str(x) + ',' + str(y) + '!'  # in Graphviz language
            node_j = pydotplus.Node(str(j+1),
                                    fontsize='25',
                                    pos=node_pos,
                                    shape='circle',
                                    fixedsize='true',
                                    style="filled",
                                    fillcolor=vColors[i])
            nodes[j] = node_j  # save reference to node
            graph.add_node(node_j)  # add node to graph

    # create node-objects for all transient states and add nodes to graph
    for i in range(n_Tr):  # for transient scc i
        for j in tr_states[i]:
            # create node j
            x, y = nodes_coord[j]  # coordinates of the nodes
            node_pos = str(x) + ',' + str(y) + '!'  # in Graphviz language
            node_j = pydotplus.Node(str(j+1),
                                    fontsize='25',
                                    pos=node_pos,
                                    shape='square',
                                    fixedsize='true',
                                    style="filled",
                                    fillcolor=vColors[i + n_EC])
            nodes[j] = node_j  # save reference to node
            graph.add_node(node_j)  # add node to graph

    # add edges to graph
    for i in range(n):
        # add edges (i,j) for all j
        for j in graph_dict[i]:
            graph.add_edge(pydotplus.Edge(nodes[i],
                                          nodes[j],
                                          penwidth = 0.1+2*A[i][j]))

    # save and plot graph
    folder = 'Pictures/'  # \\ instead of / on Windows
    graph.write_pdf(folder + graph_name+'.pdf')
    graph.write_jpg(folder + graph_name+'.jpg')
    img = mpimg.imread(folder+graph_name+'.jpg')
    plt.figure(graph_name)
    plt.imshow(img)

def plot_color_graph(A, erg_classes, tr_states, graph_name = None):
    r"""Using Graphviz, this function plots the graph corresponding to A. In
    particular, nodes from the same list in erg_classes and tr_states are
    colored in the same color. Transient states will be plotted as triangles
    while ergodic states will be plotted by circles. """

    if graph_name is None:  # check for graph name
        graph_name = 'no_file_name_given.pdf'

    # init
    erg_classes = list(reversed(erg_classes))  # so that erg. classes get
                                               # the same colors
    n = len(A)  # number of nodes
    n_EC = len(erg_classes)  # number of ergodic classes (EC)
    n_Tr = len(tr_states)  # number of scc of transient (Tr) states
    vColors = ['lightblue2', 'seagreen3', 'indianred1', 'mediumslateblue',
               'limegreen', 'darkseagreen', 'palegoldenrod', 'saddlebrown',
               'tomato', 'red', 'grey52', 'dimgrey', 'purple', 'salmon',
               'yellowgreen', 'cornflowerblue',  'mediumvioletred', 'yellow']  # load some colors
    numb_colors = len(vColors)  # number of loaded colors
    if n_EC + n_Tr > numb_colors: raise TypeError(('Please give more colors.'
                                                   '(' + str(n_EC + n_Tr) +
                                                   ' colors required)'))
    graph_dict = create_graph(A)  # create graph dictionary

    # initialize graph
    graph = pydotplus.Dot(rankdir='LR',
                          graph_type='digraph',
                          layout='circo',
                          pad="0.1",
                          nodesep="0.25",
                          ranksep="equally",
    #                          bgcolor='lightgrey'
                          )

    # create node-objects for all ergodic classes and add nodes to graph
    nodes = range(n)  # to save the references to all nodes-objects
    for i in reversed(range(n_EC)):  # for EC i
        for j in erg_classes[i]:
            # create node j
            node_j = pydotplus.Node(str(j+1),
                                    fontsize='25',
                                    shape='circle',
                                    fixedsize='true',
                                    style="filled",
                                    fillcolor=vColors[i])
            nodes[j] = node_j  # save reference to node
            graph.add_node(node_j)  # add node to graph

    # create node-objects for all transient states and add nodes to graph
    for i in range(n_Tr):  # for transient scc i
        for j in tr_states[i]:
            # create node j
            node_j = pydotplus.Node(str(j+1),
                                    fontsize='25',
                                    shape='square',
                                    fixedsize='true',
                                    style="filled",
                                    fillcolor=vColors[i + n_EC])
            nodes[j] = node_j  # save reference to node
            graph.add_node(node_j)  # add node to graph

    # add edges to graph
    for i in range(n):
        # add edges (i,j) for all j
        for j in graph_dict[i]:
            graph.add_edge(pydotplus.Edge(nodes[i],
                                          nodes[j],
                                          penwidth = 0.1+2*A[i][j]))

    # save and plot graph
    folder = 'Pictures/'  # \\ instead of / on Windows
    graph.write_pdf(folder + graph_name+'.pdf')
    graph.write_jpg(folder + graph_name+'.jpg')
    img = mpimg.imread(folder+graph_name+'.jpg')
    plt.figure(graph_name)
    plt.imshow(img)

def plot_color_graph_Zachary(A, erg_classes, tr_states, graph_name=None):
    r""" This function is a copy of the function plot_color_graph, however it
    allows the font sizes to be based on the known underlying structure in the
    Zachary data. In particular, set true_cluster_set contains the set of
    social agents that belong to Mr. Hi's faction or belong to Mr. Hi's
    club after the fission. """

    if graph_name is None:  # check for graph name
        graph_name = 'no_file_name_given.pdf'

    # Set up the true clustering (uncomment the required clustering)

#    # based on faction:
#    true_cluster_set = {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 18, 20, 22}

    # based on fission:
    true_cluster_set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17,
                        18, 20, 22}

    # init
    n = len(A)  # number of nodes
    n_EC = len(erg_classes)  # number of ergodic classes (EC)
    n_Tr = len(tr_states)  # number of scc of transient (Tr) states
    vColors = ['mediumslateblue', 'cornflowerblue', 'steelblue', 'turquoise4',
               'limegreen', 'darkseagreen', 'palegoldenrod', 'burlywood1',
               'tomato', 'red', 'grey52', 'dimgrey', 'purple', 'salmon',
               'yellowgreen', 'mediumvioletred', 'yellow']  # load some colors
    numb_colors = len(vColors)  # number of loaded colors
    if n_EC + n_Tr > numb_colors: raise TypeError(('Please give more colors.'
                                                   '(' + str(n_EC + n_Tr) +
                                                   ' colors required)'))
    graph_dict = create_graph(A)  # create graph dictionary

    # initialize graph
    graph = pydotplus.Dot(rankdir='LR',
                          graph_type='digraph',
                          layout='circo',
                          pad="0.1",
                          nodesep="0.5",
                          ranksep="2")

    # create node-objects for all ergodic classes and add nodes to graph
    nodes = range(n)  # to save the references to all nodes-objects
    for i in range(n_EC):  # for EC i
        for j in erg_classes[i]:
            # check for font color: in true clustering red, else black
            if j+1 in true_cluster_set:
                font_color = 'white'
            else:
                font_color = 'black'

            # create node j
            node_j = pydotplus.Node(str(j+1),
                                    fontcolor = font_color,
                                    fontsize='25',
                                    shape='circle',
                                    fixedsize='true',
                                    style="filled",
                                    fillcolor=vColors[i])
            nodes[j] = node_j  # save reference to node
            graph.add_node(node_j)  # add node to graph

    # create node-objects for all transient s.c.c.'s and add nodes to graph
    for i in range(n_Tr):  # for transient scc i
        for j in tr_states[i]:

            # check for font color: in true clustering red, else black
            if j+1 in true_cluster_set:
                font_color = 'white'
            else:
                font_color = 'black'

            # create node j
            node_j = pydotplus.Node(str(j+1),
                                    fontcolor = font_color,
                                    fontsize='25',
                                    shape='square',
                                    fixedsize='true',
                                    style="filled",
                                    fillcolor=vColors[i + n_EC])
            nodes[j] = node_j  # save reference to node
            graph.add_node(node_j)  # add node to graph

    # add edges to graph
    for i in range(n):
        # add edges (i,j) for all j
        for j in graph_dict[i]:
            graph.add_edge(pydotplus.Edge(nodes[i],
                                          nodes[j],
                                          #len="2",
                                          penwidth = 0.1+2*A[i][j]))

    # save and plot graph
    folder = 'Pictures/'  # in windows \\
    graph.write_pdf(folder + graph_name+'.pdf')
    graph.write_jpg(folder + graph_name+'.jpg')
    img = mpimg.imread(folder+graph_name+'.jpg')
    plt.figure(graph_name)
    plt.imshow(img)

# exact Markov chain functions:
# -----------------------------

def norm_rows(A, rows_to_normalize=None):
    r"""This function normalizes the rows of A, such that the rows sum up to
    one. In case row sum is 0, places a one on the diagonal."""

    # init
    n = len(A)
    norm_A = np.array(A)  # normalized version of A
    if rows_to_normalize is None:
        rows_to_normalize = range(n)  # consider all rows

    # start normalizing the rows of A
    for i in rows_to_normalize:  # consider i-th row
        i_row_sum = np.sum(A[i])
        if i_row_sum == 0:
            print('Warning: the ' + str(i) + '-th row sums up to zero'
                  ' in the function norm_rows from Markov_chain.py.'
                  ' The row is replaced by the ' + str(i) + '-th row of the '
                  'identity matrix.')
            norm_A[i][i] = 1
#            norm_A[i,:] = 1/n
        else:
            norm_A[i] /= i_row_sum

    return norm_A

def load_Courtois():
    r"""This function returns the Courtois Markov chain transition matrix."""

    mP = np.array([[.85, 0, .149, .0009, 0, .00005, 0, .00005],
                    [.1, .65, .249, 0, .0009, .00005, 0, .00005],
                    [.1, .8, .0996, .0003, 0, 0, .0001, 0],
                    [0, .0004, 0, .7, .2995, 0, .0001, 0],
                    [.0005, 0, .0004, .399, .6, .0001, 0, 0],
                    [0, .00005, 0, 0, .00005, .6, .2499, .15],
                    [.00003, 0, .00003, .00004, 0, .1, .8, .0999],
                    [0, .00005, 0, 0, .00005, .1999, .25, .55]])

    return mP

def load_Land_of_Oz():
    """Example Markov chain from Finite Markov Chains by Kemeny & Snell (1976).
    """

    mP = np.array([[.5, .25, .25],
                   [.5, 0, .5],
                   [.25, .25, .5]])

    return mP

def load_cockroach_graph(k):
    r"""This function returns the the cockroach graph from
    Guattery and Miller (1998) with 4*k nodes.
    The top row is labeled as 1,2,...,2k while the second row is labeled from
    4k,4k+1,...,2k+1. """

    # init
    n = 4*k  # number of nodes
    P = np.zeros((n,n))  # transition matrix

    # connect nodes from left to right in top row, then connect 2k with 2k+1
    # and then connect second row from right to left all the way to 4k.
    for i in range(n-1):
        P[i, i+1] = 1
        P[i+1, i] = 1

    # add extra connections between the two rows
    for j in range(k-1):
        P[k+j, 3*k-1-j] = 1
        P[3*k-1-j, k+j] = 1

    return norm_rows(P)

def calc_stationary_distr(mP = None):
    r"""This function returns the stationary distribution of mP."""

    if mP is None:
        mP = load_Courtois()

    # init
    n = len(mP)
    mI = np.eye(n)
    mZ = mP - mI
    mZ[:,[0]] = np.ones((n,1))

    if la.det(mZ) != 0:
        return la.inv(mZ)[[0],:]
    else:
        raise ValueError('Please enter a Markov uni-chain.')

def calc_ergodic_proj(mP = None):
    r""" This function returns the ergodic projector of a
    Markov multi-chain mP. """

    if mP is None:
        mP = load_Courtois()

    # init
    n = len(mP)
    ec, tc = det_MC_structure(mP)  # ergodic (e) and transient (t) classes (c)

    # check for structure
    if len(ec) > 1:
        # multiple ergodic classes
        print('Markov chain mP has multiple ergodic classes.')
    else:
        # only one ergodic class
        if len(tc) > 0:
            print ('Markov chain mP has transient states.')

        # calculate and return ergodic projector
        return np.dot(np.ones((n, 1)), calc_stationary_distr(mP))

    # mP is of a multi-chain structure: calculate ergodic projector

    # init
    mPiP = np.zeros((n, n))

    # for the ergodic classes:
    for e in ec:
        # determine ergodic project part of ergodic class e
        indexes_e = np.ix_(e, e)  # correct indexes of ergodic class e
        mPe = mP[indexes_e]  # submatrix of ergodic class e
        mPiP[indexes_e] = np.dot(np.ones((len(e), 1)),
                                 calc_stationary_distr(mPe))

    # for the transient states
    tc_flatten = [t for tc_sub in tc for t in tc_sub]  # flatten version of tc
    n_t = len(tc_flatten)  # number of transient states
    if n_t > 0:
        # there are transient states

        # init
        indexes_t = np.ix_(tc_flatten, tc_flatten)  # indexes of transient st.
        mPt = mP[indexes_t]  # transient part of mP
        mIt = np.eye(n_t)  # identity matrix of transient size
        mTrtoErg = npla.inv(mIt - mPt)  # erg. prob. from tr. st. to erg. st.

        # calculate the ergodic transient part
        for e in ec:
            # for ergodic class e

            # init
            indexes_e = np.ix_(e, e)  # indexes ergodic class
            indexes_te = np.ix_(tc_flatten, e)  # indexes from tr. to erg.
            mPTToE = mP[indexes_te]  # transient part to ergodic class e

            # calc. erg. projector transient part to ergodic class e
            mPiP[indexes_te] = mTrtoErg.dot(mPTToE).dot(mPiP[indexes_e])

    return mPiP

def calc_deviation_matrix(mP = None):
    r"""This function calculates the deviation matrix of a
    Markov uni-chain mP"""

    if mP is None:
        mP = load_Courtois()

    # init
    n = len(mP)
    mI = np.eye(n)
    mPiP = calc_ergodic_proj(mP)

    return la.inv(mI - mP + mPiP) - mPiP

def calc_mean_first_pass(mP = None):
    r"""This function calculates the mean first passage matrix of mP."""

    if mP is None:
        mP = load_Courtois()

    # init
    n = len(mP)
    mI = np.eye(n)
    mPiP = calc_ergodic_proj(mP)
    D_P = calc_deviation_matrix(mP)
    dg_D_P = np.diag(np.diag(D_P))
    dg_mPiP_inv = la.inv(np.diag(np.diag(mPiP)))
    mOnes = np.ones((n,n))

    return (mI-D_P+mOnes.dot(dg_D_P)).dot(dg_mPiP_inv)

def calc_Kemeny_constant(mP = None):
    mD_P = calc_deviation_matrix(mP)
    return np.trace(mD_P) + 1

def calc_Kemeny_derivatives_exact(mP = None):
    r"""This function calculates the exact expression for the Kemeny constant
    derivatives."""

    if mP is None:
        mP = load_Courtois()

    # init
    n = len(mP)
    D_P = calc_deviation_matrix(mP)
    D_P_sq = np.linalg.matrix_power(D_P, 2)

    # calc. derivatives in all directions
    mK_der = np.zeros((n, n))
    for i in range(0, n):
        substr_val = np.dot(mP[[i], :], D_P_sq[:, [i]])
        for j in range(0, n):
            mK_der[i, j] = D_P_sq[j, i] - substr_val

    return mK_der

def sort_edges(A):
    r"""Array A is seen as corresponding with edges. Based on the elements in A
    the edges are sorted from low to large values."""

    # find N smallest indexes
    n = len(A)
    A_flat = np.ndarray.flatten(A)
    sorted_indexes = np.argsort(A_flat)
    (vX, vY) = np.unravel_index(sorted_indexes, (n, n))  # vX and vY are sorted

    return vX, vY

def cut_edges(A, vX, vY, b_both_edges=False):
    r"""Set all edges (vX, vY) in A to zero and renormalize rows so that the
    rows sum up to 1 again. When b_bot_edges is set to True, also the edges
    (vY, vX) will be cut."""

    # init
    A_cut = np.array(A)

    # cut edges (vX, vY)
    A_cut[vX, vY] = 0
    rows_to_norm = vX  # for efficiency: save the norms that have to be norm.

    if b_both_edges:
        # cut edges (vY, vX)
        A_cut[vY, vX] = 0  # optional
        rows_to_norm = np.append(vX, vY)

    return norm_rows(A_cut, np.unique(rows_to_norm))

def Kemeny_cutting(mP, mK_der, N, b_sym=False, verbose=True):
    r"""This function cuts N edges from P which have smallest values for
    mK_der and for which P-value is > 0. Afterwards, mP is normalized. """

    # init
    n = len(mP)
    mPCut = np.array(mP)  # make a copy
    numbOfEdgesLeft = (mP>0).sum() - (np.diag(mP)>0).sum()   # number of edges left (ignore self-loops)

    # find N smallest indexes
    mK_der_inf = np.array(mK_der)
    mK_der_inf[mPCut==0] = np.inf  # only consider existing edges
    np.fill_diagonal(mK_der_inf, np.inf)  # only consider non-diagonal edges
    sorted_indexes = np.argsort(np.ndarray.flatten(mK_der_inf))
    (vX, vY) = np.unravel_index(sorted_indexes, (n,n))  # vX and vY are sorted

    if numbOfEdgesLeft < N:
        raise Warning('There are not enough edges left to cut. The given mP is returned.')
    else:
        # cut first N edges
        if verbose:
            print('The following edges are cut: ')
            print(zip(vX[0:N], vY[0:N]))
        mPCut[(vX[0:N], vY[0:N])] = 0
        if b_sym:
            mPCut[(vY[0:N], vX[0:N])] = 0

    return norm_rows(mPCut)

# modified resolvent functions:
# -----------------------------

def calc_G_alpha(mP = None, alpha = 10**(-6)):
    r"""This function loads $G_\alpha(P)$."""

    if mP is None:
        mP = load_Courtois()

    # init
    n = len(mP)
    mI = np.eye(n)

    return la.inv(mI-(1-alpha)*mP)

def calc_mod_resolvent(mP = None, alpha = 10**(-6), mG_alpha = None):
    r"""This function calculates the modified resolvent of a matrix"""

    if mP is None:
        mP = load_Courtois()

    if mG_alpha is None:
        mG_alpha = calc_G_alpha(mP, alpha)

    return (alpha*mP).dot(mG_alpha)

def calc_mod_resolvent_D_P(mP = None, alpha = 10**(-6), mH_alpha = None):
    r"""This function calculates the modified resolvent approximation
    for the deviation matrix $D_P$, indicated by $D_\alpha(P)$."""

    if mP is None:
        mP = load_Courtois()

    if mH_alpha is None:
        mH_alpha = calc_mod_resolvent(mP, alpha)

    # init
    n = len(mP)
    mI = np.eye(n)

    return (mI - mH_alpha).dot(mI + ((1-alpha)/alpha)*mH_alpha)

def calc_mod_resolvent_K_P(mP = None, alpha = 10**(-6), mH_alpha = None):

    return np.trace(calc_mod_resolvent_D_P(mP, alpha, mH_alpha))+1

def calc_Kemeny_derivatives(mP = None, alpha = 10**(-6),
                            mG_alpha = None, mH_alpha = None):
    r"""This function calculates the modified resolvent approximation
    for the Kemeny constant derivatives. Optionally, one can insert the
    matrices for mG_alpha and mH_alpha. """

    if mP is None:
        mP = load_Courtois()

    if mG_alpha is None:
        mG_alpha = calc_G_alpha(mP, alpha)

    if mH_alpha is None:
        mH_alpha = calc_mod_resolvent(mP, alpha, mG_alpha)

    # init
    n = len(mP)
    mI = np.eye(n)

    # calc. some measures for the Kemeny derivatives
    mQ_1 = (1-2*alpha)*mI-2*(1-alpha)*mH_alpha
    mQ_2 = np.linalg.matrix_power(mG_alpha, 2)
    mQ = np.dot(mQ_1, mQ_2)
    mK_der = np.zeros((n, n))
    # calc. derivatives in all directions
    for i in range(0, n):
        substr_val = np.dot(mP[[i], :], mQ[:, [i]])
        for j in range(0, n):
            mK_der[i, j] = mQ[j, i] - substr_val

    return mK_der

def approx_find_MC_structure(mP, alpha=10**(-6)):
    r"""This function finds the Markov chain (MC) structure based on modified
    resolvent approximations and diagonal argument via alpha. Identifying the
    structure means finding transient states and ergodic classes. """

    # init
    n = len(mP)
    H_alpha = calc_mod_resolvent(mP, alpha)  # calc. mod. res.

    # determine transient and ergodic states
    vTransSt = []  # array of all transient states indexes
    vErgSt = []  # array of all ergodic states indexes
    vEC = []  # array of arrays the ergodic class indexes
    for i in range(0, n):
        if i not in vErgSt:  # i is not evaluated yet
            if H_alpha[i, i] < alpha:  # i is transient
                vTransSt += [i]
            else:  # i is ergodic
                EC_indexes = [j for j in range(n) if H_alpha[i, j] > alpha]
                vErgSt += EC_indexes
                vEC.append(EC_indexes)

    return vEC, vTransSt

def find_weakly_connected_components(A):
    r""" Based on a ndarray: find the weakly connected
    components (WCC). """

    return sparse.csgraph.connected_components(sparse.csr_matrix(A))

# resolvent functions
# -------------------

def calc_resolvent(mP = None, alpha = 10**(-6)):
    r""" Calculate the resolvent of a Markov transition matrix mP. """

    return alpha*calc_G_alpha(mP, alpha)

def calc_res_Taylor_series(N, mP, mQ, alpha = 10**(-6), mResP = None):
    r""" Calculate a Taylor for the resolvent of mQ by using the resolvent
    of mP as root. N+1 terms will be used. """

    if mResP is None:
        # calculate the resolvent of root mP
        mResP = calc_resolvent(mP, alpha)

    # init
    const = (1-alpha)/alpha
    partTerm = const*(mQ-mP).dot(mResP)
    curTerm = mResP
    curSum = np.zeros((len(mP), len(mP)))

    for n in range(N+1):

        # add current series term
        curSum += curTerm

        # update current series term
        curTerm = np.dot(curTerm, partTerm)

    seriesApprox = curSum
    mResQ = calc_resolvent(mQ, alpha)
    approxError = np.dot(mResQ, npla.matrix_power(partTerm, N+1))

    return seriesApprox, approxError, partTerm, curTerm

def random_multi_chain(structure, a):
    r""" Determine a random multi-chain of a given structure and where the
    random numbers in (0,1) are inflated with a number a. Structure is a list
    with integers. Each integer indicates the size of an ergodic class. The
    last integer in structure gives the number of transient states. """

    # fill block diagonals
    mA = []
    for i in structure:
        mA.append(a*np.random.rand(i, i))
    mA = la.block_diag(*mA)

    # add transient part
    n = sum(structure)
    nTr = structure[-1]
    strtIdxTr = n - nTr
    mA[strtIdxTr:, :strtIdxTr] = a*np.random.rand(nTr, n-nTr)

    return norm_rows(mA)

if __name__ == "__main__":
    print('This is a Markov chain module that can be imported.')
