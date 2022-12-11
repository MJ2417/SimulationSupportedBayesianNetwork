# -*- coding: utf-8 -*-
"""
Created on 9/16/2018

Author: Joost Berkhout (CWI, email: j.berkhout@cwi.nl)

Description: This module contains different Markov chain functions.
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
from matplotlib import colors
#import matplotlib.colors as colors
import pydotplus
import matplotlib.image as mpimg
from tarjan import tarjan
import scipy.io
from scipy import sparse

from MarkovChain import util
from MarkovChain.util import *
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_agraph import write_dot
import MarkovChain.tools
import MarkovChain.Markov_chain as MCOld
from MarkovChain.tools import normalize_rows

class MarkovChain(object):
    """Capture a Markov chain with transition matrix P.

    Parameters
    ----------
    P : list/np.array/str
        A Markov chain transition matrix or name from data folder, i.e.,
        'Courtois', 'ZacharysKarateClub' or 'landOfOz'. If it is a matrix
        with non-negative values it is normalized.
    verbose : bool
        Decides information is print or not.

    Attributes
    ----------
    P : np.array
        The Markov chain probability transition matrix.
    ec : list
        List of lists with indexes of ergodic class members.
    tr : list
        List of lists with indexes of transient strongly connected components.
    bMultiChain : bool
        True when multi-chain, False else.
    bUniChain : bool
        True when uni-chain, False else.
    bTransientStates : bool
        True when there are transient states, false else.
    ranking : Ranking
        A Ranking object to calculate all kinds of rankings.
    n : int
        Number of states.
    nEc : int
        Number of ergodic classes.
    nTc : int
        Number of transient strongly connected components.
    pi : np.array
        Stationary distribution if uni-chain.
    Pi : np.array
        Ergodic projector.
    Z : np.array
        Fundamental matrix.
    D : np.array
        Deviation matrix.
    M : np.array
        Mean first passage matrix.
    V : np.array
        Variance first passage matrix.
    K : float
        Kemeny constant.
    KDer : np.array
        Kemeny constant derivatives.
    G : networkx
        Graph representation of the Markov chain (see also draw()).
    SCC : list of lists
        The strongly connected components.
    WCC : list of np.array
        The weakly connected components.
    nSCC : int
        Number of strongly connected components.

    Note
    ----
    Markov chain measures are only calculated once. Once calculated, it is
    cached and the cache is used afterwards.
    """

    def __init__(self, P, verbose=False):
        self.verbose = False #verbose
        self.P = P
        self.ec, self.tc = self.MC_structure()
        self.bMultiChain = len(self.ec) > 1
        self.bUniChain = not self.bMultiChain
        self.bTransientStates = len(self.tc) > 0
        self.ranking = Ranking(self)

        # TODO: add an option to change P into canonical shape, ensure an index mapping

    def __str__(self):
        sizesEC = [len(x) for x in self.ec]
        sizesTrSCC = [len(x) for x in self.tc]
        sizesWCC = [len(x) for x in self.WCC]
        nrTrStates = sum(sizesTrSCC)
        msg = (
            'Markov chain info\n' +
            '=================\n' +
            '{} states of which {} ergodic and {} transient\n'.format(self.n,
                                                              sum(sizesEC),
                                                              sum(sizesTrSCC)) +
            '{} ergodic classes\n'.format(len(self.ec)) +
            '{} transient strongly connected components\n'.format(len(self.tc)) +
            '{} weakly connected components\n'.format(len(self.WCC)) +
            'Sizes of the ergodic classes are {}\n'.format(sizesEC) +
            'Sizes of the transient strongly connected components are {}\n'.format(sizesTrSCC) +
            'Sizes of the weakly connected components are {}\n'.format(sizesWCC) +
            'The Kemeny constant is {}\n'.format(round(self.K,3)) +
            'The top 5 of Google\'s PageRank is {}\n'.format(self.ranking.Google_PageRank[0:5])
        )

        return msg

    @property
    def P(self):
        """Markov chain transition matrix. """
        return self._P

    @P.setter
    def P(self, P):
        """Set the Markov chain transition matrix with P.

        Parameters
        ----------
        P : list/np.array/str
            A Markov chain transition matrix or name from data folder, i.e.,
            'Courtois', 'ZacharysKarateClub' or 'landOfOz'.

        Raises
        ------
        TypeError
            If the given transition matrix is not feasible.
        """

        if isinstance(P, list):
            P = np.array(P)

        if isinstance(P, str):
            P = tools.data_loader(P)

        P = self.feasibility_check(P)

        self._P = np.array(P)  # a copy is mades

    def feasibility_check(self, P):
        """Check whether P is a feasible probability transition matrix.

        Parameters
        ----------
        P : np.array
            A Markov chain transition matrix.

        Raises
        ------
        TypeError
            If the given transition matrix is not feasible.
        """

        if np.sum(P < 0) > 0:
            raise TypeError('Found negative elements in P.')
        elif np.sum(np.sum(P, axis=1) != 1) > 0:
            if self.verbose: print('Not all row sums are 1, normalize rows.')
            P = normalize_rows(P, verbose=self.verbose)
        elif self.verbose:
            print('Loaded P seems to be feasible.')

        return P

    @util.cache_property
    def n(self):
        """Number of Markov chain nodes. """
        return len(self.P)

    @util.cache_property
    def nEc(self):
        """Number of ergodic classes. """
        return len(self.ec)

    @util.cache_property
    def nTc(self):
        """Number of transient strongly connected components. """
        return len(self.tc)

    def MC_structure(self):
        """Determine the ergodic classes (ec) and transient classes (tc). """

        # init
        sccs = tarjan(util.create_graph_dict(self.P))  # strongly connected components (sccs)
        ec = []  # a list to keep track of the ergodic classes
        tc = []  # a list to keep track of transient states
        prec = 10**(-10)  # precision used

        for scc in sccs:
            if abs(np.sum(self.P[np.ix_(scc, scc)]) - len(scc)) < prec:
                # no 'outgoing' probabilities: scc is an ergodic class
                ec.append(scc)
            else:  # scc is a transient connected component
                tc.append(scc)

        if self.verbose:
            if len(ec) > 1:
                print('P describes a Markov multi-chain.')
            else:
                print('P describes a Markov uni-chain.')

        return ec, tc

    @util.cache_property
    def I(self):
        """Identity matrix. """
        return np.eye(self.n)

    @util.cache_property
    def ones(self):
        """Matrix of ones. """
        return np.ones((self.n, self.n))

    @util.cache_property
    def pi(self):
        """Stationary distribution (if it exists). """

        if self.bUniChain:
            # stationary distribution exists
            Z = self.P - self.I
            Z[:, [0]] = np.ones((self.n, 1))
            # TODO: instead of inverse, solve a system of linear equations
            return la.inv(Z)[[0], :]
        else:
            raise Warning('Stationary distribution does not exist.')

    @util.cache_property
    def Pi(self):
        """Ergodic projector. """

        if self.bUniChain:
            # a unique stationary distribution
            return np.dot(np.ones((self.n, 1)), self.pi)

        # Markov multi-chain
        Pi = np.zeros((self.n, self.n))

        # transient (tr) states (st) preparation
        trStIdxs = [t for tc_sub in self.tc for t in tc_sub]
        nTrSt = len(trStIdxs)  # number of transient states
        ITrSt = np.eye(nTrSt)
        if self.bTransientStates:
            PTrSt = self.P[np.ix_(trStIdxs, trStIdxs)]  # transient part of transition matrix
            ITrSt = np.eye(nTrSt)  # identity matrix of transient size
            ProbTrtoErg = npla.inv(ITrSt - PTrSt)  # erg. prob. from tr. st. to erg. st.

        # fill Pi
        for e in self.ec:

            # ergodic classes
            idxsEc = np.ix_(e, e)
            Pi[idxsEc] = MarkovChain(self.P[idxsEc]).Pi  # note a uni-chain

            if self.bTransientStates:
                # transient states to ergodic classes
                trStToEcIdxs = np.ix_(trStIdxs, e)  # indexes from tr. to erg.
                PTtStToEc = self.P[trStToEcIdxs]  # transient part to ergodic class e
                Pi[trStToEcIdxs] = ProbTrtoErg.dot(PTtStToEc).dot(Pi[idxsEc])

        return Pi

    @util.cache_property
    def Z(self):
        """Fundamental matrix. """
        return la.inv(self.I - self.P + self.Pi)

    @util.cache_property
    def D(self):
        """Deviation matrix. """
        return self.Z - self.Pi

    @util.cache_property
    def M(self):
        """Mean first passage matrix. """

        # TODO: M is non-existent for multi-chains, maybe option to calculate per ergodic class?

        if len(self.ec) > 1 or len(self.tc) > 1:
            raise Exception('Mean first passage matrix does not exist for multi-chains.')

        dgMatrixD = np.diag(np.diag(self.D))
        dgMatrixPiInv = la.inv(np.diag(np.diag(self.Pi)))

        return (self.I - self.D + self.ones.dot(dgMatrixD)).dot(dgMatrixPiInv)

    @util.cache_property
    def V(self):
        """Variance first passage matrix. """

        # TODO: V is non-existent for multi-chains, maybe option to calculate per ergodic class?

        # TODO: maybe clean a bit

        if len(self.ec) > 1 or len(self.tc) > 1:
            raise Exception('Variance first passage matrix does not exist for multi-chains.')

        dgMatrixZ = np.diag(np.diag(self.Z))
        dgMatrixPiInv = la.inv(np.diag(np.diag(self.Pi)))
        dgMatrixZM = np.diag(np.diag(self.Z.dot(self.M)))
        term1 = self.M.dot(2*dgMatrixZ.dot(dgMatrixPiInv) - self.I)
        term2 = 2*(self.Z.dot(self.M) - self.ones.dot(dgMatrixZM))

        return term1 + term2 - self.M*self.M

    @util.cache_property
    def K(self):
        """Kemeny constant (based on D definition: so also defined for multi-chains). """
        return np.trace(self.D) + 1

    @util.cache_property
    def KDer(self):
        """Kemeny constant derivatives to the edges."""

        DSq = np.linalg.matrix_power(self.D, 2)
        KDer = np.zeros((self.n, self.n))

        for i in range(0, self.n):
            substrVal = np.dot(self.P[[i], :], DSq[:, [i]])
            for j in range(0, self.n):
                KDer[i, j] = DSq[j, i] - substrVal

        return KDer

    @util.cache_property
    def G(self):
        """Graph representation of the Markov chain. """
        return nx.from_numpy_matrix(self.P, create_using=nx.DiGraph())

    @util.cache_property
    def SCC(self):
        """Strongly connected components. """
        return self.ec + self.tc

    @util.cache_property
    def WCC(self):
        """Weakly connected components. """
        return tools.weakly_connected_components(self.P)

    @util.cache_property
    def nSCC(self):
        """Number of strongly connected components. """
        return len(self.SCC)

    def draw(self):
        """Draws the Markov chain. """

        # TODO: Incorporate Graphviz for better drawings (curved edges, etc.)

        fig, ax = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)
        fig.suptitle('Graph representation of the Markov chain')
        #pos = graphviz_layout(self.G, prog='dot')
        nodeSizes = 100 + self.pi[0]*1000 if self.bUniChain else 100
        nx.draw(self.G,
                #pos,
                with_labels=True,
                arrows=True,
                ax=ax,
                node_size=nodeSizes,
                edge_color=self.P[self.P > 0],
                edge_cmap=plt.get_cmap('Reds'),
                width=2,
                alpha=1)
        fig.tight_layout(rect=[0, 0, 1, 1])
        write_dot(self.G, 'graph.dot')  # only allowed when there are no self-loops in Python 2.7

    def order_edges_on_connectivity(self, existingEdgesOnly=True):
        """Order (existing) edges w.r.t. Kemeny constant derivatives. """

        orderedIdxs = np.argsort(self.KDer, axis=None)  # flattened version

        if existingEdgesOnly:
            existingIdxs = np.flatnonzero(self.P > 0)
            orderedIdxs = [x for x in orderedIdxs if x in existingIdxs]

        return np.unravel_index(orderedIdxs, self.P.shape)

    def get_most_connecting_edges(self, k, existingEdgesOnly=True):
        """Get the k (existing) edges with the smallest Kemeny constant
        derivatives. """

        # TODO: when the most connecting edge is a self-link, it may happen that KDA comes into a infinite loop. For example: when P = [[0,1,0],[1,0,0],[0,0,1]], KDer has the lowest value for self loop (3,3).

        orderedEdges = self.order_edges_on_connectivity(existingEdgesOnly)

        if len(orderedEdges[0]) < k:
            print ('Warning: The number of edges left to cut is < k. '
                   'Returned as many as possible.')
            k = len(orderedEdges[0])

        return tuple(x[:k] for x in orderedEdges)

    def get_edges_below_threshold(self, q, existingEdgesOnly=True):
        """Get the (existing) edges with Kemeny constant derivatives < q. """

        orderedEdges = self.order_edges_on_connectivity(existingEdgesOnly)

        # find the k edges with Kemeny constant derivative < q
        k = 0
        for (i, j) in zip(*orderedEdges):
            if self.KDer[i, j] >= q:
                break
            k += 1

        return tuple(x[:k] for x in orderedEdges)

    def plot(self, graphName=None, graphTitle='', markNodeIndexes=[],
             folderToSave='C:\\Mohsen\\2020-12-01-DataSetsEindhoven\\20210801Results\\', saveFormat='pdf', plotInfo=True,
             selectedColors=False):
        """Plot the Markov chain. """

        if graphName is None:  # check for graph name
            graphName = 'no_file_name_given'

        # init
        enoughSelectedColors = self.nSCC <= 8
        if selectedColors and enoughSelectedColors:
            vColors = ['limegreen', 'skyblue', 'wheat', 'lightgray', 'orange',
                       'green', 'lightcoral', 'lightpink']
        else:
            vColors = ['limegreen', 'skyblue', 'wheat', 'lightgray', 'orange',
                       'green', 'lightcoral', 'lightpink']

            # vColors = [x for x in colors.CSS4_COLORS.keys()[::-1]
            #            if x not in {'olive', 'lime', 'aqua'}]  # these colors are not recognized
        if self.nSCC > len(vColors):
            print('Warning: colors are being reused when plotting the MC.')
            vColors = self.nSCC*vColors

        graphDict = tools.create_graph_dictionary(self.P)

        # initialize graph
        graph = pydotplus.Dot(rankdir='LR',
                              graph_type='digraph',
                              layout='circo',
                              pad="0.1",
                              nodesep="0.5",
                              ranksep="2",
                              labelloc="t",
							  fontsize=16,
                              label=graphTitle)

        if plotInfo:
            info = str(self).replace(':', ';').replace('\n', '\l')
            infoNode = pydotplus.Node(info,
                                      fontname='Lucida Console',
                                      fontcolor='black',
                                      fontsize='10',
                                      shape='record',
                                      style="filled",
                                      fillcolor='white')
            graph.add_node(infoNode)

        # create node-objects for all ergodic classes and add nodes to graph
        nodes = list(range(self.n) ) # to save the references to all nodes-objects
        for i in range(self.nEc):  # for EC i
            for j in self.ec[i]:

                # create node j
                node_j = pydotplus.Node(str(j+1),
                                        fontcolor='black',
                                        fontsize='20',
                                        shape='circle',
                                        fixedsize='true',
                                        style='filled',
                                        fillcolor=vColors[i]
                                        )
                nodes[j] = node_j  # save reference to node
                graph.add_node(node_j)  # add node to graph

        # create node-objects for all transient s.c.c.'s and add nodes to graph
        for i in range(self.nTc):  # for transient scc i
            for j in self.tc[i]:

                # create node j
                node_j = pydotplus.Node(str(j+1),
                                        fontcolor='black',
                                        fontsize='20',
                                        shape='square',
                                        fixedsize='true',
                                        style="filled",
                                        fillcolor=vColors[i + self.nEc])
                nodes[j] = node_j  # save reference to node
                graph.add_node(node_j)  # add node to graph

        # add edges to graph
        for i in range(self.n):
            # add edges (i,j) for all j
            for j in graphDict[i]:
                graph.add_edge(pydotplus.Edge(nodes[i],
                                              nodes[j],
                                              penwidth=0.1+2*self.P[i][j]))

        for i in markNodeIndexes:
            nodes[i].set_style("\"bold,filled\"")
            nodes[i].set_color("red")
            nodes[i].set_label('<<U><B>' + nodes[i].get_name() + '</B></U>>')

        # save and plot graph
        graph.write(folderToSave + graphName + '.' + saveFormat,
                    format=saveFormat)
        #graph.write_png(folderToSave + graphName + '.png')
        # graph.write_jpg(folderToSave + graphName + '.jpg')
        # img = mpimg.imread(folderToSave + graphName + '.jpg')
        # # plt.figure(graphName)
        # plt.axis("off")
        # plt.imshow(img)


class Ranking(object):
    """Capture rankings methodologies for a Markov chain.

    Parameters
    ----------
    MC : MarkovChain
        The Markov chain in which we are interested.
    verbose : bool
        Decides information is print or not.

    Attributes
    ----------
    -

    Note
    ----
    Markov chain rankings are only calculated once. Once calculated, it is
    cached and the cache is used afterwards.
    """

    def __init__(self, MC, verbose=False):
        self.verbose = verbose
        self.MC = MC

    @util.cache_property
    def Google_PageRank_scores(self):
        """Calculate Google's PageRank scores with damping factor d = 0.85. """

        n = self.MC.n
        d = 0.85
        MCPageRank = MarkovChain(d*self.MC.P + (1-d)*np.ones((n, n))/n)

        return MCPageRank.pi

    @util.cache_property
    def Google_PageRank(self):
        """Calculate Google's PageRank ranking with damping factor d = 0.85. """

        return np.argsort(-self.Google_PageRank_scores)[0]

    # TODO: implement own WODES ranking method


if __name__ == "__main__":
    # can be used for testing

    MC = MarkovChain([[1, 0], [0, 1]])

    # print('pi = {}'.format(MC.pi))
    # print('K_P = {}'.format(MC.K))

    MC = MarkovChain('Courtois')
    mostConnectingEdges = MC.get_most_connecting_edges(3)

    q = -10
    edgesBelowThreshold = MC.get_edges_below_threshold(q, existingEdgesOnly=False)
    print(len(edgesBelowThreshold[0]) == np.sum(MC.KDer < q))

    q = -10
    edgesBelowThreshold = MC.get_edges_below_threshold(q, existingEdgesOnly=True)
    print(len(edgesBelowThreshold[0]) == len(set(np.flatnonzero(MC.KDer < q)).intersection(set(np.flatnonzero(MC.P > 0)))))

    # Test with old implementation
    MC = MarkovChain('Courtois')
    P = np.array(MC.P)

    print(la.norm(MCOld.calc_ergodic_proj(P) - MC.Pi))
    print( la.norm(MCOld.calc_deviation_matrix(P) - MC.D))
    print( la.norm(MCOld.calc_Kemeny_derivatives_exact(P) - MC.KDer))

    print(MC)

    MC.plot(graphName='Courtois')
    MC.plot(graphName='CourtoisMarked', markNodeIndexes=np.argsort(MC.pi[0])[-3:])


