# -*- coding: utf-8 -*-
"""
Created on 12/21/2018

Author: Joost Berkhout (CWI, email: j.berkhout@cwi.nl)

Description: A bundling of tools.
"""

from __future__ import division
import pandas as pd
import numpy as np
import os
from scipy import sparse


def data_loader(name='Courtois'):
    """Based on name, loads the numpy transition matrix. """

    path = os.path.dirname(__file__) + '\\data\\' + name + '.csv'

    return pd.read_csv(path, sep=',', header=None).values


def normalize_rows(A, verbose=False):
    """Normalizes the rows of matrix A.

    When row sum is zero, the diagonal element is set to 1.
    """

    # init
    sumRows = np.sum(A, axis=1)
    idxsZeroRowSums = np.argwhere(sumRows==0)

    if len(idxsZeroRowSums) > 0:
        # set diagonals for zero rows to 1
        if verbose: print('Some row sums are zero, '
                          'corresponding diagonals set to 1.')
        modifiedA = np.array(A)
        modifiedA[idxsZeroRowSums, idxsZeroRowSums] = 1
        return modifiedA/np.sum(modifiedA, axis=1)[np.newaxis].transpose()
    else:
        return A/sumRows[np.newaxis].transpose()


def normalize_rows_special(A):
    """Normalizes the rows of matrix A while ensuring that the relative edge
    weights remain the same, i.e., let P be the normalized version of A, then:

    P(i, j) / P(k, l) = A(i, j) / A(k, l), for all i != j and k != l.

    Essentially, the random walk is "following the edge weights".
    """

    scaling = 1.0*np.max(np.sum(A, axis=1))
    P = A/scaling
    np.fill_diagonal(P, 1 - np.sum(P, axis=1))

    return P


def weakly_connected_components(A):
    """Given ndarray A, this function finds the weakly connected components. """

    nrWCC, WCC = sparse.csgraph.connected_components(sparse.csr_matrix(A))

    return [np.flatnonzero(WCC == n) for n in range(nrWCC)]


def create_graph_dictionary(A):
    """Based on a given A, this function returns a graph in the common python
    structure: dictionaries. Every (i,j): A(i,j)>0 is an edge by assumption."""

    graph = {}
    for i in range(len(A)):
        graph[i] = np.where(A[i] > 0)[0].tolist()

    return graph


if __name__ == "__main__":

    A = np.array([[0, 0, 0], [.5, .25, .25], [.33, .33, .33]])
    print(normalize_rows(A))

    A = np.array([[0, 0, 0], [.5, .25, .25], [0, 0, 0]])
    print( normalize_rows(A))

    A = np.array([[0, 0, 0.1], [100, .25, .25], [0, 1, 2]])
    print( normalize_rows(A))

    A = np.array([[0, 10000, 1], [7000, 0, 0], [1, 0, 0]])
    ANormalized = normalize_rows_special(A)
    print(ANormalized)

