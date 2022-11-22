# -*- coding: utf-8 -*-
"""
Created on 9/15/2018

Author: Joost Berkhout (CWI, email: j.berkhout@cwi.nl)

Description: Collection of some utilities to be used throughout package.
"""

import numpy as np


def create_graph_dict(A):
    """Find common graph Python structure based on an adjacency matrix A.

    Every (i,j): A(i,j)>0 is an edge by assumption.

    Parameters
    ----------
    A : np.array
      An adjacency matrix

    Returns
    -------
    graph : dict
        graph[i] gives a list of edges from node i
    """

    graph = {}

    for i in range(len(A)):
        graph[i] = np.where(A[i] > 0)[0].tolist()

    return graph


def cache_property(fn):
    """Decorator that caches a property (lazy-evaluation). """

    attr_name = '_' + fn.__name__

    @property
    def _cache_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _cache_property


if __name__ == "__main__":
    print('Is it a module or a script?')
