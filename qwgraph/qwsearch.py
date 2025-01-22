##############################################
##                  Imports                 ##
##############################################
# Maths
import numpy as np
import math 
from math import pi 
from scipy import signal
from scipy import linalg
# Graph
import networkx as nx 
# Utilities
import copy
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import warnings
from enum import Enum
import sys

from qwgraph import qwgraph as qwfast


class Polarity(Enum):
    Minus = 0
    Plus = 1


class AddressingType(Enum):
    EDGE = 0 # (u,v)
    VIRTUAL_EDGE = 1 # u
    NODE = 2 # u
    AMPLITUDE = 3 # (u,v)



class Scattering:
    def __init__(self, data, per_node=True):
        self.data = data
        self.per_node = per_node

    def _read_nodes(self, nodes):
        ret = [[] for i in nodes]
        if type(self.data) == type(list()):
            return self.data

        for i in range(len(nodes)):
            if type(self.data) == dict: # Dictionnary
                ret[i] = self.data[nodes[i]]
            else: # Function
                ret[i] = self.data(nodes[i])
        return ret

    def _read_degree(self, degree):
        ret = [[] for i in range(max(degree)+1)]
        if type(self.data) == type(list()):
            return self.data

        for i in degree:
            if type(self.data) == dict: # Dictionnary
                ret[i] = self.data[i]
            else: # Function
                ret[i] = self.data(i)
        return ret

    def _get_scatter(self, nodes, deg_max):
        Scatter = qwfast.Scattering()
        if self.per_node:
            Scatter.set_type(3, self._read_nodes(nodes))
        else:
            Scatter.set_type(2, self._read_degree(deg_max))
        return Scatter


class UnitaryOp:
    def __init__(self, target, U, _type=AddressingType.EDGE):
        self.U = U
        self._type = _type
        self.target = target


class Operation:
    def __init__(self, _type, obj, name = ""):
        assert(_type in ["scattering", "coin", "unitary"])
        self._type = _type
        self.obj = obj
        self.name = name
    def __repr__(self):
        return str(self)
    def __str__(self):
        return "{}({})".format(self.name, self._type)






###############################################
##                  QW Class                 ##
###############################################

class QWSearch:
    """ 
    The Quantum Walk based search class. An instance of this class will be a Quantum Walk on a given graph.
    Methods are provided to modify and access the QW state and to run the QWSearch.

    Both the Quantum Walk and searching process are described in https://arxiv.org/abs/2310.10451

    Attributes:
        step (int): The current step (or epoch). Modifying this attribute will only change the step column of the `search` method.

    Args:
        graph (networkx.Graph): The graph on which the QW will be defined.
        search_nodes (bool, optional): If True, the graph will be starified and the QW will be tuned to search nodes instead of edges. 

    
    """

    ######################
    ### Init functions ###
    ######################
    def __init__(self, graph, search_nodes=False):
        self.__search_nodes = search_nodes
        
        self.__G = deepcopy(graph)
        
        if self.__search_nodes:
            self.__virtual_edges = self.__starify()
        else:
            self.__virtual_edges = {}
        
        self.__edges = list(self.__G.edges()) # List of edges
        self.__nodes = list(self.__G.nodes()) # List of nodes
        self.__E = len(self.__edges) # Number of edges
        self.__N = len(self.__nodes) # Number of nodes
        self.__index = {self.__edges[i]:i for i in range(len(self.__edges))} # Index for edges
        self.__nodes_index = {self.__nodes[i]:i  for i in range(self.__N)} # Index for nodes

        if nx.bipartite.is_bipartite(self.__G):
            color = nx.bipartite.color(self.__G) # Coloring
        else:
            color = {self.__nodes[i]:i for i in range(len(self.__nodes))} # Coloring

        self.__polarity = {}
        for (u,v) in self.__edges:
            self.__polarity[(u,v)] = ("-" if color[u]<color[v] else "+")
            self.__polarity[(v,u)] = ("+" if color[u]<color[v] else "-")

        self.__initalize_rust_object()        
        

    def __initalize_rust_object(self):
        self.__amplitude_labels = [""]*2*self.__E
        wiring = [] # For any amplitude self.state[i], says to which node it is connected. Important for the scattering.
        k = 0
        for (i,j) in self.__edges:
            edge_label = str(i) + "," + str(j)
            if self.__polarity[(i,j)]=="-":
                wiring.append(self.__nodes_index[i])
                wiring.append(self.__nodes_index[j])
            else:
                wiring.append(self.__nodes_index[j])
                wiring.append(self.__nodes_index[i])
            self.__amplitude_labels[k] = "$\psi_{"+edge_label+"}^-$"
            self.__amplitude_labels[k+1] = "$\psi_{"+edge_label+"}^+$"
            k+=2
        

        self.__qwf = qwfast.QWFast(wiring,self.__N,self.__E)
        self.__around_nodes_indices = qwfast._get_indices_around_nodes(self.__E,self.__N,wiring)

        self.reset()

    def __starify(self):
        nodes = copy.deepcopy(self.__G.nodes())
        s = {}
        for i in nodes:
            self.__G.add_edge(i,f"new_node{i}")
            s[i] = (i,f"new_node{i}")
        return s







    #####################
    ### Getters graph ###
    #####################
    def nodes(self):
        """ Returns the list of nodes. Convenient when declaring which nodes are marked.

        Args:
            real_only (boolean, optional): If True, then only the real nodes will be returned. Has no effect when search_nodes==False.

        Returns:
            (list of node): The list of nodes of the underlying graph.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.nodes()
            [0, 1, 2, 3]
            
        """
        return deepcopy(self.__nodes)

    def edges(self):
        """ Returns the list of edges. Convenient when declaring which edges are marked.

        Returns:
            (list of edge): The list of edges of the underlying graph.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.edges()
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            
        """
        return deepcopy(self.__edges)

    def graph(self):
        """ Returns the underlying graph.

        Returns:
            (networkx.Graph): The underlying graph.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.graph()
            <networkx.classes.graph.Graph at 0x7bd045d53c70>
            
        """
        return deepcopy(self.__G)

    def virtual_edges(self):
        """ Returns a dictionnary that associates its virtual edge to each node. This dictionnary is empty when the object has been built with `search_nodes==False` since there are no virtual edges in that case.

        Returns:
            (dict): A dictionnary {node: edge} that associates each node to its virtual edge.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.virtual_edges()
            {}
            >>> qw = QWSearch(nx.complete_graph(4),search_nodes=True)
            >>> qw.virtual_edges()
            {0: (0, 'new_node0'),
             1: (1, 'new_node1'),
             2: (2, 'new_node2'),
             3: (3, 'new_node3')}
            
        """
        return deepcopy(self.__virtual_edges)








    #############################
    ### Getters/Setters state ###
    #############################
    def set_polarity(self, polarity):
        b = True
        opp = lambda x:"-" if x=="+" else "+"
        for ((u,v),val) in polarity:
            if polarity[(u,v)] not in ["+","-"]:
                b = False
                break

            if (v,u) in polarity and polarity[(v,u)] != opp(polarity[(u,v)]):
                b = False
                break

        if b:
            for ((u,v),val) in polarity:
                self.__polarity[(u,v)] = val
                self.__polarity[(v,u)] = opp(val)
            self.__initalize_rust_object() 
        else:
            print("Warning: Polarity values not valid !", file=sys.stderr)

    def __get_index(self, pos, _type=AddressingType.EDGE):
        if _type == AddressingType.EDGE:
            index = self.__index[pos]
            return [2*index, 2*index+1]
        if _type == AddressingType.VIRTUAL_EDGE:
            #assert(self.__search_nodes and pos in self.__virtual_edges.keys())
            edge = self.__virtual_edges[pos]
            index = self.__index[edge]
            return [2*index, 2*index+1]
        if _type == AddressingType.NODE:
            return deepcopy(self.__around_nodes_indices[self.__nodes_index[pos]])
        if _type == AddressingType.AMPLITUDE:
            (u,v) = pos
            if (u,v) in self.__index:
                edge = (u,v)
            else:
                edge = (v,u)
            index = self.__index[edge]
            return [2*index] if self.__polarity[pos]=="-" else [2*index+1]


    def polarity(self, amplitudes):
        indices = {p:self.__get_index(p,AddressingType.AMPLITUDE)[0] for p in amplitudes}
        return {p:"-" if indices[p]%2==0 else "+" for p in amplitudes}

    def label(self, positions, _type=AddressingType.EDGE):
        indices = {p:[i for i in self.__get_index(p,_type)] for p in positions}
        return {p:[self.__amplitude_labels[i] for i in indices[p]] for p in positions}


    def state(self, positions, _type=AddressingType.EDGE):
        """ Return the amplitudes of one/several/every edges.

        For an edge (u,v), the amplitudes $\psi_{u,v}^+$ and $\psi_{u,v}^-$ will be returned in the form of a numpy array.

        Args:
            edges (list, optional): The list of edges for which we want to extract the amplitudes. If None, all the edges are extracted.

        Returns:
            (dict): A dictionnary edge:amplitudes where the amplitudes are complex numpy arrays of dimension 2.
        
        Examples:
            >>> qw = QWSearch(nx.cycle_graph(4))
            >>> qw.state()
            {(0, 1): array([0.35355339+0.j, 0.35355339+0.j]),
             (0, 3): array([0.35355339+0.j, 0.35355339+0.j]),
             (1, 2): array([0.35355339+0.j, 0.35355339+0.j]),
             (2, 3): array([0.35355339+0.j, 0.35355339+0.j])}
            >>> qw.state(qw.edges()[0:2])
            {(0, 1): array([0.35355339+0.j, 0.35355339+0.j]),
             (0, 3): array([0.35355339+0.j, 0.35355339+0.j])}
        """
        indices = {p:[i for i in self.__get_index(p,_type)] for p in positions}
        return {p:np.array([self.__qwf.state[i] for i in indices[p]],dtype=complex) for p in positions}

    def set_state(self, new_state):
        """ Change the inner state (i.e. the amplitudes for every edges).

        For an edge (u,v), the amplitudes $\psi_{u,v}^+$ and $\psi_{u,v}^-$ will be modified according to the argument.
        If the new state is not normalized, this method will automatically normalize it.

        Args:
            new_state (dict): A dictionnary of the form edge: amplitudes. Amplitudes must be numpy arrays or lists of dimension 2.
        
        Examples:
            >>> qw = QWSearch(nx.cycle_graph(4))
            >>> qw.state()
            {(0, 1): array([0.35355339+0.j, 0.35355339+0.j]),
             (0, 3): array([0.35355339+0.j, 0.35355339+0.j]),
             (1, 2): array([0.35355339+0.j, 0.35355339+0.j]),
             (2, 3): array([0.35355339+0.j, 0.35355339+0.j])}
            >>> qw.set_state({edge:[2,1j] for edge in qw.edges()})
            >>> qw.state()
            {(0, 1): array([0.4472136+0.j       , 0.       +0.2236068j]),
             (0, 3): array([0.4472136+0.j       , 0.       +0.2236068j]),
             (1, 2): array([0.4472136+0.j       , 0.       +0.2236068j]),
             (2, 3): array([0.4472136+0.j       , 0.       +0.2236068j])}
        """
        s = np.sqrt(sum([abs(new_state[e][0])**2 + abs(new_state[e][1])**2 for e in new_state]))
        state = np.array([0]*2*self.__E,dtype=complex)
        for i in range(self.__E):
            state[2*i] = new_state[self.__edges[i]][0]/s
            state[2*i+1] = new_state[self.__edges[i]][1]/s
        self.__qwf.state = state

    def proba(self, searched):
        """ Returns the probability to measure on of the searched element.

        Args:   
            searched (list of edge): The list of marked edges. Every element of the list must be an edge label (all of them are listed in `qw.edges`).

        Returns:
            (float): The probability of measuring any of the marked edges.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.get_proba([qw.edges()[0]])
            0.1666666666666667
            >>> qw.get_proba([qw.edges()[0],qw.edges()[1]])
            0.3333333333333334
            >>> qw.get_proba(qw.edges())
            1.
        """
        return self.__qwf.get_proba([self.__get_edge_index(i)[1] for i in searched])

    def reset(self):
        """ Reset the state to a diagonal one and reset the current step to 0.
        Do not return anything.

        Examples:
            >>> qw = QWSearch(nx.cycle_graph(4))
            >>> qw.state()
            {(0, 1): array([0.35355339+0.j, 0.35355339+0.j]),
             (0, 3): array([0.35355339+0.j, 0.35355339+0.j]),
             (1, 2): array([0.35355339+0.j, 0.35355339+0.j]),
             (2, 3): array([0.35355339+0.j, 0.35355339+0.j])}
            >>> qw.set_state({edge:[2,1j] for edge in qw.edges()})
            >>> qw.state()
            {(0, 1): array([0.4472136+0.j       , 0.       +0.2236068j]),
             (0, 3): array([0.4472136+0.j       , 0.       +0.2236068j]),
             (1, 2): array([0.4472136+0.j       , 0.       +0.2236068j]),
             (2, 3): array([0.4472136+0.j       , 0.       +0.2236068j])}
            >>> qw.reset()
            >>> qw.state()
            {(0, 1): array([0.35355339+0.j, 0.35355339+0.j]),
             (0, 3): array([0.35355339+0.j, 0.35355339+0.j]),
             (1, 2): array([0.35355339+0.j, 0.35355339+0.j]),
             (2, 3): array([0.35355339+0.j, 0.35355339+0.j])}

        """
        self.__qwf.reset()










    #######################
    ### Read operations ###
    #######################
    def _read_coin(self, C):
        Coin = qwfast.Coin()
        if type(C) == type(dict()): # Dictionnary
            Coin.set_micro([C[e] for e in self.edges()])
        elif len(np.shape(C)) == 2: # One matrix
            Coin.set_macro(C)
        else:
            raise "Wrong type or dimension for the coin"
        return Coin


    def _read_scatter(self, S):
        Scatter = qwfast.Scattering()
        if S == "cycle":
            Scatter.set_type(0, [])
        elif S == "grover":
            Scatter.set_type(1, [])
        elif type(S) == type(Scattering([])):
            Scatter = S._get_scatter(self.__nodes, list(set(list(dict(nx.degree(self.__G)).values()))))

        else:
            raise "Wrong argument for the scattering"
        return Scatter

    def _read_unitary(self, U):
        pos = np.reshape([self.__get_index(i, U._type) for i in U.target], (-1,))
        Unitary = qwfast.UnitaryOp(pos, U.U)
        return Unitary

    def _read_operation(self, op):
        ret = qwfast.OperationWrapper()
        if op._type == "scattering":
            ret.set_to_scattering(self._read_scatter(op.obj))
        elif op._type == "coin":
            ret.set_to_coin(self._read_coin(op.obj))
        elif op._type == "unitary":
            ret.set_to_unitary(self._read_unitary(op.obj))
        else:
            raise f"Wrong type for operation {op}"
        return ret

    def read_pipeline(self, pipeline):
        l = []
        for op in pipeline:
            l.append(self._read_operation(op))
        return l








    



    


    def run(self, pipeline, ticks=1):
        """ Run the simulation with coin `C`, oracle `R` for ticks steps and with searched elements `search`.
        Nothing will be returned but the inner state will be modified inplace.

        Args:
            C (numpy.array of complex): The coin defined as a 2x2 numpy array of complex.
            R (numpy.array of complex): The oracle defined as a 2x2 numpy array of complex.
            searched (list, optional): The list of marked elements. "elements" here means nodes if search_nodes was true when building the object, and means edges otherwise.
            ticks (int, optional): The number of time steps.

        Examples:
            >>> qw = QWSearch(nx.cycle_graph(6))
            >>> qw.set_state({edge:([1,0] if edge==qw.edges()[len(qw.edges())//2] else [0,0]) for edge in qw.edges()})
            >>> [qw.get_proba([e]) for e in qw.edges()]
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            >>> qw.run(coins.H,coins.I,ticks=3)
            >>> [np.round(qw.get_proba([e]),3) for e in qw.edges()]
            [0.0, 0.25, 0.625, 0.0, 0.125, 0.0]
        """
        self.__qwf.run(self.read_pipeline(pipeline),ticks)


    def run_with_trace(self, pipeline, ticks=1):
        """ Run the simulation with coin `C`, oracle `R` for ticks steps and with searched elements `search`.
        Nothing will be returned but the inner state will be modified inplace.

        Args:
            C (numpy.array of complex): The coin defined as a 2x2 numpy array of complex.
            R (numpy.array of complex): The oracle defined as a 2x2 numpy array of complex.
            searched (list, optional): The list of marked elements. "elements" here means nodes if search_nodes was true when building the object, and means edges otherwise.
            ticks (int, optional): The number of time steps.

        Examples:
            >>> qw = QWSearch(nx.cycle_graph(6))
            >>> qw.set_state({edge:([1,0] if edge==qw.edges()[len(qw.edges())//2] else [0,0]) for edge in qw.edges()})
            >>> [qw.get_proba([e]) for e in qw.edges()]
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            >>> qw.run(coins.H,coins.I,ticks=3)
            >>> [np.round(qw.get_proba([e]),3) for e in qw.edges()]
            [0.0, 0.25, 0.625, 0.0, 0.125, 0.0]
        """
        l = self.__qwf.run(self.read_pipeline(pipeline),ticks)
        l = [[i] + l[i] for i in range(len(l))]
        return pd.DataFrame(l,columns=["steps"] + self.__amplitude_labels)



    def search(self, pipeline, searched=[], ticks=1, _type=AddressingType.EDGE):
        """ Run the simulation with coin `C`, oracle `R` for ticks steps and with searched elements `searched`.

        This method does the same thing than `run`, but returns the probability of success at every steps. For every marked element m, the probability of measuring m at every step is returned.
        
        Args:
            C (numpy.array of complex): The coin defined as a 2x2 numpy array of complex.
            R (numpy.array of complex): The oracle defined as a 2x2 numpy array of complex.
            searched (list, optional): The list of marked elements. "elements" here means nodes if search_nodes was true when building the object, and means edges otherwise.
            ticks (int, optional): The number of time steps.
            progress (bool, optional): If True, a tqdm progress bar will be displayed.

        Returns:
            (pandas.DataFrame): A dataframe containing probabilities fo measuring marked positions. The column "step" denote the step number (or epoch) of the dynamic. For each marked element `m`, the column `m` denotes the probability of measuring `m` at any given step. The column `p_succ` denotes the probability of measuring any marked elements and is essentially the sum of all the other colmuns excepted "step".

        Examples:
            >>> qw = QWSearch(nx.complete_graph(100))
            >>> print(qw.search(coins.H,coins.I,searched=qw.edges()[0:4],ticks=10))
                step    p_succ    (0, 1)    (0, 2)    (0, 3)    (0, 4)
            0      0  0.000808  0.000202  0.000202  0.000202  0.000202
            1      1  0.003880  0.000994  0.000978  0.000962  0.000946
            2      2  0.009113  0.002467  0.002337  0.002214  0.002095
            3      3  0.013043  0.003875  0.003441  0.003044  0.002683
            4      4  0.013292  0.004433  0.003617  0.002918  0.002324
            5      5  0.010471  0.003820  0.002892  0.002162  0.001596
            6      6  0.007487  0.002620  0.002011  0.001579  0.001277
            7      7  0.005653  0.001645  0.001455  0.001324  0.001228
            8      8  0.004657  0.001321  0.001212  0.001107  0.001017
            9      9  0.004065  0.001494  0.001105  0.000824  0.000641
            10    10  0.004440  0.001913  0.001226  0.000784  0.000517
            >>> qw = QWSearch(nx.complete_graph(100),search_nodes=True)
            >>> print(qw.search(coins.H,coins.I,searched=qw.nodes()[0:4],ticks=10))
                step    p_succ         0         1         2         3
            0      0  0.000792  0.000198  0.000198  0.000198  0.000198
            1      1  0.000746  0.000198  0.000190  0.000182  0.000175
            2      2  0.003557  0.000978  0.000917  0.000859  0.000803
            3      3  0.000890  0.000280  0.000237  0.000201  0.000172
            4      4  0.000097  0.000023  0.000020  0.000023  0.000031
            5      5  0.000320  0.000072  0.000079  0.000084  0.000086
            6      6  0.004178  0.001147  0.001087  0.001014  0.000930
            7      7  0.002613  0.000864  0.000713  0.000577  0.000459
            8      8  0.002197  0.000817  0.000607  0.000446  0.000327
            9      9  0.002605  0.000897  0.000695  0.000554  0.000458
            10    10  0.000085  0.000036  0.000022  0.000015  0.000012

        """

        indices = np.reshape([self.__get_index(i, _type) for i in searched], (-1,))
        p = self.__qwf.search(self.read_pipeline(pipeline),ticks,indices)

        return p

    def get_unitary(self, pipeline, searched=[], dataframe=False, progress=False, _type=AddressingType.EDGE):
        """ For a given coin, oracle and set of searched edges, compute and return the unitary U coresponding to one step of the QW.

        This method **do not** change the state of the QW.

        Args:
            C (numpy.array of complex): The coin defined as a 2x2 numpy array of complex.
            R (numpy.array of complex): The oracle defined as a 2x2 numpy array of complex.
            searched (list, optional): The list of marked elements. "elements" here means nodes if search_nodes was true when building the object, and means edges otherwise.
            dataframe (bool, optional): If True, the result will be a pandas dataframe instead of a numpy array. 
            progress (bool, optional): If True, a tqdm progress bar will be displayed.

        Returns:
            (numpy array or pandas dataframe): The unitary operator coresponding to one step of the dynamic. If dataframe is set to True, a pandas dataframe will be returned instead.
        
        Examples:
            >>> qw = QWSearch(nx.cycle_graph(3))
            >>> qw.get_unitary(coins.H,coins.I)
            array([[ 0.        +0.j,  0.        +0.j,  0.70710678+0.j,
                     0.70710678+0.j,  0.        +0.j,  0.        +0.j],
                   [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                     0.        +0.j,  0.70710678+0.j,  0.70710678+0.j],
                   [ 0.70710678+0.j,  0.70710678+0.j,  0.        +0.j,
                     0.        +0.j,  0.        +0.j,  0.        +0.j],
                   [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                     0.        +0.j,  0.70710678+0.j, -0.70710678+0.j],
                   [ 0.70710678+0.j, -0.70710678+0.j,  0.        +0.j,
                     0.        +0.j,  0.        +0.j,  0.        +0.j],
                   [ 0.        +0.j,  0.        +0.j,  0.70710678+0.j,
                    -0.70710678+0.j,  0.        +0.j,  0.        +0.j]])
        """
        old_state = copy.deepcopy(self.__qwf.state)
        U = []
        for i in (tqdm(range(2*self.__E),ncols=100)) if progress else (range(2*self.__E)):
            self.__qwf.state = np.array([int(i==j) for j in range(2*self.__E)],dtype=complex)
            self.run(pipeline, ticks=1, searched=searched, _type=_type)
            U.append(copy.deepcopy(self.__qwf.state))
        self.__qwf.state = old_state
        U = np.array(U,dtype=complex).transpose()
        if dataframe:
            df = pd.DataFrame(U, index=self.__amplitude_labels, columns=self.__amplitude_labels)
            return df
        else:
            return U

    def get_T_P(self, pipeline, searched=[], waiting=10, _type=AddressingType.EDGE):
        """ Computes the hitting time and probability of success for a given QW. 

        The waiting parameter is used to accumalate informations about the signal (recommended to be at least 10).

        In details, this algorithm look at the time serie of the probability of success $p(t)$. 
        At any time step $t$, we define $T_{max}(t) = \\underset{{t' \\leq t}}{\\mathrm{argmax }}\\; p(t')$ and $T_{min}(t) = \\underset{{t' \\leq t}}{\\mathrm{argmin }} \\; p(t')$.
        
        The algorithms computes the series $p(t)$, $T_{max}(t)$, $T_{min}(t)$ and stop when it encounters `t>waiting` such that $p(t)<\\frac{p\\left(T_{max}(t)\\right)+p\\left(T_{max}(t)\\right)}{2}$. 
        It then returns $T_{max}(t), p\\left(T_{max}(t)\\right)$.

        **Warning:** This function will reset the state of the QW.

        Args:
            C (numpy.array of complex): The coin defined as a 2x2 numpy array of complex.
            R (numpy.array of complex): The oracle defined as a 2x2 numpy array of complex.
            searched (list, optional): The list of marked elements. "elements" here means nodes if search_nodes was true when building the object, and means edges otherwise.
            waiting (int, optional): The waiting time for the algorithm. Must be smaller than the hitting time.

        Returns:
            (int*float): T:int,P:float respectively the hitting time and probability of success.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(100))
            >>> qw.get_T_P(coins.X,-coins.X,searched=qw.edges()[0:4])
            (28, 0.9565191408575295)
        """

        self.reset()
        indices = np.reshape([self.__get_index(i, _type) for i in searched], (-1,))
        ret = self.__qwf.carac(self.read_pipeline(pipeline),indices,waiting)
        self.reset()
        return ret
