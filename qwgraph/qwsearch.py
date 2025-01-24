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


_X = np.array([[0,1],[1,0]],dtype=complex)

class AddressingType(Enum):
    """ Adressing types used for the QWSearch.
    In order to mark amplitudes for an oracle, or measure the probability to be somewhere and other similar applications,
    we need an convenient way to address amplitudes.
    This class represents the four ways to address amplitudes:
        - AddressingType.EDGE : Expect edges and assign to each edge its two amplitudes.
        - AddressingType.VIRTUAL_EDGE : Expect nodes and assign to each node the two amplitudes of the corresponding virtual edge.
        - AddressingType.NODE : Expect nodes and assign to each node the amplitudes around it.
        - AddressingType.AMPLITUDES : Expect directed edge and assign to each directed edge its amplitude.

    Examples:
        >>> qw = QWSearch(nx.complete_graph(4))
        >>> qw.label([(0,1)], _type = AddressingType.AMPLITUDE)
        {(0, 1): ['$\\psi_{0,1}^-$']}
        >>> qw.label([(1,0)], _type = AddressingType.AMPLITUDE)
        {(1, 0): ['$\\psi_{0,1}^+$']}
        >>> qw.label([(0,1)], _type = AddressingType.EDGE)
        {(0, 1): ['$\\psi_{0,1}^-$', '$\\psi_{0,1}^+$']}
        >>> qw.label([1], _type = AddressingType.NODE)
        {1: ['$\\psi_{0,1}^+$', '$\\psi_{1,2}^-$', '$\\psi_{1,3}^-$']}
        >>> qw = QWSearch(nx.complete_graph(4), starify=True)
        >>> qw.label([1], _type = AddressingType.VIRTUAL_EDGE)
        {1: ['$\\psi_{1,*1}^-$', '$\\psi_{1,*1}^+$']}
    """
    EDGE = 0 # (u,v)
    VIRTUAL_EDGE = 1 # u
    NODE = 2 # u
    AMPLITUDE = 3 # (u,v)


class Instruction(Enum):
    SCATTERING = 0
    COIN = 1
    UNITARY = 2
    PROBA = 3

class PipeLine(list): 
    """ Pipeline class for QWSearch
    This class is used to give instruction to the QWSearch. It inherits from list and is basically a list of instructions to the walk.
    Instructions are executed by the walk in order.
    There are four types of instruction:
        - Instruction.SCATTERING : An operation around the nodes.
        - Instruction.COIN : An operation on the edges.
        - Instruction.UNITARY : An operation on arbitrary amplitudes.
        - Instruction.PROBA : An instruction to extract the proba of being at some positions.
    Addition between pipelines and multiplication by an integer are implemented.

    Attributes:
        addressing_type (AddressingType): The default way to address amplitudes.

    Args:
        addressing_type (AddressingType, optional): The default way to address amplitudes.
    """
    def __init__(self, addressing_type=AddressingType.EDGE):
        super().__init__([])
        self.addressing_type=addressing_type
    def __repr__(self):
        l = [str(dic["instruction"].name) if "name" not in dic or dic["name"] == None else "{}({})".format(dic["instruction"].name,dic["name"]) for dic in self]
        return " -> ".join(l)

    def __add__(self, other):
        p = PipeLine(addressing_type=self.addressing_type)
        p.extend(self)
        p.extend(other)
        return p
    def __mul__(self,other):
        p = PipeLine(addressing_type=self.addressing_type)
        p.extend(list(self)*other)
        return p

    def _read_entry(self, dic, qw):
        op = qwfast.OperationWrapper()
        if dic["instruction"] == Instruction.COIN:
            Coin = qwfast.Coin()
            if type(dic["coin"]) == type(dict()): # Dictionnary
                Coin.set_micro([dic["coin"][e] for e in qw._edges])
            elif len(np.shape(dic["coin"])) == 2: # One matrix
                Coin.set_macro(dic["coin"])
            else:
                raise "Wrong type or dimension for the coin"
            op.set_to_coin(Coin)
            return op

        if dic["instruction"] == Instruction.PROBA:
            pos = np.reshape([qw._get_index(i, dic["addressing_type"]) for i in dic["targets"]], (-1,))
            op.set_to_proba(pos)
            return op

        if dic["instruction"] == Instruction.UNITARY:
            pos = np.reshape([qw._get_index(i, dic["addressing_type"]) for i in dic["targets"]], (-1,))
            U = dic["unitary"]
            if type(U) == type(lambda x:x):
                U = U(len(pos))
            Unitary = qwfast.UnitaryOp(pos, U)
            op.set_to_unitary(Unitary)
            return op


        if dic["instruction"] == Instruction.SCATTERING:
            Scatter = qwfast.Scattering()
            if dic["mode"]=="global":
                if dic["scattering"] == "cycle":
                    Scatter.set_type(0, [])
                elif dic["scattering"] == "grover":
                    Scatter.set_type(1, [])
                else:
                    raise "Scattering not recognized"
            elif dic["mode"]=="node":
                data = [[] for i in qw._nodes]
                for i in range(len(qw._nodes)):
                    data[i] = dic["scattering"][qw._nodes[i]]

                Scatter.set_type(3, data)

            elif dic["mode"]=="degree":
                data = [[] for i in range(max(qw._degrees)+1)]
                for i in qw._degrees:
                    data[i] = dic["scattering"](i)

                Scatter.set_type(3, data)

            else:
                raise "Wrong argument for the scattering"
            op.set_to_scattering(Scatter)
            return op


    def _read(self, qw):
        return [self._read_entry(dic,qw) for dic in self]

    def add_unitary(self, targets, unitary, addressing_type=None, name=None):
        """ Add an arbitrary unitary application to the pipeline.
        Args:
            targets (list): The list of targets.
            unitary (np.array): The matrix of the unitary operator.
            addressing_type (AddressingType, optional): The way the targets are addressed (edges, nodes, virtual edges or amplitudes). If set to None, this parameter will take the value of the class attribute addressing_type.
            name (str, optional): A name of the operation. As no effect on the dynamic but shows when printing the pipeline.
        
        Examples:
            >>> pipeline = PipeLine()
            >>> pipeline.add_unitary([(0,1)], -np.eye(2), addressing_type=AddressingType.EDGE, name="oracle on an edge")
            >>> pipeline.add_unitary([0], -np.eye(2), addressing_type=AddressingType.NODE, name="oracle around a node")
            >>> print(pipeline)
            UNITARY(oracle on an edge) -> UNITARY(oracle around a node)
        """
        dic = {"instruction":Instruction.UNITARY, "targets":targets, "unitary":unitary, "addressing_type":addressing_type}
        if name != None:
            dic["name"] = str(name)
        if addressing_type == None:
            dic["addressing_type"] = self.addressing_type
        self.append(dic)

    def add_proba(self, targets, addressing_type=None, name=None):
        """ Add an extraction of the probability of being on one of the targets.
        Args:
            targets (list): The list of targets.
            addressing_type (AddressingType, optional): The way the target is addressed (edges, nodes, virtual edges or amplitudes). If set to None, this parameter will take the value of the class attribute addressing_type.
            name (str, optional): A name of the operation. As no effect on the dynamic but shows when printing the pipeline.
        
        Examples:
            >>> pipeline = PipeLine()
            >>> pipeline.add_proba([(0,1)], addressing_type=AddressingType.EDGE, name="proba on an edge")
            >>> pipeline.add_proba([0], addressing_type=AddressingType.NODE, name="proba around a node")
            >>> print(pipeline)
            PROBA(proba on an edge) -> PROBA(proba around a node)
        """
        dic = {"instruction":Instruction.PROBA, "targets":targets, "addressing_type":addressing_type}
        if name != None:
            dic["name"] = str(name)
        if addressing_type == None:
            dic["addressing_type"] = self.addressing_type
        self.append(dic)

    def add_coin(self, coin, name=None):
        """ Add a coin operation to the pipeline.
        Coin operations are operations on the edges.
        Args:
            coin (np.array or dict): The two by two matrix of the coin. Can alternatively be a dictionnary {edge : coin} for a coin dependent of the edge.
            name (str, optional): A name of the operation. As no effect on the dynamic but shows when printing the pipeline.
        
        Examples:
            >>> qw = QWSearch(nx.complete_graph(4), True)
            >>> pipeline = PipeLine()
            >>> pipeline.add_coin(np.eye(2), name="identity coin")
            >>> pipeline.add_coin({e : np.eye(2) * np.exp(1j * np.random.random()) for e in qw.edges()}, name="coin different for each edge")
            >>> print(pipeline)
            COIN(identity coin) -> COIN(coin different for each edge)
        """
        dic = {"instruction":Instruction.COIN, "coin":coin}
        if name != None:
            dic["name"] = str(name)
        self.append(dic)

    def add_scattering(self, scattering, name=None):
        """ Add a scattering operation to the pipeline.
        Scattering operations are operations around the nodes.
        Args:
            scattering (str): "cycle" for a cycle around the node or "grover" for a Grover diffusion around the node.
            name (str, optional): A name of the operation. As no effect on the dynamic but shows when printing the pipeline.
        
        Examples:
            >>> pipeline = PipeLine()
            >>> pipeline.add_scattering("grover", name="Grover diffusion")
            >>> pipeline.add_scattering("cycle", name="cycle")
            >>> print(pipeline)
            SCATTERING(Grover diffusion) -> SCATTERING(cycle)
        """
        dic = {"instruction":Instruction.SCATTERING, "scattering":scattering, "mode":"global"}
        if name != None:
            dic["name"] = str(name)
        self.append(dic)

    def add_scattering_by_node(self, scattering, name=None):
        """ Add a different scattering operation for each node to the pipeline.
        Scattering operations are operations around the nodes.
        Args:
            scattering (dict): A dictionnary {node : unitary} associating to each node its scattering operator.
            name (str, optional): A name of the operation. As no effect on the dynamic but shows when printing the pipeline.
        
        Examples:
            >>> qw = QWSearch(nx.balanced_tree(3,1), True)
            >>> grover = lambda d: (2/d)*np.ones((d,d)) - np.eye(d)
            >>> pipeline = PipeLine()
            >>> pipeline.add_scattering_by_node({u : grover(qw.degree(u)) for u in qw.nodes()}, name="personalised scattering") # Grover on everyone
            >>> print(pipeline)
            SCATTERING(personalised scattering)
        """
        dic = {"instruction":Instruction.SCATTERING, "scattering":scattering, "mode":"node"}
        if name != None:
            dic["name"] = str(name)
        self.append(dic)

    def add_scattering_by_degree(self, scattering, name=None):
        """ Add a different scattering operation for each degree to the pipeline.
        Scattering operations are operations around the nodes.
        Args:
            scattering (function): A function associating for each possible degree the coresponding scattering operator.
            name (str, optional): A name of the operation. As no effect on the dynamic but shows when printing the pipeline.
        
        Examples:
            >>> qw = QWSearch(nx.balanced_tree(3,1), True)
            >>> grover = lambda d: (2/d)*np.ones((d,d)) - np.eye(d)
            >>> pipeline = PipeLine()
            >>> pipeline.add_scattering_by_degree(grover, name="personalised scattering") # Grover on everyone
            >>> print(pipeline)
            SCATTERING(personalised scattering)
        """
        dic = {"instruction":Instruction.SCATTERING, "scattering":scattering, "mode":"degree"}
        if name != None:
            dic["name"] = str(name)
        self.append(dic)



def walk_on_edges(coin, scattering):
    pipeline = PipeLine(addressing_type=AddressingType.EDGE)
    pipeline.add_coin(coin)
    pipeline.add_scattering(scattering)
    return pipeline

def walk_on_nodes(scattering, coin=_X):
    pipeline = PipeLine(addressing_type=AddressingType.NODE)
    pipeline.add_scattering(scattering)
    pipeline.add_coin(coin)
    return pipeline


def search_edge(coin, scattering, marked, oracle=None):
    pipeline = PipeLine(addressing_type=AddressingType.EDGE)
    if oracle == None:
        oracle = -_X
    pipeline.add_unitary(marked, oracle, name="Oracle")
    pipeline = pipeline + walk_on_edges(coin, scattering)
    pipeline.add_proba(marked)
    return pipeline

def search_virtual_edge(coin, scattering, marked, oracle=None):
    pipeline = PipeLine(addressing_type=AddressingType.VIRTUAL_EDGE)
    if oracle == None:
        oracle = -_X
    pipeline.add_unitary(marked, oracle, name="Oracle")
    pipeline = pipeline + walk_on_edges(coin, scattering)
    pipeline.add_proba(marked)
    return pipeline

def search_node(scattering, marked, oracle=lambda d:-np.eye(d), coin=_X):
    pipeline = PipeLine(addressing_type=AddressingType.NODE)
    pipeline.add_unitary(marked, oracle, name="Oracle")
    pipeline = pipeline + walk_on_nodes(scattering, coin=coin)
    pipeline.add_proba(marked)
    return pipeline






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
    def __init__(self, graph, starify=False):
        self._starified = starify
        
        self._G = deepcopy(graph)
        
        if self._starified:
            self._virtual_edges = self._starify()
        else:
            self._virtual_edges = {}
        
        self._edges = list(self._G.edges()) # List of edges
        self._nodes = list(self._G.nodes()) # List of nodes
        self._E = len(self._edges) # Number of edges
        self._N = len(self._nodes) # Number of nodes
        self._degrees = list(set(list(dict(nx.degree(self._G)).values())))
        self._index = {self._edges[i]:i for i in range(len(self._edges))} # Index for edges
        self._nodes_index = {self._nodes[i]:i  for i in range(self._N)} # Index for nodes

        if nx.bipartite.is_bipartite(self._G):
            color = nx.bipartite.color(self._G) # Coloring
        else:
            color = {self._nodes[i]:i for i in range(len(self._nodes))} # Coloring

        self._polarity = {}
        for (u,v) in self._edges:
            self._polarity[(u,v)] = ("-" if color[u]<color[v] else "+")
            self._polarity[(v,u)] = ("+" if color[u]<color[v] else "-")

        self._initalize_rust_object()        
        

    def _initalize_rust_object(self):
        self._amplitude_labels = [""]*2*self._E
        wiring = [] # For any amplitude self.state[i], says to which node it is connected. Important for the scattering.
        k = 0
        for (i,j) in self._edges:
            edge_label = str(i) + "," + str(j)
            if self._polarity[(i,j)]=="-":
                wiring.append(self._nodes_index[i])
                wiring.append(self._nodes_index[j])
            else:
                wiring.append(self._nodes_index[j])
                wiring.append(self._nodes_index[i])
            self._amplitude_labels[k] = "$\psi_{"+edge_label+"}^-$"
            self._amplitude_labels[k+1] = "$\psi_{"+edge_label+"}^+$"
            k+=2
        

        self._qwf = qwfast.QWFast(wiring,self._N,self._E)
        self._around_nodes_indices = qwfast._get_indices_around_nodes(self._E,self._N,wiring)

        self.reset()

    def _starify(self):
        nodes = copy.deepcopy(self._G.nodes())
        s = {}
        for i in nodes:
            new_name = f"*{i}"
            while new_name in self._G.nodes():
                new_name = f"*{new_name}"
            self._G.add_edge(i,new_name)
            s[i] = (i,new_name)
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
        return deepcopy(self._nodes)

    def edges(self):
        """ Returns the list of edges. Convenient when declaring which edges are marked.

        Returns:
            (list of edge): The list of edges of the underlying graph.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.edges()
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            
        """
        return deepcopy(self._edges)

    def degree(self, node):
        return len(self._G[node])

    def graph(self):
        """ Returns the underlying graph.

        Returns:
            (networkx.Graph): The underlying graph.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.graph()
            <networkx.classes.graph.Graph at 0x7bd045d53c70>
            
        """
        return deepcopy(self._G)

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
        return deepcopy(self._virtual_edges)


    def list_degrees(self):
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
        return deepcopy(self._degrees)







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
                self._polarity[(u,v)] = val
                self._polarity[(v,u)] = opp(val)
            self._initalize_rust_object() 
        else:
            print("Warning: Polarity values not valid !", file=sys.stderr)

    def _get_index(self, pos, _type=AddressingType.EDGE):
        if _type == AddressingType.EDGE:
            index = self._index[pos]
            return [2*index, 2*index+1]
        if _type == AddressingType.VIRTUAL_EDGE:
            #assert(self._search_nodes and pos in self._virtual_edges.keys())
            edge = self._virtual_edges[pos]
            index = self._index[edge]
            return [2*index, 2*index+1]
        if _type == AddressingType.NODE:
            return deepcopy(self._around_nodes_indices[self._nodes_index[pos]])
        if _type == AddressingType.AMPLITUDE:
            (u,v) = pos
            if (u,v) in self._index:
                edge = (u,v)
            else:
                edge = (v,u)
            index = self._index[edge]
            return [2*index] if self._polarity[pos]=="-" else [2*index+1]


    def polarity(self, amplitudes):
        indices = {p:self._get_index(p,AddressingType.AMPLITUDE)[0] for p in amplitudes}
        return {p:"-" if indices[p]%2==0 else "+" for p in amplitudes}

    def label(self, positions, _type=AddressingType.EDGE):
        indices = {p:[i for i in self._get_index(p,_type)] for p in positions}
        return {p:[self._amplitude_labels[i] for i in indices[p]] for p in positions}


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
        indices = {p:[i for i in self._get_index(p,_type)] for p in positions}
        return {p:np.array([self._qwf.state[i] for i in indices[p]],dtype=complex) for p in positions}

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
        state = np.array([0]*2*self._E,dtype=complex)
        for i in range(self._E):
            state[2*i] = new_state[self._edges[i]][0]/s
            state[2*i+1] = new_state[self._edges[i]][1]/s
        self._qwf.state = state

    def proba(self, searched, _type=AddressingType.EDGE):
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
        indices = {p:[i for i in self._get_index(p,_type)] for p in searched}
        return {p:self._qwf.get_proba(indices[p]) for p in searched}

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
        self._qwf.reset()





    


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

        return self._qwf.run(pipeline._read(self),ticks)


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
        l = self._qwf.run_with_trace(pipeline._read(self),ticks)
        l = [[i] + l[i] for i in range(len(l))]
        return pd.DataFrame(l,columns=["steps"] + self._amplitude_labels)

    def get_unitary(self, pipeline, dataframe=False, progress=False):
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
        old_state = copy.deepcopy(self._qwf.state)
        U = []
        for i in (tqdm(range(2*self._E),ncols=100)) if progress else (range(2*self._E)):
            self._qwf.state = np.array([int(i==j) for j in range(2*self._E)],dtype=complex)
            self.run(pipeline, ticks=1)
            U.append(copy.deepcopy(self._qwf.state))
        self._qwf.state = old_state
        U = np.array(U,dtype=complex).transpose()
        if dataframe:
            df = pd.DataFrame(U, index=self._amplitude_labels, columns=self._amplitude_labels)
            return df
        else:
            return U

    def get_T_P(self, pipeline, waiting=10):
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
        indices = np.reshape([self._get_index(i, pipeline.adressing_type) for i in pipeline.measure], (-1,))
        ret = self._qwf.carac(pipeline._read(self),indices,waiting)
        self.reset()
        return ret
