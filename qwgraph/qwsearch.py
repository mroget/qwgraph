##############################################
##                  Imports                 ##
##############################################
# Maths
import numpy as np
import math 
from math import pi 
# Graph
import networkx as nx 
# Utilities
import copy
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import warnings
from enum import Enum
from itertools import chain

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
            pos = list(chain.from_iterable([qw._get_index(i, dic["addressing_type"]) for i in dic["targets"]]))
            op.set_to_proba(pos)
            return op

        if dic["instruction"] == Instruction.UNITARY:
            pos = list(chain.from_iterable([qw._get_index(i, dic["addressing_type"]) for i in dic["targets"]]))
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

                Scatter.set_type(2, data)

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
            unitary (np.array): The matrix of the unitary operator. Alternatively, can be a function that takes one argument (the degree) and returns an unitary.
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
    """ Create a default pipeline of a QW on edges

    Args:
        coin (numpy array): coin (np.array or dict): The two by two matrix of the coin. Can alternatively be a dictionnary {edge : coin} for a coin dependent of the edge.
        scattering (str): "cycle" for a cycle around the node or "grover" for a Grover diffusion around the node.

    Returns:
        (PipeLine): The pipeline of a QW on edges.

    Examples:
        >>> walk_on_edges(coins.X,"grover")
        COIN -> SCATTERING
            
    """
    pipeline = PipeLine(addressing_type=AddressingType.EDGE)
    pipeline.add_coin(coin)
    pipeline.add_scattering(scattering)
    return pipeline

def walk_on_nodes(scattering, coin=_X):
    """ Create a default pipeline of a QW on nodes

    Args:
        scattering (str): "cycle" for a cycle around the node or "grover" for a Grover diffusion around the node.
        coin (numpy array, optional): coin (np.array or dict): The two by two matrix of the coin. Can alternatively be a dictionnary {edge : coin} for a coin dependent of the edge.

    Returns:
        (PipeLine): The pipeline of a QW on edges.

    Examples:
        >>> walk_on_nodes("grover")
        SCATTERING -> COIN
            
    """
    pipeline = PipeLine(addressing_type=AddressingType.NODE)
    pipeline.add_scattering(scattering)
    pipeline.add_coin(coin)
    return pipeline


def search_edges(coin, scattering, marked, oracle=None):
    """ Create a default pipeline of a QW on edges searching edges

    Args:
        coin (numpy array): coin (np.array or dict): The two by two matrix of the coin. Can alternatively be a dictionnary {edge : coin} for a coin dependent of the edge.
        scattering (str): "cycle" for a cycle around the node or "grover" for a Grover diffusion around the node.
        marked (list): the list of marked edges.
        oracle (numpy array, optional): The two by two unitary operator of the oracle. If None, then -X will be used.

    Returns:
        (PipeLine): The pipeline of a QW on edges searching edges.

    Examples:
        >>> qw = QWSearch(nx.complete_graph(100), starify=True)
        >>> search_edges(coins.X, "grover", qw.edges()[0:4], -coins.X)
        UNITARY(Oracle on (0, 1)) -> UNITARY(Oracle on (0, 2)) -> UNITARY(Oracle on (0, 3)) -> UNITARY(Oracle on (0, 4)) -> COIN -> SCATTERING -> PROBA
            
    """
    pipeline = PipeLine(addressing_type=AddressingType.EDGE)
    if type(oracle) == type(None):
        oracle = -_X
    for m in marked:
        pipeline.add_unitary([m], oracle, name=f"Oracle on {m}")
    pipeline = pipeline + walk_on_edges(coin, scattering)
    pipeline.add_proba(marked)
    return pipeline

def search_virtual_edges(coin, scattering, marked, oracle=None):
    """ Create a default pipeline of a QW on edges searching nodes

    Args:
        coin (numpy array): coin (np.array or dict): The two by two matrix of the coin. Can alternatively be a dictionnary {edge : coin} for a coin dependent of the edge.
        scattering (str): "cycle" for a cycle around the node or "grover" for a Grover diffusion around the node.
        marked (list): the list of marked edges.
        oracle (numpy array, optional): The two by two unitary operator of the oracle. If None, then -X will be used.

    Returns:
        (PipeLine): The pipeline of a QW on edges searching edges.

    Examples:
        >>> qw = QWSearch(nx.complete_graph(100), starify=True)
        >>> search_virtual_edges(coins.X, "grover", qw.nodes()[0:4], -coins.X)
        UNITARY(Oracle on 0) -> UNITARY(Oracle on 1) -> UNITARY(Oracle on 2) -> UNITARY(Oracle on 3) -> COIN -> SCATTERING -> PROBA
            
    """
    pipeline = PipeLine(addressing_type=AddressingType.VIRTUAL_EDGE)
    if type(oracle) == type(None):
        oracle = -_X
    for m in marked:
        pipeline.add_unitary([m], oracle, name=f"Oracle on {m}")
    pipeline = pipeline + walk_on_edges(coin, scattering)
    pipeline.add_proba(marked)
    return pipeline

def search_nodes(scattering, marked, oracle=lambda d:-np.eye(d), coin=_X):
    """ Create a default pipeline of a QW on nodes searching nodes

    Args:
        scattering (str): "cycle" for a cycle around the node or "grover" for a Grover diffusion around the node.
        marked (list): the list of marked edges.
        oracle (numpy array, optional): The unitary operator of the oracle (or a function of the degree returning the unitary). If None, then -I will be used.
        coin (numpy array, optional): coin (np.array or dict): The two by two matrix of the coin. Can alternatively be a dictionnary {edge : coin} for a coin dependent of the edge.

    Returns:
        (PipeLine): The pipeline of a QW on edges searching edges.

    Examples:
        >>> qw = QWSearch(nx.complete_graph(100), starify=True)
        >>> search_nodes("grover", qw.nodes()[0:4], lambda d:-np.eye(d), coins.X)
        UNITARY(Oracle on 0) -> UNITARY(Oracle on 1) -> UNITARY(Oracle on 2) -> UNITARY(Oracle on 3) -> SCATTERING -> COIN -> PROBA
            
    """
    pipeline = PipeLine(addressing_type=AddressingType.NODE)
    for m in marked:
        pipeline.add_unitary([m], oracle, name=f"Oracle on {m}")
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

    Args:
        graph (networkx.Graph): The graph on which the QW will be defined. Alternatively, graph can be a DiGraph. In that case, the polarity will follow the orientation of the edges. Self loops are ignored.
        starify (bool, optional): If True, the graph will be starified. 
    """

    ######################
    ### Init functions ###
    ######################
    def __init__(self, graph, starify=False):
        self._starified = starify
        
        self._G = nx.Graph()
        edges = list(graph.edges())
        edges = [(u,v) for (u,v) in edges if u!=v]
        self._G.add_edges_from(edges)
        
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
        """ Returns the degree of a node.

        Args:
            node (node): The node you want the degree of.

        Returns:
            (int): The degree of node.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.degree(0)
            3
            
        """
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
        """ Returns a dictionnary that associates its virtual edge to each node.

        Returns:
            (dict): A dictionnary {node: edge} that associates each node to its virtual edge.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.virtual_edges()
            {}
            >>> qw = QWSearch(nx.complete_graph(4),starify=True)
            >>> qw.virtual_edges()
            {0: (0, '*0'), 1: (1, '*1'), 2: (2, '*2'), 3: (3, '*3')}
            
        """
        return deepcopy(self._virtual_edges)







    #############################
    ### Getters/Setters state ###
    #############################

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



    def polarity(self, targets):
        """ Returns the polarity of the targets.

        Args:   
            targets (list of target): The list of targets. Targets must be amplitudes.

        Returns:
            (dict): The polarity ("+" or "-") of the targets in a dictionnary {target : polarity}.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.polarity([(0,1),(1,0), (2,1),(1,2)])
            {(0, 1): '-', (1, 0): '+', (2, 1): '+', (1, 2): '-'}
        """
        indices = {p:self._get_index(p,AddressingType.AMPLITUDE)[0] for p in targets}
        return {p:"-" if indices[p]%2==0 else "+" for p in targets}

    def label(self, targets, _type=AddressingType.EDGE):
        """ Returns the latex label of the targets.

        Args:   
            targets (list of target): The list of targets. targets are edges, nodes, virtual edges or amplitudes depending of the addressing type.
            _type (AddressingType, optional): The addressing type. Decides weither we addresse nodes, edges, amplitudes or virtual edges.

        Returns:
            (dict): The labels of the targets in a dictionnary {target : labels}.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.label([(0,1),(1,2)], _type = AddressingType.EDGE) # 1 edge <=> 2 amplitudes
            {(0, 1): ['$\\psi_{0,1}^-$', '$\\psi_{0,1}^+$'], (1, 2): ['$\\psi_{1,2}^-$', '$\\psi_{1,2}^+$']}
            >>> qw.label([(0,1),(1,2)], _type = AddressingType.AMPLITUDE) 
            {(0, 1): ['$\\psi_{0,1}^-$'], (1, 2): ['$\\psi_{1,2}^-$']}
            >>> qw.label([0,1], _type = AddressingType.NODE) # 1 node <=> 3 amplitudes
            {0: ['$\\psi_{0,1}^-$', '$\\psi_{0,2}^-$', '$\\psi_{0,3}^-$'], 1: ['$\\psi_{0,1}^+$', '$\\psi_{1,2}^-$', '$\\psi_{1,3}^-$']}
            >>> qw = QWSearch(nx.complete_graph(4), starify=True)
            >>> qw.label([0,1], _type = AddressingType.VIRTUAL_EDGE) # 1 edge <=> 2 amplitudes
            {0: ['$\\psi_{0,*0}^-$', '$\\psi_{0,*0}^+$'], 1: ['$\\psi_{1,*1}^-$', '$\\psi_{1,*1}^+$']}
        """
        indices = {p:[i for i in self._get_index(p,_type)] for p in targets}
        return {p:[self._amplitude_labels[i] for i in indices[p]] for p in targets}


    def proba(self, targets, _type=AddressingType.EDGE):
        """ Returns the probability to measure one of the targets.

        Args:   
            targets (list of target): The list of targets. targets are edges, nodes, virtual edges or amplitudes depending of the addressing type.
            _type (AddressingType, optional): The addressing type. Decides weither we addresse nodes, edges, amplitudes or virtual edges.

        Returns:
            (dict): The probability of measuring any of the targets as a dictionnary {target : proba}.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.proba([(0,1),(1,2)], _type = AddressingType.EDGE) # 1 edge <=> 2 amplitudes <=> probability of 2/12 with a diagonal distribution
            {(0, 1): 0.1666666666666667, (1, 2): 0.1666666666666667}
            >>> qw.proba([(0,1),(1,2)], _type = AddressingType.AMPLITUDE) # 1 amplitude <=> probability of 1/12 with a diagonal distribution
            {(0, 1): 0.08333333333333336, (1, 2): 0.08333333333333336}
            >>> qw.proba([0,1], _type = AddressingType.NODE) # 1 node <=> 3 amplitudes <=> probability of 3/12 with a diagonal distribution
            {0: 0.25000000000000006, 1: 0.25000000000000006}
            >>> qw = QWSearch(nx.complete_graph(4), starify=True)
            >>> qw.proba([0,1], _type = AddressingType.VIRTUAL_EDGE) # 1 edge <=> 2 amplitudes <=> probability of 2/20 with a diagonal distribution on the starified graph
            {0: 0.09999999999999999, 1: 0.09999999999999999}
        """
        indices = {p:[i for i in self._get_index(p,_type)] for p in targets}
        return {p:self._qwf.get_proba(indices[p]) for p in targets}


    def state(self, targets, _type=AddressingType.EDGE):
        """ Returns the probability to measure one of the targets.

        Args:   
            targets (list of target): The list of targets. targets are edges, nodes, virtual edges or amplitudes depending of the addressing type.
            _type (AddressingType, optional): The addressing type. Decides weither we addresse nodes, edges, amplitudes or virtual edges.

        Returns:
            (dict): The state/amplitude of the targets as a dictionnary {target : state}.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(4))
            >>> qw.state([(0,1),(1,2)], _type = AddressingType.EDGE) # 1 edge <=> 2 amplitudes
            {(0, 1): array([0.28867513+0.j, 0.28867513+0.j]), (1, 2): array([0.28867513+0.j, 0.28867513+0.j])}
            >>> qw.state([(0,1),(1,2)], _type = AddressingType.AMPLITUDE)
            {(0, 1): array([0.28867513+0.j]), (1, 2): array([0.28867513+0.j])}
            >>> qw.state([0,1], _type = AddressingType.NODE) # 1 node <=> 3 amplitudes
            {0: array([0.28867513+0.j, 0.28867513+0.j, 0.28867513+0.j]), 1: array([0.28867513+0.j, 0.28867513+0.j, 0.28867513+0.j])}
            >>> qw = QWSearch(nx.complete_graph(4), starify=True)
            >>> qw.state([0,1], _type = AddressingType.VIRTUAL_EDGE) # 1 edge <=> 2 amplitudes
            {0: array([0.2236068+0.j, 0.2236068+0.j]), 1: array([0.2236068+0.j, 0.2236068+0.j])}
        """
        indices = {p:[i for i in self._get_index(p,_type)] for p in targets}
        return {p:np.array([self._qwf.state[i] for i in indices[p]],dtype=complex) for p in targets}




    def set_state(self, new_state):
        """ Change the state (i.e. the amplitudes for every edges).

        For an edge (u,v), the amplitudes $\psi_{u,v}^+$ and $\psi_{u,v}^-$ will be modified according to the argument.
        If the new state is not normalized, this method will automatically normalize it.

        Args:
            new_state (dict): A dictionnary of the form edge: amplitudes. Amplitudes must be numpy arrays or lists of dimension 2.
        
        Examples:
            >>> qw = QWSearch(nx.cycle_graph(4))
            >>> qw.state(qw.edges())
            {(0, 1): array([0.35355339+0.j, 0.35355339+0.j]),
             (0, 3): array([0.35355339+0.j, 0.35355339+0.j]),
             (1, 2): array([0.35355339+0.j, 0.35355339+0.j]),
             (2, 3): array([0.35355339+0.j, 0.35355339+0.j])}
            >>> qw.set_state({edge:[2,1j] for edge in qw.edges()})
            >>> qw.state(qw.edges())
            {(0, 1): array([0.32444284+0.j        , 0.        +0.16222142j]),
             (0, 3): array([0.48666426+0.j        , 0.16222142+0.16222142j]),
             (1, 2): array([0.48666426+0.j        , 0.16222142+0.16222142j]),
             (2, 3): array([0.48666426+0.j        , 0.16222142+0.16222142j])}
        """
        s = np.sqrt(sum([abs(new_state[e][0])**2 + abs(new_state[e][1])**2 for e in new_state]))
        state = np.array([0]*2*self._E,dtype=complex)
        for i in range(self._E):
            state[2*i] = new_state[self._edges[i]][0]/s
            state[2*i+1] = new_state[self._edges[i]][1]/s
        self._qwf.state = state

    def reset(self):
        """ Reset the state to a diagonal one and reset the current step to 0.
        Do not return anything.

        Examples:
            >>> qw = QWSearch(nx.cycle_graph(4))
            >>> qw.state(qw.edges())
            {(0, 1): array([0.35355339+0.j, 0.35355339+0.j]),
             (0, 3): array([0.35355339+0.j, 0.35355339+0.j]),
             (1, 2): array([0.35355339+0.j, 0.35355339+0.j]),
             (2, 3): array([0.35355339+0.j, 0.35355339+0.j])}
            >>> qw.set_state({edge:[2,1j] for edge in qw.edges()})
            >>> qw.state(qw.edges())
            {(0, 1): array([0.4472136+0.j       , 0.       +0.2236068j]),
             (0, 3): array([0.4472136+0.j       , 0.       +0.2236068j]),
             (1, 2): array([0.4472136+0.j       , 0.       +0.2236068j]),
             (2, 3): array([0.4472136+0.j       , 0.       +0.2236068j])}
            >>> qw.reset()
            >>> qw.state(qw.edges())
            {(0, 1): array([0.35355339+0.j, 0.35355339+0.j]),
             (0, 3): array([0.35355339+0.j, 0.35355339+0.j]),
             (1, 2): array([0.35355339+0.j, 0.35355339+0.j]),
             (2, 3): array([0.35355339+0.j, 0.35355339+0.j])}

        """
        self._qwf.reset()





    


    def run(self, pipeline, ticks=1):
        """ Run the simulation with the given pipeline for ticks steps.
        The state will be modified inplace and the proba extracted (if any) will be returned. 

        Args:
            pipeline (PipeLine): The pipeline containing the operations of one step.
            ticks (int, optional): The number of time steps.

        Returns:
            (np.array): The probabilities measured in order.

        Examples:
            >>> qw = QWSearch(nx.cycle_graph(5))
            >>> qw.set_state({edge:([1,0] if edge==qw.edges()[len(qw.edges())//2] else [0,0]) for edge in qw.edges()})
            >>> qw.proba(qw.edges())
            {(0, 1): 0.0, (0, 4): 0.0, (1, 2): 1.0, (4, 3): 0.0, (2, 3): 0.0}
            >>> pipeline = PipeLine()
            >>> pipeline.add_coin(coins.H)
            >>> pipeline.add_scattering("cycle")
            >>> qw.run(pipeline, ticks=2)
            >>> qw.proba(qw.edges())
            {(0, 1): 0.0, (0, 4): 0.2499999999999999, (1, 2): 0.4999999999999998, (4, 3): 0.2499999999999999, (2, 3): 0.0}
            >>> pipeline.add_proba([(0,1)])
            >>> qw.set_state({edge:([1,0] if edge==qw.edges()[len(qw.edges())//2] else [0,0]) for edge in qw.edges()})
            >>> p = qw.run(pipeline, ticks=10)
            >>> p
            array([0.5, 0., 0.125, 0.0625, 0.25, 0.078125, 0.5078125, 0.265625, 0.29101562, 0.06347656])
        """

        return np.array(self._qwf.run(pipeline._read(self),ticks))


    def get_unitary(self, pipeline, dataframe=False, progress=False):
        """ For a given pipeline, compute and return the unitary U coresponding to one step of the QW.

        This method **do not** change the state of the QW.

        Args:
            pipeline (PipeLine): The pipeline containing the operations of one step.
            dataframe (bool, optional): If True, the result will be a pandas dataframe instead of a numpy array. 
            progress (bool, optional): If True, a tqdm progress bar will be displayed.

        Returns:
            (numpy array or pandas dataframe): The unitary operator coresponding to one step of the dynamic. If dataframe is set to True, a pandas dataframe will be returned instead.
        
        Examples:
            >>> qw = QWSearch(nx.balanced_tree(3,1))
            >>> qw.get_unitary(walk_on_edges(coins.X, "grover"))
            array([[ 0.        +0.j,  1.        +0.j,  0.        +0.j,
                     0.        +0.j,  0.        +0.j,  0.        +0.j],
                   [-0.33333333+0.j,  0.        +0.j,  0.66666667+0.j,
                     0.        +0.j,  0.66666667+0.j,  0.        +0.j],
                   [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                     1.        +0.j,  0.        +0.j,  0.        +0.j],
                   [ 0.66666667+0.j,  0.        +0.j, -0.33333333+0.j,
                     0.        +0.j,  0.66666667+0.j,  0.        +0.j],
                   [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                     0.        +0.j,  0.        +0.j,  1.        +0.j],
                   [ 0.66666667+0.j,  0.        +0.j,  0.66666667+0.j,
                     0.        +0.j, -0.33333333+0.j,  0.        +0.j]])
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

    def get_T_P(self, pipeline, waiting=10, maxiter=10000):
        """ Computes the hitting time and probability of success for a given QW. 

        The waiting parameter is used to accumalate informations about the signal (recommended to be at least 10).

        In details, this algorithm look at the time serie of the probability of success $p(t)$. 
        At any time step $t$, we define $T_{max}(t) = \\underset{{t' \\leq t}}{\\mathrm{argmax }}\\; p(t')$ and $T_{min}(t) = \\underset{{t' \\leq t}}{\\mathrm{argmin }} \\; p(t')$.
        
        The algorithms computes the series $p(t)$, $T_{max}(t)$, $T_{min}(t)$ and stop when it encounters `t>waiting` such that $p(t)<\\frac{p\\left(T_{max}(t)\\right)+p\\left(T_{max}(t)\\right)}{2}$. 
        It then returns $T_{max}(t), p\\left(T_{max}(t)\\right)$.

        **Warning:** This function won't change the state of the QW.

        If several probability extractions are present in the pipeline, the hitting time returned will be the number of extraction before reaching the first peak.

        Args:
            pipeline (PipeLine): The pipeline containing the operations of one step.
            waiting (int, optional): The number of steps the algorithm uses to collect information on the signal. Must be smaller than the hitting time.
            maxiter (int, optional): The maximum number of iterations of the QW.

        Returns:
            (int*float): T:int,P:float respectively the hitting time and probability of success.

        Examples:
            >>> qw = QWSearch(nx.complete_graph(100))
            >>> qw.get_T_P(search_edges(coins.X, "grover", [qw.edges()[0]], -coins.X))
            (55, 0.9812661464139945)
        """

        old_state = copy.deepcopy(self._qwf.state)
        ret = self._qwf.carac(pipeline._read(self),waiting,maxiter)
        self._qwf.state = old_state
        return ret
