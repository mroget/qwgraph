from qwgraph import qwsearch as qws


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
    pipeline = qws.PipeLine(addressing_type=qws.AddressingType.EDGE)
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
    pipeline = qws.PipeLine(addressing_type=qws.AddressingType.NODE)
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
    pipeline = qws.PipeLine(addressing_type=qws.AddressingType.EDGE)
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
    pipeline = qws.PipeLine(addressing_type=qws.AddressingType.VIRTUAL_EDGE)
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
    pipeline = qws.PipeLine(addressing_type=qws.AddressingType.NODE)
    for m in marked:
        pipeline.add_unitary([m], oracle, name=f"Oracle on {m}")
    pipeline = pipeline + walk_on_nodes(scattering, coin=coin)
    pipeline.add_proba(marked)
    return pipeline