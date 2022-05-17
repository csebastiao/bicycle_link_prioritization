# -*- coding: utf-8 -*-
"""
Functions to make subtractive or additive growth of a graph.
"""


import numpy as np
import networkx as nx
import shapely
from blp import metrics


# TODO: Optimize by finding minimum cut set of size 1 and remove these of the
# edgelist we try instead of passing edges that disconnect the network


def directness_subtractive_step(
        G, edgelist, euclid_mat, keep_connected = False):
    """
    Find the edge to remove on the graph G that optimizes the directness, 
    and that doesn't create an additional component if keep_connected is
    True. The directness is defined here as the linkwise directness 
    which is the sum of the ratio of the euclidean distance and the
    shortest network distance between every pairs of nodes that are on
    the same component.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Graph on which we want to remove an edge.
    edgelist : list
        List of the edges that we can remove.
    euclid_mat : numpy.ndarray
        Array of the euclidean distance between every pairs of nodes of
        the graph G, see metrics.get_euclidean_distance_matrix().
    keep_connected : bool, optional
        If True, edges that can be removed are not creating a new 
        component. The default is False.

    Returns
    -------
    new_m : float
        Value of the directness if the edge chosen is removed from G.
    choice : tuple
        Edge chosen that optimizes the directness if removed,
        defined as the 2 nodes' ID (u,v).

    """
    batch_m = [] # List of metric value after an edge have been removed
    batch_choice = [] # List of corresponding edge removed
    if keep_connected is True:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            # In some cases, removing an edge can create an isolated node
            # that we will discard but increases the number of components,
            # so we need to make sure to pass only the edges that create
            # an additional components larger than 1 node.
            if (nx.number_connected_components(H)
                > nx.number_connected_components(G)) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                new_sm = metrics.get_shortest_network_path_matrix(H)
                sdm = metrics.avoid_zerodiv_matrix(euclid_mat, new_sm)
                batch_m.append(metrics.directness_from_matrix(sdm))
                batch_choice.append(edge)
    else:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            new_sm = metrics.get_shortest_network_path_matrix(H)
            sdm = metrics.avoid_zerodiv_matrix(euclid_mat, new_sm)
            batch_m.append(metrics.directness_from_matrix(sdm))
            batch_choice.append(edge)
    batch = zip(batch_m, batch_choice) # Find maximum value based on metric
    new_m, choice = max(batch) # And associated edge
    return new_m, choice


def relative_directness_subtractive_step(
        G, edgelist, shortest_mat, keep_connected = False):
    """
    Find the edge to remove on the graph G that optimizes the relative
    directness, and that doesn't create an additional component if
    keep_connected is True. The relative directness is defined here as
    the sum of the ratio of the final shortest network distance and the
    actual shortest network distance between every pairs of nodes that are
    on the same component.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Graph on which we want to remove an edge.
    edgelist : list
        List of the edges that we can remove.
    shortest_mat : numpy.ndarray
        Array of the shortest network distance between every pairs of
        nodes of the graph G, see 
        metrics.get_shortest_network_path_matrix().
    keep_connected : bool, optional
        If True, edges that can be removed are not creating a new 
        component. The default is False.

    Returns
    -------
    new_m : float
        Value of the relative directness if the edge chosen is removed from G.
    choice : tuple
        Edge chosen that optimizes the relative directness if removed,
        defined as the 2 nodes' ID (u,v).

    """
    batch_m = [] # List of metric value after an edge have been removed
    batch_choice = [] # List of corresponding edge removed
    if keep_connected is True:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            # See directness_subtractive_step comment on this
            if (nx.number_connected_components(H)
                > nx.number_connected_components(G)) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                new_sm = metrics.get_shortest_network_path_matrix(H)
                sdm = metrics.avoid_zerodiv_matrix(shortest_mat, new_sm)
                batch_m.append(metrics.directness_from_matrix(sdm))
                batch_choice.append(edge)
    else:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            new_sm = metrics.get_shortest_network_path_matrix(H)
            sdm = metrics.avoid_zerodiv_matrix(shortest_mat, new_sm)
            batch_m.append(metrics.directness_from_matrix(sdm))
            batch_choice.append(edge)
    batch = zip(batch_m, batch_choice) # Find maximum value based on metric
    new_m, choice = max(batch) # And associated edge
    return new_m, choice


def global_efficiency_subtractive_step(
        G, edgelist, iem, keep_connected = False):
    """
    Find the edge to remove on the graph G that optimizes the global
    efficiency, and that doesn't create an additional component if
    keep_connected is True. The global efficiency is defined here as
    the ratio of the sum of the inverse of the shortest network distance
    and the sum of the inverse of the euclidean distance between every
    pairs of nodes that are on the same component.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Graph on which we want to remove an edge.
    edgelist : list
        List of the edges that we can remove.
    iem : numpy.ndarray
        Array of the inverse of the euclidean distance between
        every pairs of nodes of the graph G, see
        metrics.get_euclidean_distance_matrix().
    keep_connected : bool, optional
        If True, edges that can be removed are not creating a new 
        component. The default is False.

    Returns
    -------
    new_m : float
        Value of the global efficiency if the edge chosen is removed from G.
    choice : tuple
        Edge chosen that optimizes the global efficiency if removed,
        defined as the 2 nodes' ID (u,v).

    """
    batch_m = [] # List of metric value after an edge have been removed
    batch_choice = [] # List of corresponding edge removed
    if keep_connected is True:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            # See directness_subtractive_step comment on this
            if (nx.number_connected_components(H)
                > nx.number_connected_components(G)) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                new_em = iem.copy()
                new_sm = metrics.get_shortest_network_path_matrix(H)
                # Need to inverse values but avoiding division by 0
                # Inverted 0 gives 0 here
                new_sm = np.divide(
                    np.ones(new_sm.shape), new_sm,
                    out=np.zeros_like(np.ones(new_sm.shape)),
                    where=new_sm!=0)
                # Avoid again division by 0, see metrics.avoid_zerodiv_matrix
                new_em[new_sm == 0.] = 0.
                nge = np.sum(new_sm) / np.sum(new_em)
                batch_m.append(nge)
                batch_choice.append(edge)
    else:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            new_em = iem.copy()
            new_sm = metrics.get_shortest_network_path_matrix(H)
            new_sm = np.divide( # See above
                np.ones(new_sm.shape), new_sm,
                out=np.zeros_like(np.ones(new_sm.shape)),
                where=new_sm!=0)
            new_em[new_sm == 0.] = 0. # See above
            nge = np.sum(new_sm) / np.sum(new_em)
            batch_m.append(nge)
            batch_choice.append(edge)
    batch = zip(batch_m, batch_choice) # Find maximum value based on metric
    new_m, choice = max(batch) # And associated edge
    return new_m, choice


def coverage_subtractive_step(
        G, edgelist, geom, keep_connected = False):
    """
    Find the edge to remove on the graph G that optimizes the coverage, 
    and that doesn't create an additional component if keep_connected is
    True. The coverage is defined here as the area of the buffered edges.
    The buffer size is determined before, as buffered geometries are 
    kept in the geom dict.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Graph on which we want to remove an edge.
    edgelist : list
        List of the edges that we can remove.
    geom : dict
        Dictionary with edges of G as keys and the corresponding
        buffered geometry as values.
    keep_connected : bool, optional
        If True, edges that can be removed are not creating a new 
        component. The default is False.

    Returns
    -------
    new_m : float
        Value of the coverage if the edge chosen is removed from G.
    choice : tuple
        Edge chosen that optimizes the coverage if removed,
        defined as the 2 nodes' ID (u,v).

    """
    batch_m = [] # List of metric value after an edge have been removed
    batch_choice = [] # List of corresponding edge removed
    if keep_connected is True:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            # See directness_subtractive_step comment on this
            if (nx.number_connected_components(H)
                > nx.number_connected_components(G)) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                temp_g = geom.copy()
                temp_g.pop(edge)
                # We find the area by taking every values of the dict,
                # use shapely.ops.unary_union to merge every geometry
                # into one, then find the area of that geometry
                batch_m.append(shapely.ops.unary_union(
                    list(temp_g.values())).area)
                batch_choice.append(edge)
    else:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            temp_g = geom.copy()
            temp_g.pop(edge)
            batch_m.append(shapely.ops.unary_union( # See above
                list(temp_g.values())).area)
            batch_choice.append(edge)
    batch = zip(batch_m, batch_choice) # Find maximum value based on metric
    new_m, choice = max(batch) # And associated edge
    return new_m, choice


def relative_coverage_subtractive_step(
        G, edgelist, bef_area, geom, keep_connected = False):
    """
    Find the edge to remove on the graph G that optimizes the relative
    coverage, and that doesn't create an additional component if
    keep_connected is True. The relative coverage is defined here as the
    difference in the area before and after removing the edge, divided by
    the length of the removed edge. The buffer size is determined before,
    as buffered geometries are kept in the geom dict.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Graph on which we want to remove an edge.
    edgelist : list
        List of the edges that we can remove.
    bef_area : float
        Area of the buffered edges of G.
    geom : dict
        Dictionary with edges of G as keys and the corresponding
        buffered geometry as values.
    keep_connected : bool, optional
        If True, edges that can be removed are not creating a new 
        component. The default is False.

    Returns
    -------
    new_m : float
        Value of the coverage if the edge chosen is removed from G.
    choice : tuple
        Edge chosen that optimizes the coverage if removed,
        defined as the 2 nodes' ID (u,v).

    """
    batch_m = [] # List of metric value after an edge have been removed
    batch_choice = [] # List of corresponding edge removed
    if keep_connected is True:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            # See directness_subtractive_step comment on this
            if (nx.number_connected_components(H)
                > nx.number_connected_components(G)) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                temp_g = geom.copy()
                temp_g.pop(edge)
                # See coverage_subtractive_step comment on this
                area = shapely.ops.unary_union(list(temp_g.values())).area
                # We find the relative coverage by comparing the actual area
                # to the area before removing an edge, divided by the length
                # of the edge removed
                batch_m.append((bef_area - area) 
                               / G.edges[edge]['length'])
                batch_choice.append(edge)
    else:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            temp_g = geom.copy()
            temp_g.pop(edge)
            area = shapely.ops.unary_union( # See above
                list(temp_g.values())).area
            batch_m.append((bef_area - area) # See above
                           / G.edges[edge]['length'])
            batch_choice.append(edge)
    batch = zip(batch_m, batch_choice) # Find minimum value based on metric
    new_m, choice = min(batch) # And associated edge
    return new_m, choice



def directness_additive_step(
        G, actual_edges, edgelist, keep_connected = False):
    """
    Find the edge to add to the actual_edges list from the final graph G
    that optimizes the directness, and that doesn't create an additional
    component if keep_connected is True. The directness is defined here as
    the linkwise directness which is the sum of the ratio of the euclidean
    distance and the shortest network distance between every pairs of nodes
    that are on the same component.

    For now, not using an already measured euclidean matrix, the gain
    is quite small in time and for addition instead of substraction that 
    would be a bit harder to do, better to redo everything again

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Final graph.
    actual_edges : list
        List of the edges that are already created from G.
    edgelist : list
        List of the edges that we can add from G to actual_edges
    keep_connected : bool, optional
        If True, edges that can be added are not creating a new 
        component. The default is False.

    Returns
    -------
    new_m : float
        Value of the directness if the edge chosen is added to actual_edges.
    choice : tuple
        Edge chosen that optimizes the directness if added,
        defined as the 2 nodes' ID (u,v).
    """
    batch_m = []
    batch_choice = []
    if keep_connected is True:
        for edge in edgelist:
            temp_edges = actual_edges.copy()
            temp_edges.append(edge)
            # Make subgraph of the actual_edges + the edge that we try to add
            H = G.edge_subgraph(temp_edges).copy()
            # See directness_subtractive_step comment on this
            if (nx.number_connected_components(H)
                > nx.number_connected_components(
                    G.edge_subgraph(actual_edges))) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                dm = metrics.get_directness_matrix_networkx(H)
                batch_m.append(metrics.directness_from_matrix(dm))
                batch_choice.append(edge)
    else:
        for edge in edgelist:
            temp_edges = actual_edges.copy()
            temp_edges.append(edge)
            H = G.edge_subgraph(actual_edges).copy() # See above
            dm = metrics.get_directness_matrix_networkx(H)
            batch_m.append(metrics.directness_from_matrix(dm))
            batch_choice.append(edge)
    batch = zip(batch_m, batch_choice) # Find maximum value based on metric
    new_m, choice = max(batch) # And associated edge
    return new_m, choice


# TODO, adapt final_sm to the nodes of actual_sm only, use pandas filtering ?
def relative_directness_additive_step(
        G, actual_edges, edgelist, keep_connected = False):
    batch_m = []
    batch_choice = []
    if keep_connected is True:
        for edge in edgelist:
            temp_edges = actual_edges.copy()
            temp_edges.append(edge)
            H = G.edge_subgraph(temp_edges).copy()
            # See directness_subtractive_step comment on this
            if (nx.number_connected_components(H)
                > nx.number_connected_components(
                    G.edge_subgraph(actual_edges))) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                final_sm = metrics.get_shortest_network_path_matrix(G)
                actual_sm = metrics.get_shortest_network_path_matrix(H)
                dm = metrics.avoid_zerodiv_matrix(final_sm, actual_sm)
                batch_m.append(metrics.directness_from_matrix(dm))
                batch_choice.append(edge)
    else:
        for edge in edgelist:
            temp_edges = actual_edges.copy()
            temp_edges.append(edge)
            H = G.edge_subgraph(actual_edges).copy()
            final_sm = metrics.get_shortest_network_path_matrix(G)
            actual_sm = metrics.get_shortest_network_path_matrix(H)
            dm = metrics.avoid_zerodiv_matrix(final_sm, actual_sm)
            batch_m.append(metrics.directness_from_matrix(dm))
            batch_choice.append(edge)
    batch = zip(batch_m, batch_choice)
    new_m, choice = max(batch)
    return new_m, choice


def relative_coverage_additive_step(
        G, BUFF_SIZE, actual_edges, edgelist,
        bef_area, geom, keep_connected = False):
    """
    Find the edge to add to the actual_edges list from the final graph G
    that optimizes the relative coverage, and that doesn't create an
    additional component if keep_connected is True. The relative coverage
    is defined here as the difference in the area before and after adding
    the edge, divided by the length of the added edge. The buffer size
    is determined by the BUFF_SIZE constant.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Graph on which we want to remove an edge.
    BUFF_SIZE : float
        Constant that determines the buffer that we apply on the edge's
        geometry. For good results, the BUFFER_SIZE need to be the same
        that the one used for other values from the geom dict.
    actual_edges : list
        List of the edges that are already created from G.
    edgelist : list
        List of the edges that we can add from G to actual_edges
    bef_area : float
        Area of the buffered edges of G.
    geom : dict
        Dictionary with edges of G as keys and the corresponding
        buffered geometry as values.
    keep_connected : bool, optional
        If True, edges that can be added are not creating a new 
        component. The default is False.

    Returns
    -------
    new_m : float
        Value of the relative coverage if the edge chosen is added
        to actual_edges.
    choice : tuple
        Edge chosen that optimizes the relative coverage if added,
        defined as the 2 nodes' ID (u,v).
    """
    batch_m = []
    batch_choice = []
    if keep_connected is True:
        for edge in edgelist:
            temp_edges = actual_edges.copy()
            temp_edges.append(edge)
            H = G.edge_subgraph(temp_edges).copy()
            # See directness_subtractive_step comment on this
            if (nx.number_connected_components(H)
                > nx.number_connected_components(
                    G.edge_subgraph(actual_edges))) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                temp_g = geom.copy()
                temp_g[edge] = G.edges[edge]['geometry'].buffer(BUFF_SIZE)
                # See coverage_subtractive_step comment on this
                area = shapely.ops.unary_union(list(temp_g.values())).area
                # See relative_coverage_subtractive_step comment on this,
                # same principle but since we add and not remove, we take
                # the oppositve value
                batch_m.append((area - bef_area) / G.edges[edge]['length'])
                batch_choice.append(edge)
    else:
        for edge in edgelist:
            temp_g = geom.copy()
            temp_g.append(G.edges[edge]['geometry'].buffer(BUFF_SIZE))
            area = shapely.ops.unary_union(temp_g).area
            batch_m.append((area - bef_area) / G.edges[edge]['length'])
            batch_choice.append(edge)
    # Need to try because sometimes the difference is so small that it 
    # can't find one so will look into the edge instead of the metric
    # to avoid this, we will find them individually
    # We also maximize and not minimize compared to 
    # relative_coverage_subtractive_step because here we want to grow as much
    # as possible and not reduce the shrink as much as possible
    try:
        new_m, choice = max(zip(batch_m, batch_choice))
    except:
        new_m = max(batch_m)
        # TODO: Verify
        index_max = max(range(len(batch_m)), key=batch_m.__getitem__)
        choice = batch_choice[index_max]
    return new_m, choice

