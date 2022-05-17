# -*- coding: utf-8 -*-
"""

"""


from blp import metrics
import networkx as nx
import numpy as np
import shapely


# TODO: Optimize by finding minimum cut set of size 1 and remove these of the
# edgelist we try instead of passing edges that disconnect the network


def directness_subtractive_step(
        G, edgelist, euclid_mat, keep_connected = False):
    batch_m = []
    batch_choice = []
    if keep_connected is True:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
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
    batch = zip(batch_m, batch_choice)
    new_m, choice = max(batch)
    return new_m, choice


def relative_directness_subtractive_step(
        G, edgelist, shortest_mat, keep_connected = False):
    batch_m = []
    batch_choice = []
    if keep_connected is True:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
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
    batch = zip(batch_m, batch_choice)
    new_m, choice = max(batch)
    return new_m, choice


def global_efficiency_subtractive_step(
        G, edgelist, iem, keep_connected = False):
    batch_m = []
    batch_choice = []
    if keep_connected is True:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            if (nx.number_connected_components(H)
                > nx.number_connected_components(G)) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                new_em = iem.copy()
                new_sm = metrics.get_shortest_network_path_matrix(H)
                new_sm = np.divide(
                    np.ones(new_sm.shape), new_sm,
                    out=np.zeros_like(np.ones(new_sm.shape)),
                    where=new_sm!=0)
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
            new_sm = np.divide(
                np.ones(new_sm.shape), new_sm,
                out=np.zeros_like(np.ones(new_sm.shape)),
                where=new_sm!=0)
            new_em[new_sm == 0.] = 0.
            nge = np.sum(new_sm) / np.sum(new_em)
            batch_m.append(nge)
            batch_choice.append(edge)
    batch = zip(batch_m, batch_choice)
    new_m, choice = max(batch)
    return new_m, choice


def coverage_subtractive_step(
        G, edgelist, geom, keep_connected = False):
    batch_m = []
    batch_choice = []
    if keep_connected is True:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            if (nx.number_connected_components(H)
                > nx.number_connected_components(G)) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                temp_g = geom.copy()
                temp_g.pop(edge)
                batch_m.append(shapely.ops.unary_union(
                    list(temp_g.values())).area)
                batch_choice.append(edge)
    else:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            temp_g = geom.copy()
            temp_g.pop(edge)
            batch_m.append(shapely.ops.unary_union(
                list(temp_g.values())).area)
            batch_choice.append(edge)
    batch = zip(batch_m, batch_choice)
    new_m, choice = max(batch)
    return new_m, choice


def relative_coverage_subtractive_step(
        G, edgelist, bef_area, geom, keep_connected = False):
    batch_m = []
    batch_choice = []
    if keep_connected is True:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            if (nx.number_connected_components(H)
                > nx.number_connected_components(G)) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                temp_g = geom.copy()
                temp_g.pop(edge)
                area = shapely.ops.unary_union(
                    list(temp_g.values())).area
                batch_m.append((bef_area - area) 
                               / G.edges[edge]['length'])
                batch_choice.append(edge)
    else:
        for edge in edgelist:
            H = G.copy()
            H.remove_edge(*edge)
            temp_g = geom.copy()
            temp_g.pop(edge)
            area = shapely.ops.unary_union(
                list(temp_g.values())).area
            batch_m.append((bef_area - area) 
                           / G.edges[edge]['length'])
            batch_choice.append(edge)
    batch = zip(batch_m, batch_choice)
    new_m, choice = min(batch)
    return new_m, choice


def directness_additive_step(
        G, actual_edges, edgelist, keep_connected = False):
    batch_m = []
    batch_choice = []
    if keep_connected is True:
        for edge in edgelist:
            temp_edges = actual_edges.copy()
            temp_edges.append(edge)
            H = G.edge_subgraph(temp_edges).copy()
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
            H = G.edge_subgraph(actual_edges).copy()
            dm = metrics.get_directness_matrix_networkx(H)
            batch_m.append(metrics.directness_from_matrix(dm))
            batch_choice.append(edge)
    batch = zip(batch_m, batch_choice)
    new_m, choice = max(batch)
    return new_m, choice

# TODO
def relative_directness_additive_step(
        G, actual_edges, edgelist, keep_connected = False):
    batch_m = []
    batch_choice = []
    if keep_connected is True:
        for edge in edgelist:
            temp_edges = actual_edges.copy()
            temp_edges.append(edge)
            H = G.edge_subgraph(temp_edges).copy()
            if (nx.number_connected_components(H)
                > nx.number_connected_components(
                    G.edge_subgraph(actual_edges))) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                dm = 0 # TODO
                batch_m.append(metrics.directness_from_matrix(dm))
                batch_choice.append(edge)
    else:
        for edge in edgelist:
            temp_edges = actual_edges.copy()
            temp_edges.append(edge)
            H = G.edge_subgraph(actual_edges).copy()
            dm = metrics.get_directness_matrix_networkx(H)
            batch_m.append(metrics.directness_from_matrix(dm))
            batch_choice.append(edge)
    batch = zip(batch_m, batch_choice)
    new_m, choice = max(batch)
    return new_m, choice


def relative_coverage_additive_step(
        G, BUFF_SIZE, actual_edges, edgelist,
        bef_area, geom, keep_connected = False):
    batch_m = []
    batch_choice = []
    if keep_connected is True:
        for edge in edgelist:
            temp_edges = actual_edges.copy()
            temp_edges.append(edge)
            H = G.edge_subgraph(temp_edges).copy()
            if (nx.number_connected_components(H)
                > nx.number_connected_components(
                    G.edge_subgraph(actual_edges))) and (
                    len(sorted(nx.connected_components(H), key=len)[0]) > 1):
                pass
            else:
                temp_g = geom.copy()
                temp_g.append(G.edges[edge]['geometry'].buffer(BUFF_SIZE))
                area = shapely.ops.unary_union(temp_g).area
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
    # to avoid this in this case we will find them individually
    try:
        new_m, choice = max(zip(batch_m, batch_choice))
    except:
        new_m = max(batch_m)
        index_max = max(range(len(batch_m)), key=batch_m.__getitem__)
        choice = batch_choice[index_max]
    return new_m, choice

