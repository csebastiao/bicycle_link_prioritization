# -*- coding: utf-8 -*-
"""
Script to test and time different ways to measure directness on a graph.
"""


import networkx as nx
import igraph as ig
import blp.directness as directness
import blp.utils as utils
import time
import numpy as np
import tqdm


if __name__ == "__main__":
    G = nx.read_gpickle(
        "../data/copenhagen_protected_bicycling_graph.gpickle")
    
    node_index = utils.create_node_index(G)
    node_index_rev = utils.create_node_index(G, revert=True)
    
    # bef = time.time()
    # dm = directness.get_directness_matrix_networkx(G, separate=False)
    # aft = time.time()
    # print(aft - bef, "seconds")
    # print(directness.directness_from_matrix(dm))

    # # Still 1s per directness, but better than 4 minutes
    # # Could only parallelize this, measuring each submatrix directness
    # new_d = 0
    # choice = 0
    # for node in tqdm.tqdm(G.nodes):
    #     sdm = directness.remove_matrix_link(dm, node_index[node])
    #     if directness.directness_from_matrix(sdm) > new_d:
    #         new_d = directness.directness_from_matrix(sdm)
    #         choice = node

    
    lcc_G = G.subgraph(max(nx.connected_components(G),
                                    key=len)).copy()
    
    # bef = time.time()
    # d = directness.get_sampled_directness_networkx(
    #     lcc_G, n=100)
    # aft = time.time()
    # print("Sampled directness with 100 nodes in networkx:", d)
    # print(aft - bef, "seconds")

    # bef = time.time()
    # d = directness.get_directness_networkx(lcc_G)
    # aft = time.time()
    # print("Directness in networkx:", d)
    # print(aft - bef, "seconds")


    # bef = time.time()
    # d = directness.get_sampled_directness_igraph(
    #     lcc_H, n=2000)
    # aft = time.time()
    # print("Sampled directness with 100 nodes in igraph:",
    #       d)
    # print(aft - bef, "seconds")

    lcc_H = ig.Graph.from_networkx(lcc_G)

    # bef = time.time()
    # d = directness.get_directness_linkwise_igraph(
    #     lcc_H, n=100)
    # aft = time.time()
    # print("Linkwise directness with 100 nodes in igraph:",
    #       d)
    # print(aft - bef, "seconds")

    # bef = time.time()
    # d = directness.get_directness_igraph(lcc_H) # really long
    # aft = time.time()
    # print("Directness in igraph:", d)
    # print(aft - bef, "seconds")
