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

if __name__ == "__main__":
    G = nx.read_gpickle(
        "../data/copenhagen_protected_bicycling_graph.gpickle")
    
    node_index = utils.create_node_index(G)
    
    bef = time.time()
    sm, em = directness.get_directness_matrix_networkx(G, separate=True)
    aft = time.time()
    print(aft - bef, "seconds")
    print(directness.directness_from_matrix(np.divide(em, sm)))
    
    # lcc_G = G.subgraph(max(nx.connected_components(G),
    #                                key=len)).copy()
    
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

    # lcc_H = ig.Graph.from_networkx(lcc_G)

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
