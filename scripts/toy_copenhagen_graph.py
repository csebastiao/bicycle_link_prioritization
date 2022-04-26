# -*- coding: utf-8 -*-
"""
Take a smaller and connected part of the bicycle network in Copenhagen
to test more quickly the workflow
"""


import networkx as nx
import nerds_osmnx.simplification as sf
import blp.directness as directness
import blp.utils as utils
import numpy as np
import tqdm
import osmnx as ox
from matplotlib import pyplot as plt

if __name__ == "__main__":
    com_G = nx.read_gpickle(
        "../data/copenhagen_protected_bicycling_graph.gpickle")
    lcc_G = com_G.subgraph(max(nx.connected_components(com_G), key=len)).copy()
    node_pos = [12.5500, 55.6825]
    n = ox.nearest_nodes(lcc_G, *node_pos)
    rad_G = nx.ego_graph(lcc_G, n, radius=2000, distance='length')
    rad_G.graph['simplified'] = False
    sim_G = sf.momepy_simplify_graph(nx.MultiDiGraph(rad_G))
    fin_G = sf.multidigraph_to_graph(sim_G)
    G = fin_G.copy()
    
    # ox.plot_graph(nx.MultiDiGraph(lcc_G))
    # ox.plot_graph(nx.MultiDiGraph(rad_G))
    # ox.plot_graph(sim_G)
    
    node_index = utils.create_node_index(G)
    node_index_rev = utils.create_node_index(G, revert=True)

    dm = directness.get_directness_matrix_networkx(G, separate=False)
    d = directness.directness_from_matrix(dm)
    d_history = [d]
    choice_history = []
    
    for i in tqdm.tqdm(range(len(G) - 2)):
        # if i%20 == 0:
        #     ox.plot_graph(nx.MultiDiGraph(G))
        new_d = 0
        choice = 0
        for node in G.nodes:
            sdm = directness.remove_matrix_link(dm, node_index[node])
            if directness.directness_from_matrix(sdm) > new_d:
                new_d = directness.directness_from_matrix(sdm)
                choice = node
        d_history.append(new_d)
        choice_history.append(choice)
        G.remove_node(choice)
        dm[node_index[choice],:] = 0
        dm[:, node_index[choice]] = 0

    plt.figure(figsize=(12,8))
    plt.plot(np.arange(len(d_history)), d_history, linewidth=5)
    plt.xlabel("Step")
    plt.ylabel("Linkwise directness")
