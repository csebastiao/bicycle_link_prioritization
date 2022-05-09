# -*- coding: utf-8 -*-
"""

"""

import pickle

import networkx as nx
import osmnx as ox

if __name__ == "__main__":
    G = nx.read_gpickle("../data/s2000_copenhagen_bicycle_graph.gpickle")
    folder_name = "s2000_copenhagen_coverage"
    with open(
            "../data/arrchoice_" + folder_name + ".pickle",
            "rb") as fp:
        choice_history = pickle.load(fp)
    for idx, edge in enumerate(choice_history):
        G.edges[edge]['order_color'] = idx
    for edge in G.edges:
        if edge not in choice_history:
            G.edges[edge]['order_color'] = len(G.edges)
    ec = ox.plot.get_edge_colors_by_attr(G, 'order_color',
                                         cmap='jet')
    G = nx.MultiGraph(G)
    # Red is protected, blue unprotected
    ox.plot_graph(G, figsize = (12, 8), bgcolor='w',
                  node_color='black', node_size=10,
                  edge_color=ec, edge_linewidth=2,
                  filepath="../data/" + folder_name + "_color_order.png",
                  save=True)
