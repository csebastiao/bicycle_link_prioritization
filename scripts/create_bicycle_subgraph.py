# -*- coding: utf-8 -*-
"""

"""


import networkx as nx
import utils

cityname = "copenhagen"

raw_G = nx.read_gpickle(
    "./data/" + cityname + "_graph_simplified.gpickle")
G = utils.create_bicycle_subgraph(raw_G, attr_name='protected_bicycling')




# ox.plot_graph(nx.MultiDiGraph(lcc_G), figsize=(12, 8), bgcolor='w', 
#               node_color='black', node_size=10,
#               edge_color='r', edge_linewidth=1)

nx.write_gpickle(G, ("./data/" + cityname +
                     "_protected_bicycling_graph.gpickle"))