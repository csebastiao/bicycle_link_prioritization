# -*- coding: utf-8 -*-
"""

"""

# Custom packages
from nerds_osmnx import simplification as sf
from blp import directness
from blp import utils

# Network extraction, analysis and manipulation
import networkx as nx
import osmnx as ox

if __name__ == "__main__":
    com_G = nx.read_gpickle(
        "../data/copenhagen_protected_bicycling_graph.gpickle")
    lcc_G = com_G.subgraph(max(nx.connected_components(com_G),
                               key=len)).copy()
    node_pos = [12.5500, 55.6825] # find central node
    n = ox.nearest_nodes(lcc_G, *node_pos)
    RAD = 2000 # make subgraph as radius around central node
    rad_G = nx.ego_graph(lcc_G, n, radius=RAD, distance='length')
    rad_G.graph['simplified'] = False
    sim_G = sf.momepy_simplify_graph(nx.MultiDiGraph(rad_G)) # simplify
    fin_G = sf.multidigraph_to_graph(sim_G)
    G = ox.project_graph(sim_G, to_crs='EPSG:25832')
    H = ox.consolidate_intersections(G, dead_ends=True)
    H_fin = sf.multidigraph_to_graph(H)
    
    