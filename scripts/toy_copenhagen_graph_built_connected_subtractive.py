# -*- coding: utf-8 -*-
"""
Test metrics with part of the network built, keeping the network
connected and doing so in a subtractive manner on a toy graph of
Copenhagen bicycle network.
"""

# System
import os

# Profiling
import cProfile
import pstats
import io
import tqdm

# Save
import pickle

# Custom packages
from nerds_osmnx import simplification as sf
from blp import metrics
from blp import utils
from blp import growth

# Network extraction, analysis and manipulation
import networkx as nx
import osmnx as ox

# Math
import numpy as np

# Geometry
import shapely

# Visualization
from matplotlib import pyplot as plt
import matplotlib as mpl


if __name__ == "__main__":
    mpl.rcParams.update({'font.size': 16})
    metric_list = ['relative_coverage', 'directness', 'relative_directness']
    for metric_choice in metric_list:
        pr = cProfile.Profile() # profiler to see what takes time in the script
        pr.enable()
        com_G = nx.read_gpickle(
            "../data/all_graph/copenhagen_protected_bicycling_graph.gpickle")
        lcc_G = com_G.subgraph(max(nx.connected_components(com_G),
                                   key=len)).copy()
        node_pos = [12.5500, 55.6825] # find central node
        n = ox.nearest_nodes(lcc_G, *node_pos)
        RAD = 1000 # make subgraph as radius around central node
        rad_G = nx.ego_graph(lcc_G, n, radius=RAD, distance='length')
        rad_G.graph['simplified'] = False
        sim_G = sf.momepy_simplify_graph(nx.MultiDiGraph(rad_G)) # simplify
        fin_G = sf.multidigraph_to_graph(sim_G)
        G = fin_G.copy()
        # Make built subgraph that we can't touch during growth
        G_built = nx.ego_graph(G, n, radius=RAD/2, distance='length').copy()
        nx.set_edge_attributes(G, 0, name='built')
        for edge in G.edges:
            if edge in G_built.edges:
                G.edges[edge]['built'] = 1
        # df = ox.graph_to_gdfs(nx.MultiDiGraph(G), nodes=False, edges=True)
        # planned = df.loc[df['built'] == 0]
        edgelist = [(u, v) for u, v in G.edges if G.edges[u, v]['built'] == 0]

        folder_name = (
            "../data/s" + f"{RAD}" 
            + f"_copenhagen_built_connected_subtractive_{metric_choice}")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
    
        # Coverage
        BUFF_SIZE = 0.002
        geom = dict()
        for edge in G.edges:
            geom[edge] = G.edges[edge]['geometry'].buffer(BUFF_SIZE)
        bef_area = shapely.ops.unary_union(list(geom.values())).area
        area_history = [bef_area]
    
        # Directness
        node_index = utils.create_node_index(G)
        sm = metrics.get_shortest_network_path_matrix(G)
        em = metrics.get_euclidean_distance_matrix(G)
        dm = metrics.avoid_zerodiv_matrix(em, sm)
        d = metrics.directness_from_matrix(dm)
        d_history = [d]
    
        choice_history = []
        for i in tqdm.tqdm(range(len(edgelist))):
            if metric_choice == 'directness':
                new_m, choice = growth.directness_subtractive_step(
                    G, edgelist, em, keep_connected = True)
            if metric_choice == 'relative_directness':
                new_m, choice = growth.relative_directness_subtractive_step(
                    G, edgelist, sm, keep_connected = True)
            elif metric_choice == 'relative_coverage':
                new_m, choice = growth.relative_coverage_subtractive_step(
                    G, edgelist, bef_area, geom, keep_connected = True)
            choice_history.append(choice)
            G.remove_edge(*choice)
            edgelist.remove(choice)
    
            # Coverage
            geom.pop(choice)
            bef_area = shapely.ops.unary_union(list(geom.values())).area
            area_history.append(bef_area)
    
            node_removed = utils.find_isolated_node(G) # find node without edge
            for n in node_removed:
                em = np.delete(em, node_index[n], 0) # delete row
                em = np.delete(em, node_index[n], 1) # delete column
                sm = np.delete(sm, node_index[n], 0) # delete row
                sm = np.delete(sm, node_index[n], 1) # delete column
                G.remove_node(n) # remove the isolated node
                node_index = utils.create_node_index(G) # modify index
    
            # Directness
            d_history.append(metrics.directness_from_matrix(
                metrics.avoid_zerodiv_matrix(
                    em, metrics.get_shortest_network_path_matrix(G))))
            
        with open(folder_name + "/arrdir.pickle", "wb") as fp:
            pickle.dump(d_history, fp)
        with open(folder_name + "/arrcov.pickle", "wb") as fp:
            pickle.dump(area_history, fp)
        with open(folder_name + "/arrchoice.pickle", "wb") as fp:
            pickle.dump(choice_history, fp)
        pr.disable()
        s = io.StringIO() # get results of profiler in a text file
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
    
        with open(folder_name + '/profile.txt', 'w+') as f:
            f.write(s.getvalue())


