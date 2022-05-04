# -*- coding: utf-8 -*-
"""
Take a smaller and connected part of the bicycle network in Copenhagen
to test more quickly the workflow. Measure the linkwise directness
of the entire network for every choice at every step, the best one
but also the longest one. Alternative, just change the structure 
so that we don't measure again the entire euclidian distance for
every value every time, using the same euclidian distance matrix 
and just removing node that are not here anymore. Slighlty quicker, 
scale better with the number of nodes.
"""


# Profiling
import cProfile
import pstats
import io

# Custom packages
from nerds_osmnx import simplification as sf
from blp import directness
from blp import utils

# Network extraction, analysis and manipulation
import networkx as nx
import osmnx as ox

# Math
import numpy as np

# Visualization
from matplotlib import pyplot as plt


if __name__ == "__main__":
    pr = cProfile.Profile() # profiler to see what takes time in the script
    pr.enable()

    com_G = nx.read_gpickle(
        "../data/copenhagen_protected_bicycling_graph.gpickle")
    lcc_G = com_G.subgraph(max(nx.connected_components(com_G), key=len)).copy()
    node_pos = [12.5500, 55.6825] # find central node
    n = ox.nearest_nodes(lcc_G, *node_pos)
    RAD = 2000 # make subgraph as radius around central node
    rad_G = nx.ego_graph(lcc_G, n, radius=RAD, distance='length')
    rad_G.graph['simplified'] = False
    sim_G = sf.momepy_simplify_graph(nx.MultiDiGraph(rad_G)) # simplify
    fin_G = sf.multidigraph_to_graph(sim_G)
    G = fin_G.copy()
    node_index = utils.create_node_index(G)
    sm = directness.get_shortest_network_path_matrix(G)
    em = directness.get_euclidian_distance_matrix(G)
    dm = directness.avoid_zerodiv_matrix(em, sm)
    d = directness.directness_from_matrix(dm)
    d_history = [d]
    choice_history = []

    PAD = len(str(len(G))) # know how many 0 you need to pad for png name
    folder_name = ("s" + f"{RAD}" +
                   "_copenhagen_linkwise_directness_every_edge_alt")

    fig, ax = ox.plot_graph(  #this allow to save every step as a png
        nx.MultiDiGraph(G),
        filepath="../data/" + folder_name + f"/image_{0:0{PAD}}.png",
        save=True, show=False, close=True)
    xlim = ax.get_xlim() # keep same size of image for video
    ylim = ax.get_ylim()
    bb = [ylim[1], ylim[0], xlim[1], xlim[0]]

    COUNT = 1
    while len(G) > 2:
        batch_d = []
        batch_choice = []
        for u, v in G.edges:
            H = G.copy()
            H.remove_edge(u, v)
            new_sm = directness.get_shortest_network_path_matrix(H)
            sdm = directness.avoid_zerodiv_matrix(em, new_sm) # new directness
            batch_d.append(directness.directness_from_matrix(sdm))
            batch_choice.append([u, v])
        batch = zip(batch_d, batch_choice)
        new_d, choice = max(batch) # find max directness + edge we remove
        d_history.append(new_d)
        choice_history.append(choice)
        G.remove_edge(*choice) # remove edge that maximize directness


        node_removed = utils.find_isolated_node(G) # find node without edge
        for n in node_removed:
            em = np.delete(em, node_index[n], 0) # delete row
            em = np.delete(em, node_index[n], 1) # delete column
            G.remove_node(n) # remove the isolated node
            node_index = utils.create_node_index(G) # modify index

        ## Same as this, slighlty quicker
        # G = utils.clean_isolated_node(G) # remove node without edge
        # em = directness.get_euclidian_distance_matrix(G)
        fig, ax = ox.plot_graph(
            nx.MultiDiGraph(G), bbox=bb,
            filepath="../data/" + folder_name + f"/image_{COUNT:0{PAD}}.png",
            save=True, show=False, close=True)
        COUNT += 1

    plt.figure(figsize=(12,8)) # evolution of directness
    plt.plot(range(len(d_history)), d_history, linewidth=5)
    plt.xlabel("Step")
    plt.ylabel("Linkwise directness")
    plt.savefig("../data/plot_" + folder_name + ".png")

    pr.disable()
    s = io.StringIO() # get results of profiler in a text file
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('../data/profile_' + folder_name + '.txt', 'w+') as f:
        f.write(s.getvalue())