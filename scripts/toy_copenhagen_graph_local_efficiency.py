# -*- coding: utf-8 -*-
"""
Take a smaller and connected part of the bicycle network in Copenhagen
to test more quickly the workflow. Measure the local efficiency, using the
same tools as for the directness.
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
    # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
    sm = np.divide(np.ones(sm.shape), sm,
                   out=np.zeros_like(np.ones(sm.shape)), where=sm!=0)
    em = directness.get_euclidian_distance_matrix(G)
    em = np.divide(np.ones(em.shape), em,
                   out=np.zeros_like(np.ones(em.shape)), where=em!=0)
    em[sm == 0.] = 0.
    ge = np.sum(sm) / np.sum(em)
    ge_history = [ge]
    choice_history = []

    PAD = len(str(len(G))) # know how many 0 you need to pad for png name
    folder_name = ("s" + f"{RAD}" +
                   "_copenhagen_global_efficiency")

    fig, ax = ox.plot_graph(  #this allow to save every step as a png
        nx.MultiDiGraph(G),
        filepath="../data/" + folder_name + f"/image_{0:0{PAD}}.png",
        save=True, show=False, close=True)
    xlim = ax.get_xlim() # keep same size of image for video
    ylim = ax.get_ylim()
    bb = [ylim[1], ylim[0], xlim[1], xlim[0]]

    COUNT = 1
    while len(G) > 2:
        batch_ge = []
        batch_choice = []
        for u, v in G.edges:
            H = G.copy()
            H.remove_edge(u, v)
            new_em = em.copy()
            new_sm = directness.get_shortest_network_path_matrix(H)
            new_sm = np.divide(np.ones(new_sm.shape), new_sm,
                           out=np.zeros_like(np.ones(new_sm.shape)),
                           where=new_sm!=0)
            new_em[new_sm == 0.] = 0.
            nge = np.sum(new_sm) / np.sum(new_em)
            batch_ge.append(nge)
            batch_choice.append([u, v])
        batch = zip(batch_ge, batch_choice)
        new_ge, choice = max(batch) # find max directness + edge we remove
        ge_history.append(new_ge)
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
    plt.plot(range(len(ge_history)), ge_history, linewidth=5)
    plt.xlabel("Step")
    plt.ylabel("Global Efficiency")
    plt.savefig("../data/plot_" + folder_name + ".png")

    pr.disable()
    s = io.StringIO() # get results of profiler in a text file
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('../data/profile_' + folder_name + '.txt', 'w+') as f:
        f.write(s.getvalue())