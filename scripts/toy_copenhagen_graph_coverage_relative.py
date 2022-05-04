# -*- coding: utf-8 -*-
"""

"""


# Profiling
import cProfile
import pstats
import io

# Custom packages
from nerds_osmnx import simplification as sf
from blp import utils

# Network extraction, analysis and manipulation
import networkx as nx
import osmnx as ox

# Geometry
import shapely

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

    BUFF_SIZE = 0.002
    geom = dict()
    for edge in G.edges:
        geom[edge] = G.edges[edge]['geometry'].buffer(BUFF_SIZE)
    bef_area = shapely.ops.unary_union(list(geom.values())).area
    c_history = [0]
    area_history = [bef_area]
    choice_history = []

    PAD = len(str(len(G))) # know how many 0 you need to pad for png name
    folder_name = ("s" + f"{RAD}" +
                   "_copenhagen_relative_coverage")

    fig, ax = ox.plot_graph(  #this allow to save every step as a png
        nx.MultiDiGraph(G),
        filepath="../data/" + folder_name + f"/image_{0:0{PAD}}.png",
        save=True, show=False, close=True)
    xlim = ax.get_xlim() # keep same size of image for video
    ylim = ax.get_ylim()
    bb = [ylim[1], ylim[0], xlim[1], xlim[0]]

    COUNT = 1
    while len(G) > 2:
        batch_c = []
        batch_choice = []
        for edge in G.edges:
            temp_g = geom.copy()
            temp_g.pop(edge)
            area = shapely.ops.unary_union(list(temp_g.values())).area
            batch_c.append((bef_area - area) / G.edges[edge]['length'])
            batch_choice.append(edge)
        batch = zip(batch_c, batch_choice)
        new_c, choice = min(batch)
        geom.pop(choice)
        bef_area = shapely.ops.unary_union(list(geom.values())).area
        area_history.append(bef_area)
        c_history.append(new_c)
        choice_history.append(choice)
        G.remove_edge(*choice)
        G = utils.clean_isolated_node(G) # remove node without edge
        fig, ax = ox.plot_graph(
            nx.MultiDiGraph(G), bbox=bb,
            filepath="../data/" + folder_name + f"/image_{COUNT:0{PAD}}.png",
            save=True, show=False, close=True)
        COUNT += 1

    plt.figure(figsize=(12,8)) # evolution of directness
    plt.plot(range(len(area_history)), area_history, linewidth=5)
    plt.xlabel("Step")
    plt.ylabel("Coverage")
    plt.savefig("../data/plot_" + folder_name + ".png")

    plt.figure(figsize=(12,8)) # evolution of directness
    plt.plot(range(len(c_history)), c_history, linewidth=5)
    plt.xlabel("Step")
    plt.ylabel("Relative coverage")
    plt.savefig("../data/plot_" + folder_name + "_relative.png")

    pr.disable()
    s = io.StringIO() # get results of profiler in a text file
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('../data/profile_' + folder_name + '.txt', 'w+') as f:
        f.write(s.getvalue())