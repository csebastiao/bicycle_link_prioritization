# -*- coding: utf-8 -*-
"""
Test metrics with part of the network built, keeping the network
connected and doing so in a subtractive or additive manner on a toy graph of
Copenhagen bicycle network.
"""


from nerds_osmnx import simplification as sf
from blp import growth
from blp import plot
import networkx as nx
import osmnx as ox


if __name__ == "__main__":
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
    G = sf.multidigraph_to_graph(sim_G)

    # Make built subgraph that we can't touch during growth
    # df = ox.graph_to_gdfs(nx.MultiDiGraph(G), nodes=False, edges=True)
    # planned = df.loc[df['built'] == 0]
    G_built = nx.ego_graph(G, n, radius=RAD/2, distance='length').copy()
    nx.set_edge_attributes(G, 0, name='built')
    for edge in G.edges:
        if edge in G_built.edges:
            G.edges[edge]['built'] = 1

    name = f"../data/s{RAD}_copenhagen"
    metric_list = ['relative_coverage', 'directness']
    order = 'additive'
    # order = 'subtractive
    for metric_choice in metric_list:
        if order == 'additive':
            f_name = growth.optimize_additive_growth(G, name, metric_choice)
            plot.make_image_from_array(f_name, order=order, built=True)
            plot.make_video_from_image(f_name + "/network_images",
                                       reverse=False)
        elif order == 'subtractive':
            f_name = growth.optimize_additive_growth(G, name, metric_choice)
            plot.make_image_from_array(f_name, order=order, built=True)
            plot.make_video_from_image(f_name + "/network_images",
                                       reverse=True)
        plot.plot_coverage_directness(f_name,
                                      optimized=metric_choice, save=True)