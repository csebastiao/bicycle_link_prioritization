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
    RAD = 2000 # make subgraph as radius around central node
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
    import pyproj
    import shapely
    local_proj = 'epsg:25832'
    buff_size = 200
    proj = pyproj.Transformer.from_proj(
            pyproj.Proj(init='epsg:4326'),
            pyproj.Proj(local_proj))
    
    import geopandas as gpd
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_axis_off()
    buff_size_list = [500, 200, 2]
    color_list = [0.1, 0.5, 1]
    color_dict = dict(zip(buff_size_list, color_list))
    for buff_size in buff_size_list:
        test_buff = dict()
        for edge in G.edges:
            test_buff[edge] = shapely.ops.transform(
                proj.transform, G.edges[edge]['geometry']).buffer(buff_size)
        poly = shapely.ops.unary_union(list(test_buff.values()))
        p = gpd.GeoSeries(poly)
        p.plot(ax=ax, color='dodgerblue', alpha=color_dict[buff_size])

    

    name = f"../data/s{RAD}_copenhagen"
    metric_list = ['relative_coverage', 'directness']
    orders = ['subtractive', 'additive']
    list_built = [False]
    list_connected = [True]
    for built in list_built:
        for connected in list_connected:
            for order in orders:
                for metric_choice in metric_list:
                    if order == 'additive':
                        f_name = growth.optimize_additive_growth(
                            G, name, metric_choice, local_proj,
                            buff_size=buff_size, override_naming=False,
                            built=built, keep_connected=connected,
                            profiling=True, save_network=True,
                            save_metrics=True)
                        plot.make_image_from_array(f_name, G=None, order=order,
                                                   built=built, cmap='Reds')
                        plot.make_video_from_image(f_name + "/network_images",
                                                   reverse=False,
                                                   video_name=None, fps=5)
                    elif order == 'subtractive':
                        f_name = growth.optimize_subtractive_growth(
                            G, name, metric_choice, local_proj,
                            buff_size=buff_size, override_naming=False,
                            built=built, keep_connected=connected,
                            profiling=True, save_network=True,
                            save_metrics=True)
                        plot.make_image_from_array(f_name, G=None, order=order,
                                                   built=built, cmap='Reds')
                        plot.make_video_from_image(f_name + "/network_images",
                                                   reverse=True,
                                                   video_name=None, fps=5)
                    plot.plot_coverage_directness(f_name,
                                                  optimized=metric_choice, 
                                                  coverage_name=None,
                                                  directness_name=None,
                                                  save=True)