# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Observe multiple metrics on a toy graph while optimizing for one only.
"""


# Profiling
import cProfile
import pstats
import io

# Save
import pickle

# Custom packages
from nerds_osmnx import simplification as sf
from blp import directness
from blp import utils

# Network extraction, analysis and manipulation
import networkx as nx
import osmnx as ox

# Math
import numpy as np

# Geometry
import shapely

# Visualization
from matplotlib import pyplot as plt
from matplotlib import rcParams


if __name__ == "__main__":
    pr = cProfile.Profile() # profiler to see what takes time in the script
    pr.enable()
    rcParams.update({'font.size': 16})
    metrics = ['relative_coverage', 'coverage']
    for metric_choice in metrics:
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
        G = fin_G.copy()
    
        # Coverage
        BUFF_SIZE = 0.002
        geom = dict()
        for edge in G.edges:
            geom[edge] = G.edges[edge]['geometry'].buffer(BUFF_SIZE)
        bef_area = shapely.ops.unary_union(list(geom.values())).area
        area_history = [bef_area]
    
        # Directness
        node_index = utils.create_node_index(G)
        sm = directness.get_shortest_network_path_matrix(G)
        em = directness.get_euclidian_distance_matrix(G)
        dm = directness.avoid_zerodiv_matrix(em, sm)
        d = directness.directness_from_matrix(dm)
        d_history = [d]
    
        # Global efficiency
        ism = np.divide(np.ones(sm.shape), sm,
                       out=np.zeros_like(np.ones(sm.shape)), where=sm!=0)
        iem = np.divide(np.ones(em.shape), em,
                       out=np.zeros_like(np.ones(em.shape)), where=em!=0)
        iem[ism == 0.] = 0.
        ge = np.sum(ism) / np.sum(iem)
        ge_history = [ge]
    
        choice_history = []
        while len(G) > 2:
            batch_m = []
            batch_choice = []
            if metric_choice == 'directness':
                for edge in G.edges:
                    H = G.copy()
                    H.remove_edge(*edge)
                    if (nx.number_connected_components(H) > 1) and (
                            len(sorted(nx.connected_components(H),
                                       key=len)[0]) > 1):
                        pass
                    else:
                        new_sm = directness.get_shortest_network_path_matrix(H)
                        sdm = directness.avoid_zerodiv_matrix(em, new_sm)
                        batch_m.append(directness.directness_from_matrix(sdm))
                        batch_choice.append(edge)
            elif metric_choice == 'global_efficiency':
                for edge in G.edges:
                    H = G.copy()
                    H.remove_edge(*edge)
                    if (nx.number_connected_components(H) > 1) and (
                            len(sorted(nx.connected_components(H),
                                       key=len,)[0]) > 1):
                        pass
                    else:
                        new_em = iem.copy()
                        new_sm = directness.get_shortest_network_path_matrix(H)
                        new_sm = np.divide(
                            np.ones(new_sm.shape), new_sm,
                            out=np.zeros_like(np.ones(new_sm.shape)),
                            where=new_sm!=0)
                        new_em[new_sm == 0.] = 0.
                        nge = np.sum(new_sm) / np.sum(new_em)
                        batch_m.append(nge)
                        batch_choice.append(edge)
            elif metric_choice == 'coverage':
                for edge in G.edges:
                    H = G.copy()
                    H.remove_edge(*edge)
                    if (nx.number_connected_components(H) > 1) and (
                            len(sorted(nx.connected_components(H),
                                       key=len)[0]) > 1):
                        pass
                    else:
                        temp_g = geom.copy()
                        temp_g.pop(edge)
                        batch_m.append(shapely.ops.unary_union(
                            list(temp_g.values())).area)
                        batch_choice.append(edge)
            elif metric_choice == 'relative_coverage':
                for edge in G.edges:
                    H = G.copy()
                    H.remove_edge(*edge)
                    if (nx.number_connected_components(H) > 1) and (
                            len(sorted(nx.connected_components(H),
                                       key=len)[0]) > 1):
                        pass
                    else:
                        temp_g = geom.copy()
                        temp_g.pop(edge)
                        area = shapely.ops.unary_union(
                            list(temp_g.values())).area
                        batch_m.append((bef_area - area) 
                                       / G.edges[edge]['length'])
                        batch_choice.append(edge)
            batch = zip(batch_m, batch_choice)
            if metric_choice == 'relative_coverage':
                new_m, choice = min(batch)
            else:
                new_m, choice = max(batch)
            choice_history.append(choice)
            G.remove_edge(*choice)
    
            # Coverage
            geom.pop(choice)
            bef_area = shapely.ops.unary_union(list(geom.values())).area
            area_history.append(bef_area)
    
            node_removed = utils.find_isolated_node(G) # find node without edge
            for n in node_removed:
                em = np.delete(em, node_index[n], 0) # delete row
                em = np.delete(em, node_index[n], 1) # delete column
                iem = np.delete(iem, node_index[n], 0) # delete row
                iem = np.delete(iem, node_index[n], 1) # delete column
                G.remove_node(n) # remove the isolated node
                node_index = utils.create_node_index(G) # modify index
    
            # Directness
            d_history.append(directness.directness_from_matrix(
                directness.avoid_zerodiv_matrix(
                    em, directness.get_shortest_network_path_matrix(G))))
    
            # Global efficiency
            ism = directness.get_shortest_network_path_matrix(G)
            ism = np.divide(np.ones(ism.shape), ism,
                           out=np.zeros_like(np.ones(ism.shape)), where=ism!=0)
            iem[ism == 0.] = 0.
            ge_history.append(np.sum(ism) / np.sum(iem))
        

        if metric_choice == 'coverage':
            colors = ['r', 'b', 'b']
            metric_history = area_history
        elif metric_choice == 'relative_coverage':
            colors = ['r', 'b', 'b']
            metric_history = area_history
        elif metric_choice == 'directness':
            colors = ['b', 'r', 'b']
            metric_history = d_history
        elif metric_choice == 'global_efficiency':
            colors = ['b', 'b', 'r']
            metric_history = ge_history
        fig, axs = plt.subplots(1, 3, figsize=(24, 12))
        axs[0].plot(range(len(area_history)), area_history,
                    linewidth=5, color=colors[0])
        axs[0].set_xlabel("Step")
        axs[0].set_ylabel("Coverage")
        axs[0].set_title("Area under curve (xy_normalized):{}".format(
            round(utils.get_area_under_curve(
                area_history, normalize_y=True, normalize_x=True), 3)))
    
        axs[1].plot(range(len(d_history)), d_history,
                    linewidth=5, color=colors[1])
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel("Linkwise directness")
        axs[1].set_title("Area under curve (x_normalized):{}".format(
            round(utils.get_area_under_curve(
                d_history, normalize_x=True), 3)))
    
        axs[2].plot(range(len(ge_history)), ge_history,
                    linewidth=5, color=colors[2])
        axs[2].set_xlabel("Step")
        axs[2].set_ylabel("Global Efficiency")
        axs[2].set_title("Area under curve (x_normalized):{}".format(
            round(utils.get_area_under_curve(
                ge_history, normalize_x=True), 3)))
        fig.savefig(f"../data/plot_s{RAD}_copenhagen_connected_mulobs_{metric_choice}")
        plt.close(fig)
        
        with open(
                f"../data/arrmetric_s{RAD}_copenhagen_connected_{metric_choice}.pickle",
                "wb") as fp:
            pickle.dump(metric_history, fp)
        with open(
                f"../data/arrchoice_s{RAD}_copenhagen_connected_{metric_choice}.pickle",
                "wb") as fp:
            pickle.dump(choice_history, fp)


    pr.disable()
    s = io.StringIO() # get results of profiler in a text file
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open(f'../data/profile_s{RAD}_copenhagen_connected_mulobs.txt', 'w+') as f:
        f.write(s.getvalue())