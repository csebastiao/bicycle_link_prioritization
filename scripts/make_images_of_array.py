# -*- coding: utf-8 -*-
"""
From a pickle file with the order of the removed edge, make images of 
the evolution of the graph in a subtractive order.
"""


import pickle

import networkx as nx
import osmnx as ox

from blp import utils


if __name__ == "__main__":
    G = nx.read_gpickle("../data/s2000_copenhagen_bicycle_graph.gpickle")
    folder_name = "s2000_copenhagen_connected_relative_coverage"
    PAD = len(str(len(G)))
    with open(
            "../data/arrchoice_" + folder_name + ".pickle",
            "rb") as fp:
        choice_history = pickle.load(fp)
    fig, ax = ox.plot_graph(  #this allow to save every step as a png
        nx.MultiDiGraph(G),
        filepath="../data/" + folder_name + f"/image_{0:0{PAD}}.png",
        save=True, show=False, close=True)
    xlim = ax.get_xlim() # keep same size of image for video
    ylim = ax.get_ylim()
    bb = [ylim[1], ylim[0], xlim[1], xlim[0]]
    for idx, edge in enumerate(choice_history):
        G.remove_edge(*edge)
        G = utils.clean_isolated_node(G) # remove node without edge
        fig, ax = ox.plot_graph(
            nx.MultiDiGraph(G), bbox=bb,
            filepath="../data/" + folder_name + f"/image_{idx+1:0{PAD}}.png",
            save=True, show=False, close=True)
   