# -*- coding: utf-8 -*-
"""
Functions to visualize results of the growth of a graph.
"""

import os
import pickle
import matplotlib as mpl
import cv2
import networkx as nx
import osmnx as ox
from blp import utils


def make_image_from_array(
        G, folder_name, array_name = None,
        order = 'subtractive', built = False, cmap = 'Reds'):
    """
    Make a folder with images for every step of the growth of the
    graph from an array of the choice of the edge removed/added with 
    the subtractive/additive growth.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Final network.
    folder_name : str
        Name of the folder for the images.
    array_name : str, optional
        Name of the array used to know the subtractive/additive order.
        If None, name of the array is based on folder_name.
        The default is None.
    order : str, optional
        Order of the history, either subtractive or additive.
        The default is 'subtractive'.
    built : bool, optional
        If True, find and color differently the already built network
        that is fixed and the planned network. The default is False.
    cmap : str, optional
        Name of the colormap used. The default is 'Reds'.

    Raises
    ------
    ValueError
        Raised if the order value is not valid.

    Returns
    -------
    None.

    """
    PAD = len(str(len(G))) # number of 0 to pad to order images
    if array_name is None: # name by default of the array
        with open(folder_name + "_arrchoice.pickle", "rb") as fp:
            choice_history = pickle.load(fp)
    else:
        with open(array_name,"rb") as fp:
            choice_history = pickle.load(fp)
    if order == 'subtractive':
        if built is False:
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
                    filepath="../data/" + folder_name
                    + f"/image_{idx+1:0{PAD}}.png",
                    save=True, show=False, close=True)
        else:
            c = mpl.cm.get_cmap(cmap) # color to see built and planned
            built_color = c(1.0) # color of the built part
            ec = ox.plot.get_edge_colors_by_attr(nx.MultiDiGraph(G),
                                                 'built', cmap = cmap)
            fig, ax = ox.plot_graph(  #this allow to save every step as a png
                nx.MultiDiGraph(G), edge_color=ec,
                filepath="../data/" + folder_name + f"/image_{0:0{PAD}}.png",
                save=True, show=False, close=True)
            xlim = ax.get_xlim() # keep same size of image for video
            ylim = ax.get_ylim()
            bb = [ylim[1], ylim[0], xlim[1], xlim[0]]
            for idx, edge in enumerate(choice_history[:-1]):
                G.remove_edge(*edge)
                G = utils.clean_isolated_node(G) # remove node without edge
                ec = ox.plot.get_edge_colors_by_attr(nx.MultiDiGraph(G),
                                                     'built', cmap = cmap)
                fig, ax = ox.plot_graph(
                    nx.MultiDiGraph(G), bbox=bb, edge_color=ec,
                    filepath="../data/" + folder_name
                    + f"/image_{idx+1:0{PAD}}.png",
                    save=True, show=False, close=True)
            # when every planned edge removed, we can't use the function
            # to find the right color, so need to put it manually
            G.remove_edge(*choice_history[-1])
            G = utils.clean_isolated_node(G) # remove node without edge
            fig, ax = ox.plot_graph(
                nx.MultiDiGraph(G), bbox=bb, edge_color=built_color,
                filepath="../data/" + folder_name
                + f"/image_{idx+2:0{PAD}}.png",
                save=True, show=False, close=True)
    elif order == 'additive':
        fig, ax = ox.plot_graph(  #this allow to have the good bounding box
            nx.MultiDiGraph(G), show=False, close=True)
        xlim = ax.get_xlim() # keep same size of image for video
        ylim = ax.get_ylim()
        bb = [ylim[1], ylim[0], xlim[1], xlim[0]]
        if built is False:
            actual_edges = []
            for idx, edge in enumerate(choice_history):
                actual_edges.append(edge)
                ec = ox.plot.get_edge_colors_by_attr(
                    nx.MultiDiGraph(G.edge_subgraph(actual_edges)),
                    'built', cmap = cmap)
                fig, ax = ox.plot_graph(
                    nx.MultiDiGraph(G.edge_subgraph(actual_edges)), bbox=bb,
                    filepath="../data/" + folder_name
                    + f"/image_{idx:0{PAD}}.png",
                    save=True, show=False, close=True)
        else:
            actual_edges = [
                edge for edge in G.edges if edge not in choice_history]
            c = mpl.cm.get_cmap(cmap)
            built_color = c(1.0)
            fig, ax = ox.plot_graph(  #this allow to save every step as a png
                nx.MultiDiGraph(G.edge_subgraph(actual_edges)),
                edge_color=built_color,  bbox=bb,
                filepath="../data/" + folder_name + f"/image_{0:0{PAD}}.png",
                save=True, show=False, close=True)
            for idx, edge in enumerate(choice_history):
                actual_edges.append(edge)
                ec = ox.plot.get_edge_colors_by_attr(
                    nx.MultiDiGraph(G.edge_subgraph(actual_edges)),
                    'built', cmap = cmap)
                fig, ax = ox.plot_graph(
                    nx.MultiDiGraph(G.edge_subgraph(actual_edges)),
                    bbox=bb, edge_color=ec,
                    filepath="../data/" + folder_name
                    + f"/image_{idx+1:0{PAD}}.png",
                    save=True, show=False, close=True)
    else:
        raise ValueError("""
                         Incorrect value for the order attribute, please
                         put subtractive or additive.
                         """)


def make_video_from_image(folder_name, reverse = False, video_name = None):
    """
    From a folder of ordered images, make a video, in the inverted 
    order if reverse is True.

    Parameters
    ----------
    folder_name : str
        Name of the folder where the images are.
    reverse : bool, optional
        If True, the order of the images in the video is reversed.
        The default is False.
    video_name : str, optional
        Name of the video. If None, the name is by default the name of the
        folder, with reverse added if reverse is True. The default is None.

    Returns
    -------
    None.

    """
    if video_name is None: # default name of the video
        video_name = folder_name
        if reverse is True:
            video_name = video_name + "_reverse.mp4"
        else:
            video_name = video_name + ".mp4"
    images = [img for img in os.listdir(folder_name) if img.endswith(".png")]
    if reverse is True: # reverse order of the images
        images.reverse() 
    # dimensions between images need to be constant
    frame = cv2.imread(os.path.join(folder_name, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 5, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(folder_name, image)))
    cv2.destroyAllWindows()
    video.release()


# TODO: Before polishing, be sure what we want to do exactly
def color_order_growth_graph(G, folder_name, array_name = None, cmap = 'jet'):
    if array_name is None:
        with open(folder_name + "_arrchoice.pickle", "rb") as fp:
            choice_history = pickle.load(fp)
    else:
        with open(array_name, "rb") as fp:
            choice_history = pickle.load(fp)
    for idx, edge in enumerate(choice_history):
        G.edges[edge]['order_color'] = idx
    for edge in G.edges:
        if edge not in choice_history:
            G.edges[edge]['order_color'] = len(G.edges)
    ec = ox.plot.get_edge_colors_by_attr(nx.MultiGraph(G), 'order_color',
                                         cmap=cmap)
    ox.plot_graph(nx.MultiGraph(G), figsize = (12, 8), bgcolor='w',
                  node_color='black', node_size=10,
                  edge_color=ec, edge_linewidth=2,
                  filepath=folder_name + "_color_order.png",
                  save=True, show=False, close=True)

