# -*- coding: utf-8 -*-
"""
Useful functions
"""


import numpy as np
import networkx as nx


def create_bicycle_subgraph(G, attr_name = 'protected_bicycling'):
    """
    Make a subgraph of the graph G as to create a bicycle network
    from a road network with a boolean attribute for bikeable street.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Road network with an attribute for bikeable street.
    attr_name : str, optional
        Name of the boolean attribute saying whether or not there are 
        protected bicycling infrastructure.
        The default is 'protected_bicycling'.

    Returns
    -------
    bicycle_subgraph : networkx.classes.graph.Graph
        Protected bicycling infrastructure subgraph.

    """
    df = nx.to_pandas_edgelist(G)
    filtered_df = df[df[attr_name] == True]
    edgelist = filtered_df.loc[:, ['source', 'target']].values.tolist()
    edgelist = [tuple(x) for x in edgelist] # In order to make it hashable
    bicycle_subgraph = G.edge_subgraph(edgelist).copy()
    # crs = G.graph['crs'] # Alt, as quick but more attribute manip
    # bicycle_subgraph = nx.from_pandas_edgelist(filtered_df, edge_attr=True) 
    # bicycle_subgraph.graph['crs'] = crs
    # for node in bicycle_subgraph.nodes():
    #     for attr in G.nodes[node]:
    #         bicycle_subgraph.nodes[node][attr] = G.nodes[node][attr]
    return bicycle_subgraph


def make_summary(G):
    """Print a summary of useful information on the graph G."""
    lcc_G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print("""
          Number of nodes: {} \n
          Number of edges: {} \n
          Number of connected components: {} \n
          Size of the largest connected components: {} nodes and {} edges
          """.format(
          len(G.nodes), len(G.edges), len(list(nx.connected_components(G))),
          len(lcc_G.nodes), len(lcc_G.edges)
          ))


def get_node_positions(G, package = 'networkx'):
    """Get array of coordinates of every node of the graph G."""
    if package == 'networkx':
        lon = list(nx.get_node_attributes(G, 'x').values())
        lat = list(nx.get_node_attributes(G, 'y').values())
    elif package == 'igraph':
        lon = G.vs.get_attribute_values('x')
        lat = G.vs.get_attribute_values('y')
    else:
        raise ValueError("package need to be networkx or igraph")
    pos_list = np.transpose(np.array([lat, lon]))
    return pos_list

