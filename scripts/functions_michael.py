# -*- coding: utf-8 -*-
"""

"""

import copy
import itertools 
import random
import numpy as np
import igraph as ig
import networkx as nx
import osmnx as ox
from haversine import haversine, haversine_vector
from shapely.geometry import Polygon, LineString, Point
import shapely.ops as ops
import pyproj

def dist(v1, v2):
    dist = haversine((v1['y'],v1['x']),(v2['y'],v2['x']), unit="m") # x is lon, y is lat
    return dist

def dist_vector(v1_list, v2_list):
    dist_list = haversine_vector(v1_list, v2_list, unit="m") # [(lat,lon)], [(lat,lon)]
    return dist_list


def delete_overlaps(G_res, G_orig, verbose = False):
    """Deletes inplace all overlaps of G_res with G_orig (from G_res)
    based on node ids. In other words: G_res -= G_orig
    """
    del_edges = []
    for e in list(G_res.es):
        try:
            n1_id = e.source_vertex["id"]
            n2_id = e.target_vertex["id"]
            # If there is already an edge in the original network, delete it
            n1_index = G_orig.vs.find(id = n1_id).index
            n2_index = G_orig.vs.find(id = n2_id).index
            if G_orig.are_connected(n1_index, n2_index):
                del_edges.append(e.index)
        except:
            pass
    G_res.delete_edges(del_edges)
    # Remove isolated nodes
    isolated_nodes = G_res.vs.select(_degree_eq=0)
    G_res.delete_vertices(isolated_nodes)
    if verbose: print("Removed " + str(len(del_edges)) + " overlapping edges and " + str(len(isolated_nodes)) + " nodes.")

def calculate_directness(G, numnodepairs = 500):
    """Calculate directness on G over all connected node pairs in indices. This calculation method divides the total sum of euclidian distances by total sum of network distances.
    """
    
    indices = random.sample(list(G.vs), min(numnodepairs, len(G.vs)))

    poi_edges = []
    total_distance_direct = 0
    for c, v in enumerate(indices):
        poi_edges.append(G.get_shortest_paths(v, indices[c:], weights = "length", output = "epath"))
        temp = G.get_shortest_paths(v, indices[c:], weights = "length", output = "vpath")
        total_distance_direct += sum(dist_vector([(G.vs[t[0]]["y"], G.vs[t[0]]["x"]) for t in temp], [(G.vs[t[-1]]["y"], G.vs[t[-1]]["x"]) for t in temp])) # must be in format lat,lon = y, x
    
    total_distance_network = 0
    for paths_e in poi_edges:
        for path_e in paths_e:
            # Sum up distances of path segments from first to last node
            total_distance_network += sum([G.es[e]['length'] for e in path_e])
    
    return total_distance_direct / total_distance_network

def calculate_directness_linkwise(G, numnodepairs = 500):
    """Calculate directness on G over all connected node pairs in indices. This is maybe the common calculation method: It takes the average of linkwise euclidian distances divided by network distances.
        If G has multiple components, node pairs in different components are discarded.
    """

    indices = random.sample(list(G.vs), min(numnodepairs, len(G.vs)))

    directness_links = np.zeros(int((len(indices)*(len(indices)-1))/2))
    ind = 0
    for c, v in enumerate(indices):
        poi_edges = G.get_shortest_paths(v, indices[c:], weights = "length", output = "epath")
        for c_delta, path_e in enumerate(poi_edges[1:]): # Discard first empty list because it is the node to itself
            if path_e: # if path is non-empty, meaning the node pair is in the same component
                distance_network = sum([G.es[e]['length'] for e in path_e]) # sum over all edges of path
                distance_direct = dist(v, indices[c+c_delta+1]) # dist first to last node, must be in format lat,lon = y, x

                directness_links[ind] = distance_direct / distance_network
                ind += 1
    directness_links = directness_links[:ind] # discard disconnected node pairs

    return np.mean(directness_links)


def listmean(lst): 
    try: return sum(lst) / len(lst)
    except: return 0

def calculate_coverage_edges(G, buffer_m = 500, return_cov = False, G_prev = ig.Graph(), cov_prev = Polygon()):
    """Calculates the area and shape covered by the graph's edges.
    If G_prev and cov_prev are given, only the difference between G and G_prev are calculated, then added to cov_prev.
    """

    G_added = copy.deepcopy(G)
    delete_overlaps(G_added, G_prev)

    # https://gis.stackexchange.com/questions/121256/creating-a-circle-with-radius-in-metres
    loncenter = listmean([v["x"] for v in G.vs])
    latcenter = listmean([v["y"] for v in G.vs])
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(latcenter, loncenter)
    # Use transformer: https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    wgs84_to_aeqd = pyproj.Transformer.from_proj(
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection))
    aeqd_to_wgs84 = pyproj.Transformer.from_proj(
        pyproj.Proj(local_azimuthal_projection),
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"))
    edgetuples = [((e.source_vertex["x"], e.source_vertex["y"]), (e.target_vertex["x"], e.target_vertex["y"])) for e in G_added.es]
    # Shapely buffer seems slow for complex objects: https://stackoverflow.com/questions/57753813/speed-up-shapely-buffer
    # Therefore we buffer piecewise.
    cov_added = Polygon()
    for c, t in enumerate(edgetuples):
        # if cov.geom_type == 'MultiPolygon' and c % 1000 == 0: print(str(c)+"/"+str(len(edgetuples)), sum([len(pol.exterior.coords) for pol in cov]))
        # elif cov.geom_type == 'Polygon' and c % 1000 == 0: print(str(c)+"/"+str(len(edgetuples)), len(pol.exterior.coords))
        buf = ops.transform(aeqd_to_wgs84.transform, ops.transform(wgs84_to_aeqd.transform, LineString(t)).buffer(buffer_m))
        cov_added = ops.unary_union([cov_added, Polygon(buf)])

    # Merge with cov_prev
    if not cov_added.is_empty: # We need this check because apparently an empty Polygon adds an area.
        cov = ops.unary_union([cov_added, cov_prev])
    else:
        cov = cov_prev

    cov_transformed = ops.transform(wgs84_to_aeqd.transform, cov)
    covered_area = cov_transformed.area / 1000000 # turn from m2 to km2

    if return_cov:
        return (covered_area, cov)
    else:
        return covered_area


def calculate_poiscovered(G, cov, nnids):
    """Calculates how many nodes, given by nnids, are covered by the shapely (multi)polygon cov
    """
    
    pois_indices = set()
    for poi in nnids:
        pois_indices.add(G.vs.find(id = poi).index)

    poiscovered = 0
    for poi in pois_indices:
        v = G.vs[poi]
        if Point(v["x"], v["y"]).within(cov):
            poiscovered += 1
    
    return poiscovered


def calculate_efficiency_global(G, numnodepairs = 500, normalized = True):
    """Calculates global network efficiency.
    If there are more than numnodepairs nodes, measure over pairings of a 
    random sample of numnodepairs nodes.
    """

    if G is None: return 0
    if G.vcount() > numnodepairs:
        nodeindices = random.sample(list(G.vs.indices), numnodepairs)
    else:
        nodeindices = list(G.vs.indices)
    d_ij = G.shortest_paths(source = nodeindices, target = nodeindices, weights = "length")
    d_ij = [item for sublist in d_ij for item in sublist] # flatten
    EG = sum([1/d for d in d_ij if d != 0])
    if not normalized: return EG
    pairs = list(itertools.permutations(nodeindices, 2))
    if len(pairs) < 1: return 0
    l_ij = dist_vector([(G.vs[p[0]]["y"], G.vs[p[0]]["x"]) for p in pairs],
                            [(G.vs[p[1]]["y"], G.vs[p[1]]["x"]) for p in pairs]) # must be in format lat,lon = y,x
    EG_id = sum([1/l for l in l_ij if l != 0])
    return EG / EG_id


def calculate_efficiency_local(G, numnodepairs = 500, normalized = True):
    """Calculates local network efficiency.
    If there are more than numnodepairs nodes, measure over pairings of a 
    random sample of numnodepairs nodes.
    """

    if G is None: return 0
    if G.vcount() > numnodepairs:
        nodeindices = random.sample(list(G.vs.indices), numnodepairs)
    else:
        nodeindices = list(G.vs.indices)
    EGi = []
    for i in nodeindices:
        if len(G.neighbors(i)) > 1: # If we have a nontrivial neighborhood
            G_induced = G.induced_subgraph(G.neighbors(i))
            EGi.append(calculate_efficiency_global(G_induced, numnodepairs, normalized))
    return listmean(EGi)

def edge_lengths(G):
    """Returns the total length of edges in an igraph graph.
    """
    return sum([e['length'] for e in G.es])


def intersect_igraphs(G1, G2):
    """Generates the graph intersection of igraph graphs G1 and G2, copying also link and node attributes.
    """
    # Ginter = G1.__and__(G2) # This does not work with attributes.
    if G1.ecount() > G2.ecount(): # Iterate through edges of the smaller graph
        G1, G2 = G2, G1
    inter_nodes = set()
    inter_edges = []
    inter_edge_attributes = {}
    inter_node_attributes = {}
    edge_attribute_name_list = G2.edge_attributes()
    node_attribute_name_list = G2.vertex_attributes()
    for edge_attribute_name in edge_attribute_name_list:
        inter_edge_attributes[edge_attribute_name] = []
    for node_attribute_name in node_attribute_name_list:
        inter_node_attributes[node_attribute_name] = []
    for e in list(G1.es):
        n1_id = e.source_vertex["id"]
        n2_id = e.target_vertex["id"]
        try:
            n1_index = G2.vs.find(id = n1_id).index
            n2_index = G2.vs.find(id = n2_id).index
        except ValueError:
            continue
        if G2.are_connected(n1_index, n2_index):
            inter_edges.append((n1_index, n2_index))
            inter_nodes.add(n1_index)
            inter_nodes.add(n2_index)
            edge_attributes = e.attributes()
            for edge_attribute_name in edge_attribute_name_list:
                inter_edge_attributes[edge_attribute_name].append(edge_attributes[edge_attribute_name])

    # map nodeids to first len(inter_nodes) integers
    idmap = {n_index:i for n_index,i in zip(inter_nodes, range(len(inter_nodes)))}

    G_inter = ig.Graph()
    G_inter.add_vertices(len(inter_nodes))
    G_inter.add_edges([(idmap[e[0]], idmap[e[1]]) for e in inter_edges])
    for edge_attribute_name in edge_attribute_name_list:
        G_inter.es[edge_attribute_name] = inter_edge_attributes[edge_attribute_name]

    for n_index in idmap.keys():
        v = G2.vs[n_index]
        node_attributes = v.attributes()
        for node_attribute_name in node_attribute_name_list:
            inter_node_attributes[node_attribute_name].append(node_attributes[node_attribute_name])
    for node_attribute_name in node_attribute_name_list:
        G_inter.vs[node_attribute_name] = inter_node_attributes[node_attribute_name]

    return G_inter

def simplify_ig(G):
    """Simplify an igraph with ox.simplify_graph
    """
    G_temp = copy.deepcopy(G)
    G_temp.es["length"] = G_temp.es["length"]
    output = ig.Graph.from_networkx(ox.simplify_graph(nx.MultiDiGraph(G_temp.to_networkx())).to_undirected())
    output.es["length"] = output.es["length"]
    return output

def calculate_metrics(G, GT_abstract, G_big, nnids, calcmetrics = {"length":0,
          "length_lcc":0,
          "coverage": 0,
          "directness": 0,
          "directness_lcc": 0,
          "poi_coverage": 0,
          "components": 0,
          "overlap_biketrack": 0,
          "overlap_bikeable": 0,
          "efficiency_global": 0,
          "efficiency_local": 0,
          "directness_lcc_linkwise": 0,
          "directness_all_linkwise": 0
         }, buffer_walk = 500, numnodepairs = 500, verbose = False, return_cov = True, G_prev = ig.Graph(), cov_prev = Polygon(), ignore_GT_abstract = False, Gexisting = {}):
    """Calculates all metrics (using the keys from calcmetrics).
    """
    
    output = {}
    for key in calcmetrics:
        output[key] = 0
    cov = Polygon()

    # Check that the graph has links (sometimes we have an isolated node)
    if G.ecount() > 0 and GT_abstract.ecount() > 0:

        # Get LCC
        cl = G.clusters()
        LCC = cl.giant()

        # EFFICIENCY
        if not ignore_GT_abstract:
            if verbose and ("efficiency_global" in calcmetrics or "efficiency_local" in calcmetrics): print("Calculating efficiency...")
            if "efficiency_global" in calcmetrics:
                output["efficiency_global"] = calculate_efficiency_global(GT_abstract, numnodepairs)
            if "efficiency_local" in calcmetrics:
                output["efficiency_local"] = calculate_efficiency_local(GT_abstract, numnodepairs) 
        
        # EFFICIENCY ROUTED
        if verbose and ("efficiency_global_routed" in calcmetrics or "efficiency_local_routed" in calcmetrics): print("Calculating efficiency (routed)...")
        if "efficiency_global_routed" in calcmetrics:
            try:
                output["efficiency_global_routed"] = calculate_efficiency_global(simplify_ig(G), numnodepairs)
            except:
                print("Problem with efficiency_global_routed.") # This try is needed for some pathological cases, for example loops generating empty graphs (only happened in Zurich, railwaystation/closeness)
                pass
        if "efficiency_local_routed" in calcmetrics:
            try:
                output["efficiency_local_routed"] = calculate_efficiency_local(simplify_ig(G), numnodepairs)
            except:
                print("Problem with efficiency_local_routed.") # This try is needed for some pathological cases, for example loops generating empty graphs (only happened in Zurich, railwaystation/closeness)
                pass

        # LENGTH
        if verbose and ("length" in calcmetrics or "length_lcc" in calcmetrics): print("Calculating length...")
        if "length" in calcmetrics:
            output["length"] = sum([e['weight'] for e in G.es])
        if "length_lcc" in calcmetrics:
            if len(cl) > 1:
                output["length_lcc"] = sum([e['weight'] for e in LCC.es])
            else:
                output["length_lcc"] = output["length"]
        
        # COVERAGE
        if "coverage" in calcmetrics:
            if verbose: print("Calculating coverage...")
            # G_added = G.difference(G_prev) # This doesnt work
            covered_area, cov = calculate_coverage_edges(G, buffer_walk, return_cov, G_prev, cov_prev)
            output["coverage"] = covered_area
            # OVERLAP WITH EXISTING NETS
            if Gexisting:
                if "overlap_biketrack" in calcmetrics:
                    output["overlap_biketrack"] = edge_lengths(intersect_igraphs(Gexisting["biketrack"], G))
                if "overlap_bikeable" in calcmetrics:
                    output["overlap_bikeable"] = edge_lengths(intersect_igraphs(Gexisting["bikeable"], G))

        # POI COVERAGE
        if "poi_coverage" in calcmetrics:
            if verbose: print("Calculating POI coverage...")
            output["poi_coverage"] = calculate_poiscovered(G_big, cov, nnids)

        # COMPONENTS
        if "components" in calcmetrics:
            if verbose: print("Calculating components...")
            output["components"] = len(list(G.components()))
        
        # DIRECTNESS
        if verbose and ("directness" in calcmetrics or "directness_lcc" in calcmetrics): print("Calculating directness...")
        if "directness" in calcmetrics:
            output["directness"] = calculate_directness(G, numnodepairs)
        if "directness_lcc" in calcmetrics:
            if len(cl) > 1:
                output["directness_lcc"] = calculate_directness(LCC, numnodepairs)
            else:
                output["directness_lcc"] = output["directness"]

        # DIRECTNESS LINKWISE
        if verbose and ("directness_lcc_linkwise" in calcmetrics): print("Calculating directness linkwise...")
        if "directness_lcc_linkwise" in calcmetrics:
            if len(cl) > 1:
                output["directness_lcc_linkwise"] = calculate_directness_linkwise(LCC, numnodepairs)
            else:
                output["directness_lcc_linkwise"] = calculate_directness_linkwise(G, numnodepairs)
        if verbose and ("directness_all_linkwise" in calcmetrics): print("Calculating directness linkwise (all components)...")
        if "directness_all_linkwise" in calcmetrics:
            if "directness_lcc_linkwise" in calcmetrics and len(cl) <= 1:
                output["directness_all_linkwise"] = output["directness_lcc_linkwise"]
            else: # we have >1 components
                output["directness_all_linkwise"] = calculate_directness_linkwise(G, numnodepairs) # number of components is checked within calculate_directness_linkwise()

    if return_cov: 
        return (output, cov)
    else:
        return output
