from read_dict_to_make_loe_lon import get_my_graph
from GraphConverter import GraphConverter

import networkx as nx
import numpy as np
import pandas as pd


def degree_centrality(G):
    if len(G) <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for n, d in G.degree(weight='weight')}
    return centrality

def get_the_whole_graph(filename):
    my_graph_obj_dict = get_my_graph(filename)
    nx_graph_modelled = GraphConverter().convert_my_graph_to_nx_graph(my_graph_obj_dict.list_of_nodes, my_graph_obj_dict.list_of_edges)
    return nx_graph_modelled

def get2HopDcGivenNX(myNx):
    dictToBeReturned = {}
    for n in myNx.__iter__():
        listOfNeighbors = []
        neighbors1 = myNx.neighbors(n)
        for n1 in neighbors1:
            listOfNeighbors.append(str(n1))
            neighbors2 = myNx.neighbors(n1)
            for n2 in neighbors2:
                listOfNeighbors.append(str(n2))
        listOfNeighbors = set(listOfNeighbors)
        listOfNeighbors.remove(n)
        dictToBeReturned[n]=(len(listOfNeighbors)*1.0)/len(myNx)
    return dictToBeReturned


nx_graph_modelled_pos = get_the_whole_graph('Pos221_edgetable.csv')
nx_graph_modelled_neg = get_the_whole_graph('Neg221_edgetable.csv')
print(len(nx_graph_modelled_neg))

print(len(list(nx.connected_component_subgraphs(nx_graph_modelled_pos))))
print(len(list(nx.connected_component_subgraphs(nx_graph_modelled_neg))))

a = list(nx.connected_component_subgraphs(nx_graph_modelled_neg))

nx_graph_modelled_neg = max(nx.connected_component_subgraphs(nx_graph_modelled_neg), key=len)
print(len(nx_graph_modelled_neg))
print(len(list(nx.connected_component_subgraphs(nx_graph_modelled_neg))))


degrees_pos = nx_graph_modelled_pos.degree(weight='weight')
nodes_sorted_on_degree_pos = list(dict(sorted(degrees_pos, key=lambda x: x[1], reverse=True)).keys())
lst1 = nodes_sorted_on_degree_pos[:20]
print(lst1)

degrees_neg = nx_graph_modelled_neg.degree(weight='weight')
nodes_sorted_on_degree_neg = list(dict(sorted(degrees_neg, key=lambda x: x[1], reverse=True)).keys())
lst2 =nodes_sorted_on_degree_neg[:20]
print(lst2)

print(set(lst1).intersection(lst2))


# clusteringCoeffs_pos = nx.clustering(nx_graph_modelled_pos, weight='weight')
# nodes_sorted_on_clusteringCoeffs_pos = list(dict(sorted(clusteringCoeffs_pos.items(), key=lambda x: x[1])).keys())
# PositiveClusteringCoefs= open("PositiveClusteringCoefs.csv", 'w')
# PositiveClusteringCoefs.write(','.join(nodes_sorted_on_clusteringCoeffs_pos))
# PositiveClusteringCoefs.close()
# print("PositiveClusteringCoefs")
#
# clusteringCoeffs_neg = nx.clustering(nx_graph_modelled_neg, weight='weight')
# nodes_sorted_on_clusteringCoeffs_neg = list(dict(sorted(clusteringCoeffs_neg.items(), key=lambda x: x[1])).keys())
# NegativeClusteringCoefs= open("NegativeClusteringCoefs.csv", 'w')
# NegativeClusteringCoefs.write(','.join(nodes_sorted_on_clusteringCoeffs_neg))
# NegativeClusteringCoefs.close()
# print("NegativeClusteringCoefs")
#
# print("degree")
# g_distance_dict = {(e1, e2): 1 / weight for e1, e2, weight in nx_graph_modelled_pos.edges(data='weight')}
# nx.set_edge_attributes(nx_graph_modelled_pos, g_distance_dict, 'distance')
# closenessCentralities_pos = nx.closeness_centrality(nx_graph_modelled_pos, distance='distance')
# nodes_sorted_on_closenessCentralities_pos = list(dict(sorted(closenessCentralities_pos.items(), key=lambda x: x[1])).keys())
# PositiveClosenessCentralities = open("PositiveClosenessCentralities.csv", 'w')
# PositiveClosenessCentralities.write(','.join(nodes_sorted_on_closenessCentralities_pos))
# PositiveClosenessCentralities.close()
# print("PositiveClosenessCentralities")
#
# g_distance_dict = {(e1, e2): 1 / weight for e1, e2, weight in nx_graph_modelled_neg.edges(data='weight')}
# nx.set_edge_attributes(nx_graph_modelled_neg, g_distance_dict, 'distance')
# closenessCentralities_neg = nx.closeness_centrality(nx_graph_modelled_neg, distance='distance')
# nodes_sorted_on_closenessCentralities_neg = list(dict(sorted(closenessCentralities_neg.items(), key=lambda x: x[1])).keys())
# NegativeClosenessCentralities = open("NegativeClosenessCentralities.csv", 'w')
# NegativeClosenessCentralities.write(','.join(nodes_sorted_on_closenessCentralities_neg))
# NegativeClosenessCentralities.close()
# print("NegativeClosenessCentralities")
#
# degreeCentralities_pos = degree_centrality(nx_graph_modelled_pos)
# nodes_sorted_on_degreeCentralities_pos = list(dict(sorted(degreeCentralities_pos.items(), key=lambda x: x[1])).keys())
# PositiveDegreeCentralities = open("PositiveDegreeCentralities.csv", 'w')
# PositiveDegreeCentralities.write(','.join(nodes_sorted_on_degreeCentralities_pos))
# PositiveDegreeCentralities.close()
# print("PositiveDegreeCentralities")
#
# degreeCentralities_neg = degree_centrality(nx_graph_modelled_neg)
# nodes_sorted_on_degreeCentralities_neg = list(dict(sorted(degreeCentralities_neg.items(), key=lambda x: x[1])).keys())
# NegativeDegreeCentralities = open("NegativeDegreeCentralities.csv", 'w')
# NegativeDegreeCentralities.write(','.join(nodes_sorted_on_degreeCentralities_neg))
# NegativeDegreeCentralities.close()
# print("NegativeDegreeCentralities")
#
#
# edgeBetweenessCentralities_pos = nx.edge_betweenness_centrality(nx_graph_modelled_pos,weight='weight')
# nodes_sorted_on_edgeBetweenessCentralities_pos = list(dict(sorted(edgeBetweenessCentralities_pos.items(), key=lambda x: x[1])).keys())
# PositiveEdgeBetweenessCentralities = open("PositiveEdgeBetweenessCentralities.csv", 'w')
# for tup in nodes_sorted_on_edgeBetweenessCentralities_pos:
#     print(",".join(list(tup)))
#     PositiveEdgeBetweenessCentralities.write(",".join(list(tup))+ '\n')
# PositiveEdgeBetweenessCentralities.close()
# print("PositiveEdgeBetweenessCentralities")
#
# edgeBetweenessCentralities_neg = nx.edge_betweenness_centrality(nx_graph_modelled_neg,weight='weight')
# nodes_sorted_on_edgeBetweenessCentralities_neg = list(dict(sorted(edgeBetweenessCentralities_neg.items(), key=lambda x: x[1])).keys())
# NegativeEdgeBetweenessCentralities = open("NegativeEdgeBetweenessCentralities.csv", 'w')
# for tup in nodes_sorted_on_edgeBetweenessCentralities_neg:
#     print(",".join(list(tup)))
#     NegativeEdgeBetweenessCentralities.write(",".join(list(tup))+ '\n')
# NegativeEdgeBetweenessCentralities.close()
# print("NegativeEdgeBetweenessCentralities")
#
#
# CurrentFlowCentralities_pos = nx.current_flow_betweenness_centrality(nx_graph_modelled_pos,weight='weight')
# nodes_sorted_on_CurrentFlowCentralities_pos = list(dict(sorted(CurrentFlowCentralities_pos.items(), key=lambda x: x[1])).keys())
# PositiveCurrentFlowCentralities = open("PositiveCurrentFlowCentralities.csv", 'w')
# PositiveCurrentFlowCentralities.write(','.join(nodes_sorted_on_CurrentFlowCentralities_pos))
# PositiveCurrentFlowCentralities.close()
# print("PositiveCurrentFlowCentralities")
#
# CurrentFlowCentralities_neg = nx.current_flow_betweenness_centrality(nx_graph_modelled_neg,weight='weight')
# nodes_sorted_on_CurrentFlowCentralities_neg = list(dict(sorted(CurrentFlowCentralities_neg.items(), key=lambda x: x[1])).keys())
# NegativeCurrentFlowCentralities = open("NegativeCurrentFlowCentralities.csv", 'w')
# NegativeCurrentFlowCentralities.write(','.join(nodes_sorted_on_CurrentFlowCentralities_neg))
# NegativeCurrentFlowCentralities.close()
# print("NegativeCurrentFlowCentralities")
#
#
# CurrentFlowClosenessCentralities_pos = nx.current_flow_closeness_centrality(nx_graph_modelled_pos,weight='weight')
# nodes_sorted_on_CurrentFlowClosenessCentralities_pos = list(dict(sorted(CurrentFlowClosenessCentralities_pos.items(), key=lambda x: x[1])).keys())
# PositiveCurrentFlowClosenessCentralities = open("PositiveCurrentFlowClosenessCentralities.csv", 'w')
# PositiveCurrentFlowClosenessCentralities.write(','.join(nodes_sorted_on_CurrentFlowClosenessCentralities_pos))
# PositiveCurrentFlowClosenessCentralities.close()
# print("PositiveCurrentFlowClosenessCentralities")
#
# CurrentFlowClosenessCentralities_neg = nx.current_flow_closeness_centrality(nx_graph_modelled_neg,weight='weight')
# nodes_sorted_on_CurrentFlowClosenessCentralities_neg = list(dict(sorted(CurrentFlowClosenessCentralities_neg.items(), key=lambda x: x[1])).keys())
# NegativeCurrentFlowClosenessCentralities = open("NegativeCurrentFlowClosenessCentralities.csv", 'w')
# NegativeCurrentFlowClosenessCentralities.write(','.join(nodes_sorted_on_CurrentFlowClosenessCentralities_neg))
# NegativeCurrentFlowClosenessCentralities.close()
# print("NegativeCurrentFlowClosenessCentralities")
#
#
# get2HopDcGivenNX_pos = get2HopDcGivenNX(nx_graph_modelled_pos)
# nodes_sorted_on_get2HopDcGivenNX_pos = list(dict(sorted(get2HopDcGivenNX_pos.items(), key=lambda x: x[1])).keys())
# Positiveget2HopDcGivenNX = open("Positiveget2HopDcGivenNX.csv", 'w')
# Positiveget2HopDcGivenNX.write(','.join(nodes_sorted_on_get2HopDcGivenNX_pos))
# Positiveget2HopDcGivenNX.close()
# print("Positiveget2HopDcGivenNX")
#
# get2HopDcGivenNX_neg = get2HopDcGivenNX(nx_graph_modelled_neg)
# nodes_sorted_on_get2HopDcGivenNX_neg = list(dict(sorted(get2HopDcGivenNX_neg.items(), key=lambda x: x[1])).keys())
# Negativeget2HopDcGivenNX = open("Negativeget2HopDcGivenNX.csv", 'w')
# Negativeget2HopDcGivenNX.write(','.join(nodes_sorted_on_get2HopDcGivenNX_neg))
# Negativeget2HopDcGivenNX.close()
# print("Negativeget2HopDcGivenNX")
#
#
#
#
#
