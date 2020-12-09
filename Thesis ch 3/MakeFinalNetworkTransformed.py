from read_dict_to_make_loe_lon import get_my_graph
from GraphConverter import GraphConverter
import networkx as nx
import matplotlib.pyplot as plt


def get_the_whole_graph(filename):
    my_graph_obj_dict = get_my_graph(filename)
    nx_graph_modelled = GraphConverter().convert_my_graph_to_nx_graph(my_graph_obj_dict.list_of_nodes, my_graph_obj_dict.list_of_edges)
    return nx_graph_modelled

K = 55
path_d = "C://Users//Arushi//PycharmProjects//ThesisChap3//"

genenamesFile = open(path_d + "transformedColumnNames221.txt",'r').readline().rstrip('\n').split(',')

nx_graph_modelled_pos = get_the_whole_graph('Pos221_edgetable.csv')
nx_graph_modelled_neg = get_the_whole_graph('Neg221_edgetable.csv')
nx_graph_modelled_neg = max(nx.connected_component_subgraphs(nx_graph_modelled_neg), key=len)

print(len(nx_graph_modelled_pos))
print(len(nx_graph_modelled_neg))

selectedFeaturesfile = open(path_d+'SelectedFeatures.csv').readlines()
flag = 0
list_of_gene_numbers = []
for line in selectedFeaturesfile:
        list_of_gene_names = line.rstrip('\n').split(',')
        if len(list_of_gene_names) == K:
            break

new_list_of_genenames = []
for gene in genenamesFile:
    if gene not in list_of_gene_names:
        new_list_of_genenames.append(gene)

nx_graph_modelled_pos.remove_nodes_from(new_list_of_genenames)
nx_graph_modelled_neg.remove_nodes_from(new_list_of_genenames)

print(len(nx_graph_modelled_neg))
print(len(nx_graph_modelled_pos))

nx_graph_modelled_neg_layout = nx.spring_layout(nx_graph_modelled_neg, k=0.8)
nx.draw_networkx_nodes(nx_graph_modelled_neg, nx_graph_modelled_neg_layout, node_size=1000)
nx.draw_networkx_edges(nx_graph_modelled_neg, nx_graph_modelled_neg_layout,
                       width=5,arrowsize=10.0)
nx.draw_networkx_labels(nx_graph_modelled_neg, nx_graph_modelled_neg_layout, font_size=8, font_family='sans-serif')

plt.show()

nx_graph_modelled_pos_layout = nx.spring_layout(nx_graph_modelled_pos, k=0.8)
nx.draw_networkx_nodes(nx_graph_modelled_pos, nx_graph_modelled_pos_layout, node_size=1000)
nx.draw_networkx_edges(nx_graph_modelled_pos, nx_graph_modelled_pos_layout,
                       width=5,arrowsize=10.0 )
nx.draw_networkx_labels(nx_graph_modelled_pos, nx_graph_modelled_pos_layout, font_size=8, font_family='sans-serif')




# nx.draw(nx_graph_modelled_pos, pos=nx_agraph.graphviz_layout(nx_graph_modelled_pos), node_size=1200, node_color='lightblue',
#     linewidths=0.25, font_size=10, font_weight='bold', with_labels=True, dpi=1000)
plt.show()



