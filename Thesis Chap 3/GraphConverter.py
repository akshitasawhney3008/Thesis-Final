import networkx as nx


class GraphConverter:
    @staticmethod
    def convert_my_graph_to_nx_graph(list_of_nodes, list_of_edges):
        G = nx.Graph()
        for n in list_of_nodes:
            G.add_node(n.id)
        for e in list_of_edges:
            G.add_edge(e.source, e.target, weight = float(e.weight))
        return G

class SubGraphConverter:
    @staticmethod
    def convert_my_subgraph_to_nx_graph(list_of_nodes, list_of_edges):
        G = nx.DiGraph()
        for n in list_of_nodes:
            G.add_node(n.id)
        for e in list_of_edges:
            G.add_edge(e.source, e.target, weight = float(e.weight))
        return G