class Node:
    def __init__(self, id, is_visited):
        self.id = id
        self.is_visited = is_visited


class Edge:
    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight


class My_Graph:
    def __init__(self, list_of_nodes, list_of_edges):
        self.list_of_nodes = list_of_nodes
        self.list_of_edges = list_of_edges