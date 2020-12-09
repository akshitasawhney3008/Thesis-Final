from My_Graph import My_Graph, Node, Edge
import pickle

list_of_nodes = []


def get_my_graph(fname):
    file2 = open('transformedColumnNames221.txt', 'r')
    list_of_nodes_as_string = []
    list_of_edges = []
    flag = 0
    file1 = open(fname, 'r')
    file1_read = file1.readlines()
    list_of_available_genes = file2.readline().rstrip(',\n').split(',')

    idx = 0
    for lines in file1_read:
        idx = idx + 1
        print(idx)
        if flag == 0:
            flag = 1
        else:
            line_split = lines.rstrip('\n').split(',')
            source = list_of_available_genes[int(line_split[4].lstrip('"').split(' (Aracne) ')[0].split(" ")[1])-1]
            target = list_of_available_genes[int(line_split[4].rstrip('"').split(' (Aracne) ')[1].split(" ")[1])-1]
            weight = float(line_split[1].lstrip('"').rstrip('"'))
            e = Edge(source,target,weight=weight)
            list_of_nodes_as_string.append(source)
            list_of_nodes_as_string.append(target)
            list_of_edges.append(e)


    list_of_nodes_as_string = list(set(list_of_nodes_as_string))
    for n in list_of_nodes_as_string:
        list_of_nodes.append(Node(n, 0))

    g = My_Graph(list_of_nodes, list_of_edges)
    return g