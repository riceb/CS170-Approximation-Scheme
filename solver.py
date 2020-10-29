import networkx as nx
from parse import read_input_file, write_output_file
from utils import *
import os
import sys
import PriorityQueue

def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    # degree stores the node with the highest degree on its second index
    degree = [0, 0]
    tree_list = []
    for node in G.nodes():
        # when a node can reach all the other nodes, simply return the node
        if len(list(G.neighbors(node))) == len(G.nodes())-1:
            return G.subgraph(node)
        # perform tree operation on SPT sourced from the node
        tree = spt_alt(G, node)
        T = tree_operation(G, tree)
        tree_list.append(T) 
    M = nx.minimum_spanning_tree(G)
    mst = tree_operation(G, M)
    tree_list.append(mst)
    # find argmin between trimed MST and trimed SPT for shortest average pairwise distance
    return min(tree_list, key = lambda x: average_pairwise_distance_fast(x))

def shortest_path_tree(G, node):
    """
    Find the shortest path tree from the given source node. This uses the
    dijkstra built in networkx.
    Does not work due to some bugs and long runtime :(((
    """
    tree = nx.Graph()
    edges = []
    tree.add_nodes_from(G.nodes())
    paths = nx.single_source_dijkstra_path(G, node)
    for k in paths:
        i = 0
        path = paths[k]
        while i + 1 < len(path):
            u = path[i]
            v = path[i+1]
            if (u,v) not in edges or (v,u) not in edges:
                tree.add_edge(u, u, weight = G[u][v]["weight"])
    assert nx.is_tree(tree)
    return tree

def spt_alt(G, node):
    """
    Alternative method to find the shortest path from a given source nodeself.
    Uses the idea of a priority queue implemented through two concurrent
    dictionaries.
    """
    tree = nx.Graph()
    tree.add_nodes_from(G.nodes())
    # dict_pop stores shortest distance seen so far keyed by nodes
    dict_pop = {}
    # dict_store stores the potential path (edges) seen so far keyed by nodes
    dict_store = {}
    for n in G.nodes():
        dict_pop[n] = float('inf')
        dict_store[n] = []
    dict_pop[node] = 0
    dict_store[node] = []
    # popping and updating paths and distances according to Dijkstra
    while dict_pop:
        vertex = min(dict_pop.keys(), key = lambda x: dict_pop[x])
        for v in G.neighbors(vertex):
            if v in dict_pop.keys():
                newDis = dict_pop[vertex] + G[vertex][v]["weight"]
                if dict_pop[v] > newDis:
                    dict_pop[v] = newDis
                    newPath = dict_store[vertex].copy()
                    newPath.append((v,vertex, G[vertex][v]["weight"]))
                    dict_store[v] = newPath
        dict_pop.pop(vertex)
    # add the stored edges to tree
    for n in dict_store:
        edges = dict_store[n]
        for edge in edges:
            """
            Don't need to worry about repeated edges since nx.Graph() handles it
            and only allows one edge between two vertices. This method might waste
            runtime since one edge might be added multiple times but it's polynomial
            anyways.
            """
            tree.add_edge(edge[0], edge[1], weight = edge[2])
    assert nx.is_tree(tree)
    return tree


def tree_operation(G, M):
    """
    Takes in a tree and compares the optimal way to trim or untrim the leaf nodes
    for a lower average pairwise distance. Returns the trimmed tree.
    """
    nodes = list(M.nodes())
    if len(nodes) == 1:
        return M
    elif len(nodes) == 2:
        return M.subgraph(nodes[0])
    # stores the leaf node and its edge weight
    N = M.copy()
    dict = {}
    for node in nodes:
        if M.degree(node) == 1:
            a = list(M.edges(node))
            dict[node] = a[0]
    for n in dict:
        N.remove_node(n)
    assert nx.is_dominating_set(G, N.nodes())
    shortest = average_pairwise_distance_fast(N)
    for n in dict:
        N.add_node(n)
        u = dict[n][0]
        v = dict[n][1]
        N.add_edge(u, v, weight = M[v][u]["weight"])
        newDist = average_pairwise_distance_fast(N)
        if newDist < shortest:
            shortest = newDist
        else:
            N.remove_node(n)
    return M.subgraph(N.nodes())

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in
if __name__ == "__main__":
    output_dir = "outputs"
    input_dir = "inputs"
    for input_path in os.listdir(input_dir):
        graph_name = input_path.split(".")[0]
        G = read_input_file(f"{input_dir}/{input_path}")
        T = solve(G)
        assert is_valid_network(G, T)
        print("Average  pairwise distance: {}".format(average_pairwise_distance_fast(T)))
        write_output_file(T, f"{output_dir}/{graph_name}.out")
# if __name__ == '__main__':
#     assert len(sys.argv) ==  2
#     path = sys.argv[1]
#     G = read_input_file(path)
#     T = solve(G)
#     assert is_valid_network(G, T)
#     print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
#     write_output_file(T, 'out/test.out')
