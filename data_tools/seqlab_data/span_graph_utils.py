import networkx as nx


def create_graph(edges):
    G = nx.Graph()
    for edge in edges:
        G.add_edge(*edge)
    return G


def connected_components(G):
    return [c for c in nx.connected_components(G)]
