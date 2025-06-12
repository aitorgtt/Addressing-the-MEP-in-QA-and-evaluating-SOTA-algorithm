import random
import dimod
import networkx as nx


def instance_generator(n, p, save=True):
    """ 
        ER graph = Erdös Rényi graph
        n: size
        p: density
    """

    G = nx.erdos_renyi_graph(n, p)

    if save:
        path = (f'instances/ER_{n}_{p}.txt')
        f = open(path, "w")
        edge_number = len(G.edges)
        f.write(f"{n} {edge_number}\n")

    linear = {}
    for i in G.nodes:
        val = 2 * random.random() - 1
        linear[i] = val
        if save:
            f.write(f"{i} {val}\n")

    quadratic = {}
    for v in G.edges:
        val = 2 * random.random() - 1
        quadratic[v] = val
        if save:
            f.write(f"{v[0]} {v[1]} {val}\n")
    if save:
        f.close()

    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, 'ISING')
