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

def instance_reader(filename: str):
    linear = {}
    quadratic = {}

    n = -1
    with open(filename, encoding="utf8") as infile:
        header = True
        m = -1
        bias_count = 0
        weigth_count = 0
        for line in infile:
            v = map(lambda e: float(e), line.split())
            if header:
                n, m = v
                header = False
            elif bias_count < n:
                var, bias = v
                linear[int(var)] = bias
                bias_count += 1
            else:
                var1, var2, weight = v
                quadratic[(int(var1), int(var2))] = weight
                weigth_count += 1
        assert m == weigth_count

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')

    return bqm