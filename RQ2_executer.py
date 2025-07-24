from dwave.system import DWaveSampler
import dimod
import time
import numpy as np
from utils import *
import pickle
from joblib import Parallel, delayed
import sys

# Choose a method:
from minorminer import find_embedding
#  from minorminer.layout import find_embedding

token = 'some token'
dw = DWaveSampler(token=token)      # This is not necesary, but if you have a token is an easy way of getting the exact broken Pegasus graph.
target_nodelist, target_edgelist, target_adjacency = dimod.child_structure_dfs(dw)
target_graph = nx.Graph()
target_graph.add_edges_from(target_edgelist)

def find_embedding_plus(source, n, p, idx):
    sys.stdout = open(f'./outputs/output_er_{n}_{p}_{idx}.txt', 'wt')
    try:
        out = find_embedding(source, target_graph, verbose=3, interactive=True,
                             return_overlap=True)
    except:
        print('impossible to initialize')
        return None
    else:
        return out


densities = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]  # 20
sizes = np.arange(10, 301, 5)  # 59

percentages = np.zeros((20, 59))
times = np.zeros((20, 59))
acl = np.zeros((20, 59, 64))
num_qubits = np.zeros((20, 59, 64))

embedding_container = []


for density in range(20):
    for size in range(59):
        print(f"Working on size {sizes[size]} of the density {densities[density]}")
        source_graph = graph_generator(sizes[size], densities[density], save=True)
        try:
            source_graph = graph_reader(f'./instances/instances_RQ2/ER_{sizes[size]}_{densities[density]}.txt')
        except FileNotFoundError:
           break

        t0 = time.time()
        emb_list = Parallel(n_jobs=64)(
            delayed(find_embedding_plus)(source_graph, sizes[size], densities[density], j)
            for j in range(64))
        t1 = time.time()
        times[density, size] = t1 - t0
        
        embedding_container.append([source_graph, emb_list])

        valid = 0
        for k in range(64):
            if emb_list[k][1] == 1:
                luzeera = 0
                for emb in emb_list[k][0].values():
                    luzeera += len(emb)
                acl[density, size, k] = luzeera / sizes[size]
                num_qubits[density, size, k] = luzeera
                valid += 1
            else:
                acl[density, size, k] = np.nan
                num_qubits[density, size, k] = np.nan

        percentages[density, size] = 100 * valid / 64

        with open('RQ2.pkl', 'wb') as f:
            pickle.dump([percentages, times, acl, num_qubits], f)
            pickle.dump(embedding_container, f)

        if valid == 0:
            break

        