from minorminer.minorminer import find_embedding
from dwave.system import DWaveSampler
from dwave.embedding import embed_bqm
import dimod
import numpy as np
from utils import er_generator_indexed
import pickle
from joblib import Parallel as Para
from joblib import delayed
import sys
import hybrid
import time
import pandas as pd
from dwave.cloud.api import Problems
import json
from dwave.embedding.chain_breaks import majority_vote
import dimod

def unembed_sampleset_mine(sampleset, embedding, bqm):
    variables = list(bqm.variables)  # need this ordered
    try:
        chains = [embedding[v] for v in variables]
    except KeyError:
        raise ValueError("given bqm does not match the embedding")

    record = sampleset.record

    unembedded, idxs = majority_vote(sampleset, chains)

    reserved = {'sample', 'energy'}
    vectors = {name: record[name][idxs]
            for name in record.dtype.names if name not in reserved}

    info = sampleset.info.copy()

    return dimod.SampleSet.from_samples_bqm((unembedded, variables), bqm, info=info, **vectors)



def find_embedding_and_solve(bqm, n, p, problem_index, embedding_index):

    sampler = DWaveSampler(solver='Advantage_system4.1')
    target_edgelist = sampler.edgelist
    target_adjacency = sampler.adjacency

    workflow = (hybrid.Parallel(hybrid.TabuProblemSampler(), hybrid.SimulatedAnnealingProblemSampler()) | hybrid.ArgMin() )

    initial_state = hybrid.State.from_problem(bqm)
    states_updated = workflow.run(initial_state).result()
    reference_energy = states_updated.samples.first.energy

    sys.stdout = open(f'./outputs_TCP6/output_er_{n}_{p}_{problem_index}_{embedding_index}.txt', 'wt')
    source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
    
    embedding = find_embedding(source_edgelist, target_edgelist, verbose=3, interactive=True, chainlength_patience=embedding_index)
    
    if embedding:
        embedded_bqm = embed_bqm(bqm, embedding, target_adjacency, chain_strength=None)
        sampleset = sampler.sample(embedded_bqm, num_reads=1000, label=f'BIKAINTEK_EMBEDDING_ER_{n}_{p}_{problem_index}_{embedding_index}')
        unembedded_sampleset = unembed_sampleset_mine(sampleset, embedding, bqm)

        length = 0
        for chain in embedding.values():
            length += len(chain)
        acl = length / n
        num_qubits = length

        
        record = sampleset.record
        read = 0
        energies = np.zeros((1000))
        for rec_sample in record:
            occ = rec_sample[2]
            while occ > 0:
                energies[read] = rec_sample[1]
                occ -= 1
                read += 1
        
        unembedded_record = unembedded_sampleset.record
        read = 0
        unembedded_energies = np.zeros((1000))
        for rec_sample in unembedded_record:
            occ = rec_sample[2]
            while occ > 0:
                unembedded_energies[read] = rec_sample[1]
                occ -= 1
                read += 1
        
        return embedding, sampleset, acl, num_qubits, energies, unembedded_energies, reference_energy
    else:
        return None, None, None, None, None, None, reference_energy

densities = [round(0.05 + 0.05*i, 2) for i in range(20)]  # 20
sizes = [25+10*i for i in range(16)] # 16
n_problems = 5
n_embeddings = 10
chain_strength_prefactor = [2]


# with open('unfinished.pkl', 'rb') as f:
#     acl, num_qubits, energies, unembedded_energies, reference_energies, relative_errors, unembedded_relative_errors = pickle.load(f)
#     container = pickle.load(f)

acl = np.zeros((20, 16, 5, 10))
num_qubits = np.zeros((20, 16, 5, 10))
energies = np.zeros((20, 16, 5, 10, 1000))
unembedded_energies = np.zeros((20, 16, 5, 10, 1000))
reference_energies = np.zeros((20, 16, 5))
relative_errors = np.zeros((20, 16, 5, 10, 1000))
unembedded_relative_errors = np.zeros((20, 16, 5, 10, 1000))
container = []

for density in range(len(densities)):
    for size in range(len(sizes)):
        # if # unfinished:
        #     continue
        print(f"Working on size {sizes[size]} of the density {densities[density]}")
        for problem in range(n_problems):
            bqm = er_generator_indexed(sizes[size], densities[density], problem, save=True, generate_weights=True)
            
            embedding_sample_list = Para(n_jobs=n_embeddings)(
                delayed(find_embedding_and_solve)(bqm, sizes[size], densities[density], problem, j)
                for j in range(n_embeddings))

            container.append((bqm, embedding_sample_list))

            reference_energy = min(embedding_sample_list[n_embedding][-1] for n_embedding in range(n_embeddings))
            reference_energies[density, size, problem] = reference_energy

            for n_embedding in range(n_embeddings):
                embedding, _, acl_current, num_qubits_current, energies_current, unembedded_energies_current, _ = embedding_sample_list[n_embedding]
                if embedding:
                    acl[density, size, problem, n_embedding] = acl_current
                    num_qubits[density, size, problem, n_embedding] = num_qubits_current
                    energies[density, size, problem, n_embedding, :] = energies_current
                    unembedded_energies[density, size, problem, n_embedding, :] = unembedded_energies_current
                    relative_errors[density, size, problem, n_embedding, :] = abs((reference_energy - energies_current)/reference_energy)
                    unembedded_relative_errors[density, size, problem, n_embedding, :] = abs((reference_energy - unembedded_energies_current)/reference_energy)

                else:
                    acl[density, size, problem, n_embedding] = np.nan
                    num_qubits[density, size, problem, n_embedding] = np.nan
                    energies[density, size, problem, n_embedding, :] = np.nan
                    unembedded_energies[density, size, problem, n_embedding, :] = np.nan
                    relative_errors[density, size, problem, n_embedding, :] = np.nan
                    unembedded_relative_errors[density, size, problem, n_embedding, :] = np.nan

        with open('RQ1.1.pkl', 'wb') as f:
            pickle.dump([acl, num_qubits, energies, unembedded_energies, reference_energies, relative_errors, unembedded_relative_errors], f)
            pickle.dump(container, f)

    #LO DE ENEKO PARA GUARDAR LAS RUNS
    api = Problems.from_config()
    ps = api.list_problems(max_results=800)
    problems = pd.DataFrame()
    for count, p in enumerate(ps):
        p_json = json.loads(p.json())
    # store results in pandas dataframe
        for i in p_json.keys():
            problems.loc[count, i] = p_json.get(i)
    
    namefile = time.time()
    problems.to_csv("reports_TCP6/" + str(namefile) + ".csv", index = False)
