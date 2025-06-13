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
from dwave.embedding.chain_breaks import majority_vote, broken_chains
from dwave.embedding.chain_strength import uniform_torque_compensation
import dimod


def unembed_sampleset_mine(sampleset, embedding, bqm, chain_break_fraction=False):
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
    
    if chain_break_fraction:
        broken = broken_chains(sampleset, chains)
        if broken.size:
            vectors['chain_break_fraction'] = broken.mean(axis=1)[idxs]
        else:
            vectors['chain_break_fraction'] = 0

    info = sampleset.info.copy()

    return dimod.SampleSet.from_samples_bqm((unembedded, variables),
                                            bqm,
                                            info=info,
                                            **vectors)




def just_solve(bqm, n, p, problem_index, embedding_index, prev_output):

    embedding, sampleset, acl, num_qubits, energies, unembedded_energies, reference_energy = prev_output

    sampler = DWaveSampler(solver='Advantage_system4.1')
    target_adjacency = sampler.adjacency
    
    if embedding:
        broken_chain_fractions = np.zeros((2, 1000))
        unembedded_record = unembed_sampleset_mine(sampleset, embedding, bqm, chain_break_fraction=True).record
        read = 0
        for rec_sample in unembedded_record:
            occ = rec_sample[2]
            while occ > 0:
                broken_chain_fractions[0, read] = rec_sample[-1]
                occ -= 1
                read += 1


        embedded_bqm = embed_bqm(bqm, embedding, target_adjacency, chain_strength=lambda bqm: uniform_torque_compensation(bqm = bqm, prefactor=2))
        new_sampleset = sampler.sample(embedded_bqm, num_reads=1000, label=f'BIKAINTEK_EMBEDDING_ER_{n}_{p}_{problem_index}_{embedding_index}_2')
        unembedded_sampleset = unembed_sampleset_mine(new_sampleset, embedding, bqm, chain_break_fraction=True)

        length = 0
        for chain in embedding.values():
            length += len(chain)
        acl = length / n
        num_qubits = length

        
        record = sampleset.record
        read = 0
        new_energies = np.zeros((2, 1000))
        new_energies[0] = energies
        for rec_sample in record:
            occ = rec_sample[2]
            while occ > 0:
                new_energies[1, read] = rec_sample[1]
                occ -= 1
                read += 1
        
        unembedded_record = unembedded_sampleset.record
        read = 0
        new_unembedded_energies = np.zeros((2, 1000))
        new_unembedded_energies[0] = unembedded_energies
        for rec_sample in unembedded_record:
            occ = rec_sample[2]
            while occ > 0:
                new_unembedded_energies[1, read] = rec_sample[1]
                broken_chain_fractions[1, read] = rec_sample[-1]
                occ -= 1
                read += 1
        
        return embedding, (sampleset, new_sampleset), acl, num_qubits, new_energies, new_unembedded_energies, broken_chain_fractions, reference_energy
    else:
        return None, None, None, None, None, None, None, reference_energy

densities = [round(0.05 + 0.05*i, 2) for i in range(20)]  # 20
sizes = [25+10*i for i in range(16)] # 16
n_problems = 5
n_embeddings = 10
chain_strength_prefactor = [2]


with open('pickles/RQ1.1_saved_data.pkl', 'rb') as f:
    acl_array, num_qubits_array, energies, unembedded_energies, reference_energies, relative_errors, unembedded_relative_errors = pickle.load(f)
    container = pickle.load(f)

# acl = np.zeros((20, 16, 5, 10))
# num_qubits = np.zeros((20, 16, 5, 10))
new_energies = np.zeros((20, 16, 5, 10, 2, 1000))
new_unembedded_energies = np.zeros((20, 16, 5, 10, 2, 1000))
# reference_energies = np.zeros((20, 16, 5))
new_relative_errors = np.zeros((20, 16, 5, 10, 2, 1000))
new_unembedded_relative_errors = np.zeros((20, 16, 5, 10, 2, 1000))
broken_chain_fractions = np.zeros((20, 16, 5, 10, 2, 1000))
new_container = []

for density in range(len(densities)):
    for size in range(len(sizes)):
        # if # unfinished:
        #     continue
        print(f"Working on size {sizes[size]} of the density {densities[density]}")
        for problem in range(n_problems):
            bqm = container[80*density+5*size+problem][0]

            embedding_sample_list = []
            for emb in range(n_embeddings):
                embedding, sampleset, acl, num_qubits, energies, unembedded_energies, reference_energy = container[80*density+5*size+problem][1][emb]
                embedding_sample_list.append(just_solve(bqm, sizes[size], densities[density], problem, emb, (embedding, sampleset, acl, num_qubits, energies, unembedded_energies, reference_energy)))

            new_container.append((bqm, embedding_sample_list))

            reference_energy = reference_energies[density, size, problem]

            for n_embedding in range(n_embeddings):
                embedding, _, acl_current, num_qubits_current, energies_current, unembedded_energies_current, broken_chain_fractions_current, _ = embedding_sample_list[n_embedding]
                
                if embedding:
                    new_energies[density, size, problem, n_embedding, :] = energies_current
                    new_unembedded_energies[density, size, problem, n_embedding, :] = unembedded_energies_current
                    new_relative_errors[density, size, problem, n_embedding, :] = abs((reference_energy - energies_current)/reference_energy)
                    new_unembedded_relative_errors[density, size, problem, n_embedding, :] = abs((reference_energy - unembedded_energies_current)/reference_energy)
                    broken_chain_fractions[density, size, problem, n_embedding, :] = broken_chain_fractions_current

                else:
                    new_energies[density, size, problem, n_embedding, :] = np.nan
                    new_unembedded_energies[density, size, problem, n_embedding, :] = np.nan
                    new_relative_errors[density, size, problem, n_embedding, :] = np.nan
                    new_unembedded_relative_errors[density, size, problem, n_embedding, :] = np.nan
                    broken_chain_fractions[density, size, problem, n_embedding, :] = np.nan


        with open('RQ1.1_extended.pkl', 'wb') as f:
            pickle.dump([acl_array, num_qubits_array, energies, unembedded_energies, reference_energies, relative_errors, unembedded_relative_errors, broken_chain_fractions], f)
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
    problems.to_csv("reports_rerun_RQ1.1/" + str(namefile) + ".csv", index = False)
