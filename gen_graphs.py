from locale import currency
from math import gamma
from pathlib import Path
import numpy as np
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.subgraphs import extract_subgraph_by_sequence_position
from graphein.ml.conversion import GraphFormatConvertor
import re
import dgl
import torch
import pickle
import os
import networkx as nx
import glob

type2form = {
            "atom_type": "str",
            "b_factor": "float",
            "chain_id": "str",
            "coords": "np.array",
            "dist_mat": "np.array",
            "element_symbol": "str",
            "node_id": "str",
            "residue_name": "str",
            "residue_number": "int",
            "edge_index": "torch.tensor",
            "kind": "str",
        }
columns = [
                    "atom_type",
                    "b_factor",
                    "chain_id",
                    "chain_ids",
                    "config",
                    "coords",
                    "dist_mat",
                    "edge_index",
                    "element_symbol",
                    "kind",
                    "name",
                    "node_id",
                    "node_type",
                    "pdb_df",
                    "raw_pdb_df",
                    "residue_name",
                    "residue_number",
                    "rgroup_df",
                    "sequence_A",
                    "sequence_B",
                ]

def convert_nx_to_dgl(G: nx.Graph) -> dgl.DGLGraph:
        """
        Converts ``NetworkX`` graph to ``DGL``
        :param G: ``nx.Graph`` to convert to ``DGLGraph``
        :type G: nx.Graph
        :return: ``DGLGraph`` object version of input ``NetworkX`` graph
        :rtype: dgl.DGLGraph
        """
        g = dgl.DGLGraph()
        node_id = list(G.nodes())
        G = nx.convert_node_labels_to_integers(G)

        ## add node level feat
        node_dict = {}
        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            #print(feat_dict.items())
            for key, value in feat_dict.items():
                if str(key) in columns:
                    node_dict[str(key)] = (
                        [value] if i == 0 else node_dict[str(key)] + [value]
                    )

        string_dict = {}
        node_dict_transformed = {}
        for i, j in node_dict.items():
            if i == "coords":
                node_dict_transformed[i] = torch.Tensor(np.asarray(j)).type(
                    "torch.FloatTensor"
                )
            elif i == "dist_mat":
                node_dict_transformed[i] = torch.Tensor(
                    np.asarray(j[0].values)
                ).type("torch.FloatTensor")
            elif type2form[i] == "str":
                string_dict[i] = j
            elif type2form[i] in ["float", "int"]:
                node_dict_transformed[i] = torch.Tensor(np.array(j))
        g.add_nodes(
            len(node_id),
            node_dict_transformed,
        )

        edge_dict = {}
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

        # add edge level features
        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
            for key, value in feat_dict.items():
                if str(key) in columns:
                    edge_dict[str(key)] = (
                        list(value)
                        if i == 0
                        else edge_dict[str(key)] + list(value)
                    )

        edge_transform_dict = {}
        for i, j in node_dict.items():
            if type2form[i] == "str":
                string_dict[i] = j
            elif type2form[i] in ["float", "int"]:
                edge_transform_dict[i] = torch.Tensor(np.array(j))
        g.add_edges(edge_index[0], edge_index[1], edge_transform_dict)

        # add graph level features
        graph_dict = {
            str(feat_name): [G.graph[feat_name]]
            for feat_name in G.graph
            if str(feat_name) in columns
        }
        node_features = []
        for i in range(len(g.nodes())):
            numerical_feature = []
            numerical_feature.append(G.nodes[i]['b_factor'])
            numerical_feature.extend(G.nodes[i]['meiler'].tolist())
            node_features.append(numerical_feature)
        node_features = torch.tensor(node_features)
        g.ndata['feature'] = node_features
        # transform to bi-directed graph with self-loops
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.add_self_loop(g)
        return g

def read_cath_ids(path):
    pdb_ids = []
    chain = []
    domain = []
    identifiers = []
    with open(path, 'r') as f:
        for n_domains, line in enumerate(f):
            # skip header lines
            if line.startswith("#"):
                continue
            data = line.split()
            identifier = data[0]
            pdb_ids.append(identifier[:4].upper())
            chain.append(identifier[5])
            domain.append(identifier[6:])
            identifiers.append(identifier)
    return pdb_ids, chain, domain, identifiers

root = Path.cwd()
data_dir = root / 'data' / 'ProtTucker'
graph_data_dir = root / 'data' / 'ProtTucker' / 'graphs'
save_path = root / 'data' / 'ProtTucker'
cath_label_path = data_dir / 'cath-domain-list.txt'
cath_seqs_path = data_dir / 'cath-domain-seqs.txt'
train_path =  data_dir / 'train66k.fasta'
val_path =  data_dir / 'val200.fasta'
pdb_ids, chain, domain, identifiers = read_cath_ids(cath_label_path)

dataset = []
train = []
with open(train_path, 'r') as f:
    for n_domains, line in enumerate(f):
        # skip header lines
        if line.startswith(">"):
            id = line[1:-1]
            dataset.append(id)
            train.append(id)

val = []
with open(val_path, 'r') as f:
    for n_domains, line in enumerate(f):
        # skip header lines
        if line.startswith(">"):
            id = line[1:-1]
            dataset.append(id)
            val.append(id)

current_id = None
starting_pos = None
seq_test = dict()
ids = list()
print("starting")
if not os.path.exists(f'{data_dir}/data.pickle'):
    with open(cath_seqs_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                line = line.replace(">", "")
                if '|' in line:
                    seq_id = line.split('|')[2]
                else:
                    seq_id = line
                if seq_id in ids:  # some weird double entries in CATH test set..
                    continue
                id = seq_id.split('/')[0]
                pdb = id[:4]
                domains = seq_id.split('/')[1]
                indices = []
                if id not in dataset:
                    continue
                if id[-2:] == '00':
                    indices = ['00']
                elif id == "1nsaA01":
                    indices = range(1, 93)
                elif id == "1yrbA01":
                    indices = range(1, 245)
                elif id == "3gljA01":
                    indices = range(1, 99)
                elif id == "1ef0B01":
                    indices = list(range(1, 183))
                    indices.extend(range(416, 458))
                elif id in ["2hk2A01", "2hk3A01"]:
                    indices = range(1, 186)
                elif id in ["1pfzA02","3plnA01", "3qvcA01", "4n6hA01", "5jcpA02" ]:
                    indices = ['00']
                else:
                    for ranges in domains.split('_'):
                        ranges = re.sub(r'\s?\(.*?\)', '', ranges)
                        if ranges[0] != '-':
                            indices.extend(range(int(ranges.split('-')[0]), int(ranges.split('-')[1] ) + 1))
                        else:
                            if ranges.count('-') == 2:
                                if pdb == current_id:
                                    indices.extend(range(int(1) + starting_pos, int(ranges.split('-')[1]) + 2 + int(ranges.split('-')[2]) + starting_pos))
                                else:
                                    indices.extend(range(int(1), int(ranges.split('-')[1]) + 2 + int(ranges.split('-')[2])))
                            else:
                                if pdb == current_id:
                                    indices.extend(range(int(1) + starting_pos, int(ranges.split('-')[1]) + 2 - int(ranges.split('-')[3]) + starting_pos))
                                else:
                                    indices.extend(range(int(1), int(ranges.split('-')[1]) + 2 - int(ranges.split('-')[3])))
                        if pdb != current_id:
                            current_id = pdb
                            #print(indices)
                            starting_pos = int(indices[0])
                seq_test[id] = indices
    # some identical sequences need to be removed
    # assert that no identical seqs are in the sets
    pickle.dump(seq_test, open(f'{data_dir}/data.pickle', 'wb'))

seq_test = pickle.load(open(f'{data_dir}/data.pickle', 'rb'))
graphs_done = [x.split('/')[-1][:-4] for x in glob.glob(f'{graph_data_dir}/*.bin')]
train_graphs = []
val_graphs = []
invalid_graphs = []
config = ProteinGraphConfig()
converter = GraphFormatConvertor("nx", "dgl")
for n, id in enumerate(train[:15002]):
    if id not in graphs_done:
        try:
            g = construct_graph(config=config, pdb_code=id[:4])
            print(n, id, len(g.nodes()))
            if seq_test[id] != ["00"]:
                g = extract_subgraph_by_sequence_position(g, seq_test[id])
            dgl.save_graphs(str(graph_data_dir) + '/' + f'{id}.bin', convert_nx_to_dgl(g))
        except:
            invalid_graphs.append(id)

for n, id in enumerate(val):
    if id not in graphs_done:
        try:
            g = construct_graph(config=config, pdb_code=id[:4])
            print(n, id, len(g.nodes()))
            if seq_test[id] != ["00"]:
                g = extract_subgraph_by_sequence_position(g, seq_test[id])
            dgl.save_graphs(str(graph_data_dir) + '/' + f'{id}.bin', convert_nx_to_dgl(g))
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
            invalid_graphs.append(id)
