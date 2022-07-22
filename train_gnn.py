#2qrtA03 and 4ue3L00 removed

from requests import get
import seaborn as sn
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from pathlib import Path
import pickle
import math
import torch
from dgl.nn import GraphConv, GATConv, SAGEConv, SGConv, TAGConv
from dgl.nn.pytorch.glob import SumPooling
from torch.nn import ModuleList
from torch.nn.functional import one_hot
import glob
import dgl
from train_prottucker import DataSplitter
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import train_test_split
import wandb
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
wandb.init(project="eat_gnn")
print(device)

class GNN(nn.Module):
    def __init__(self, n_layer = 2, feature_len = 8, dim = 1024):
        super(GNN, self).__init__()
        self.n_layer = n_layer
        self.feature_len = feature_len
        self.dim = dim
        self.gnn_layers = ModuleList([])
        for i in range(n_layer):
            hops = 2
            self.gnn_layers.append(TAGConv(in_feats=feature_len if i == 0 else dim,
                                           out_feats=dim,
                                           activation=None if i == n_layer - 1 else torch.relu,
                                           k=hops))
        self.pooling_layer = dgl.nn.MaxPooling()
        self.factor = None
        self.output = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 3),
            nn.LogSoftmax(dim= 1),
        )

    def single_pass(self, graph):
        h = graph.ndata['feature']
        for layer in self.gnn_layers:
            h = layer(graph, h)
        if self.factor is None:
            self.factor = math.sqrt(self.dim) / float(torch.mean(torch.linalg.norm(h, dim=1)))
        h *= self.factor
        graph_embedding = self.pooling_layer(graph, h)
        return graph_embedding
    
    def forward(self, X):
        emb = self.single_pass(X)
        emb = self.output(emb)
        return emb

def parse_label_mapping_cath(data_dir):
        id2label = dict()
        label2id = dict()
        cath_label_path = data_dir / 'cath-domain-list.txt'
        with open(cath_label_path, 'r') as f:
            for n_domains, line in enumerate(f):
                # skip header lines
                if line.startswith("#"):
                    continue
                data = line.split()
                identifier = data[0]
                # skip annotations of proteins without embedding (not part of data set)
                cath_class = int(data[1])
                cath_arch = int(data[2])
                cath_topo = int(data[3])
                cath_homo = int(data[4])

                if cath_class not in label2id:
                    label2id[cath_class] = dict()
                if cath_arch not in label2id[cath_class]:
                    label2id[cath_class][cath_arch] = dict()
                if cath_topo not in label2id[cath_class][cath_arch]:
                    label2id[cath_class][cath_arch][cath_topo] = dict()
                if cath_homo not in label2id[cath_class][cath_arch][cath_topo]:
                    label2id[cath_class][cath_arch][cath_topo][cath_homo] = list()

                id2label[identifier] = [cath_class,
                                        cath_arch, cath_topo, cath_homo]
                label2id[cath_class][cath_arch][cath_topo][cath_homo].append(
                    identifier)

        print('Finished parsing n_domains: {}'.format(n_domains))
        print("Total length of id2label: {}".format(len(id2label)))
        return id2label, label2id

def get_graphs(data_dir):
    graph_dict = dict()
    for graph_file in glob.glob(str(data_dir / "graphs" / "*.bin")):
        if str(Path(graph_file).stem) not in ["train_graphs", "val_graphs"]:
            graph_dict[Path(graph_file).stem] = dgl.load_graphs(graph_file)[0][0]
    print("Total Graphs Loaded: {}".format(len(list(graph_dict.keys()))))
    return graph_dict

def accuracy(predictions, labels):
    acc_c = np.mean(np.argmax(predictions, axis=1) == labels)
    print(f"Validation Accuracy C:{round(acc_c, 5)}")
    wandb.log({"Validation Accuracy C": round(acc_c, 5)})
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=np.argmax(predictions, axis=1), preds=labels,)})

def evaluate(model, data_loader):
    model.eval()
    with torch.no_grad():
        # calculate embeddings of all products as the candidate pool
        batch_predictions = []
        batch_true_labels = []
        for batch_graph, labels in data_loader:
            predictions = model(batch_graph)
            batch_predictions.extend(predictions.cpu().numpy())
            batch_true_labels.extend(labels.cpu().numpy())
        accuracy(np.array(batch_predictions), np.array(batch_true_labels))
        # calculate metrics

loss_fn_c = torch.nn.CrossEntropyLoss()
loss_fn_a = torch.nn.CrossEntropyLoss()
loss_fn_t = torch.nn.CrossEntropyLoss()
loss_fn_h = torch.nn.CrossEntropyLoss()

def Custom_cross_entropy(predictions, labels):
    loss_c = loss_fn_c(predictions[:, 0,:], labels)
    loss_a = loss_fn_a(predictions[:, 1,:], labels)
    loss_t = loss_fn_t(predictions[:, 2,:], labels)
    loss_h = loss_fn_h(predictions[:, 3,:], labels)
    return loss_c + loss_a + loss_t + loss_h

root = Path.cwd()
data_dir = root / 'data' / 'ProtTucker' # create a directory for logging your experiments
log_dir = root / 'log' / 'your_log_directory'
id2label, label2id = parse_label_mapping_cath(data_dir)
graph_dict = get_graphs(data_dir)

batch_size = 124
epochs = 100
lr = 1e-4

X = []
y = []
for graph in graph_dict.keys():
    if id2label[graph][0] - 1 != 3:
        X.append(graph_dict[graph])
        y.append(id2label[graph][0] - 1)

print("Labels", np.unique(y, return_counts=True))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
train_data = [(graph.to(device), label) for graph, label in zip(X_train, y_train)]
val_data = [(graph.to(device), label) for graph, label in zip(X_val, y_val)]

train_dataloader = GraphDataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
val_dataloader = GraphDataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)

model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad = True)
loss_fn = torch.nn.CrossEntropyLoss()

wandb.watch(model, log_freq=100)

for i in range(epochs):
        # train
        model.train()
        loss_batch = 0
        predictions_batch = []
        labels_batch = []
        for iter, (batch_graph, labels) in enumerate(train_dataloader):
            batch_graph = batch_graph.to(device)
            labels = labels.to(device)
            predictions = model(batch_graph)
            loss = loss_fn(predictions, labels)
            loss_batch += loss.detach().item()
            predictions_batch.extend(predictions.cpu().detach().numpy())
            labels_batch.extend(labels.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: ", i+1)
        wandb.log({"epoch: ": i+1})
        print("Training Loss:", round(loss_batch, 5))
        wandb.log({'Training Loss': loss_batch})
        print("Training Accuracy:", np.mean(np.argmax(np.array(predictions_batch), axis=1) == np.array(labels_batch)))
        wandb.log({'Training Accuracy': np.mean(np.argmax(np.array(predictions_batch), axis=1) == np.array(labels_batch))})
        # evaluate on the validation set
        val_mrr = evaluate(model, val_dataloader)
        print()

print()
model.eval()
graph_embeddings = dict()
for graph in graph_dict.keys():
    if id2label[graph][0] - 1 != 3:
        graph_dict[graph] = model.single_pass(graph_dict[graph].to(device)).cpu().detach().numpy()
    else:
        del graph_dict[graph]

print('Total Graphs Saved:', len(graph_dict.keys()))
with open(data_dir / 'graph_embeddings.pickle', 'wb') as handle:
    pickle.dump(graph_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)