import torch
from torch_geometric.data import InMemoryDataset, download_url
import numpy as np
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#import tensorflow as tf
import torch
from torch_geometric.data import Data, DataLoader
import dgmc
from urllib.parse import urlparse
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
#from functools import cached_property

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_dict(table, col):
    table_dict = {}
    for ix, value in enumerate(table[col]):
        #print(value)
        if value != value:
            continue
        if value in table_dict:
            table_dict[value].append(ix)
        else:
            table_dict[value] = [ix]
    return table_dict


def connect_edges(df, column):
    edges = []
    lookup_dict = create_dict(df, column)
    for ix, value in enumerate(df[column]):
        if value in lookup_dict:
            for item in lookup_dict[value]:
                edges.append((ix, item, column, value))
    edges = pd.DataFrame(
        edges, 
        columns=['source', 'target', 'column', 'value']
    )
    return edges


def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()


class ZetaData():
    def __init__(self, file_path, column, target, feature_cols=None, parse_url=True, expand_x=None):
        #super(MyOwnDataset, self).__init__()
        self.column = column
        self.target = target
        self.feature_cols = feature_cols
        self.expand_x = expand_x
        
        self.df = pd.read_csv(file_path, low_memory=False)
        self.df.columns = [i.split(".")[1] for i in self.df.columns]
        
        # parse URLs
        if parse_url == True and column in ["url", "referrer"]:
            self.df[column] = self.df[column].apply(
                lambda x:urlparse(x).netloc if pd.notnull(x) else x
            )

    @property
    def x(self):
        feature_enc = OneHotEncoder(handle_unknown="ignore")
        features = pd.DataFrame(
            feature_enc.fit_transform(self.df[self.feature_cols]).toarray(), 
            columns=feature_enc.get_feature_names(self.feature_cols)
        )
        
        if self.expand_x is not None:
            new_cols = [f"fake_{self.expand_x-i}" for i in range(self.expand_x - features.shape[1])][::-1]
            for column in new_cols:
                features[column] = 0
            
        return torch.tensor(features.values, dtype=torch.float).to(device)

    @property
    def edge_index(self):
        edges = connect_edges(self.df, self.column)
        return torch.tensor(
            edges[['source','target']].T.values, dtype=torch.long
        ).to(device)

    @property
    def y(self):
        label_enc = LabelEncoder()
        labels = label_enc.fit_transform(self.df[self.target])
        return torch.tensor(labels, dtype=torch.long).to(device)
    
    @property
    def train_mask(self):
        return df[target].isna()

    @property
    def num_features(self):
        return self.x.shape[1]

    @property
    def num_classes(self):
        #return len(np.unique(self.y))
        return 10
        
    @property
    def node_count(self):
        return self.x.shape[0]
    
    @property
    def edge_count(self):
        return self.edge_index.shape[1]
    
    @property
    def in_channels(self):
        return self.x.shape[1]
    
    @property
    def out_channels(self):
        return len(np.unique(self.y))


class GCN(pl.LightningModule):
    def __init__(self, dataset):
        super(GCN, self).__init__()
        torch.manual_seed(0)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)
        
        self.in_channels = dataset.num_features
        self.out_channels = dataset.num_classes

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h


def train(data):
    model.train()
    optimizer.zero_grad()  # Clear gradients
    out, h = model(d.x, d.edge_index)  # Perform a single forward pass
    loss = criterion(out[d.train_mask], d.y[d.train_mask]) # calc loss on training nodes
    loss.backward()  # Derive gradients
    optimizer.step()  # Update parameters based on gradients
    return loss, h

def test(d, model):
    model.eval()
    out, h = model(d.x, d.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[~d.train_mask] == d.y[~d.train_mask]
    test_acc = int(test_correct.sum()) / int(~d.train_mask.sum())
    print(f"Correct predictions: {sum(test_correct)}")
    return test_acc