# Preamble

import time

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

from tqdm import trange
from tqdm._utils import _term_move_up

from pandas import get_dummies
from sklearn.preprocessing import normalize

# Defining the Graph Convolutional Network
# TODO add in a parser

class Convolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data, a = 0, mode = 'fan_in')

    def forward(self, input, adjacency_matrix):
        weighted_input = torch.mm(input, self.weight)
        return torch.spmm(adjacency_matrix, weighted_input)

class Graph_Convolutional_Network(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super(Graph_Convolutional_Network, self).__init__()
        self.convolution_1 = Convolution(in_features, hidden)
        self.convolution_2 = Convolution(hidden, out_features)

    def forward(self, input, adjacency_matrix):
        input = self.convolution_1(input, adjacency_matrix)
        input = functional.relu(input)
        input = functional.dropout(input, 0.5, training = self.training)
        input = self.convolution_2(input, adjacency_matrix)
        return functional.log_softmax(input, dim = 1)


# Load data (cora citation network)
# TODO turn into object (keep it simple, methods: load_data, construct_network, accuracy, train_network, test_netwwork)
# TODO add in a logger
print('Loading the cora network dataset ...')
index_features_labels = np.genfromtxt("cora.content", dtype = np.dtype(str))
features = normalize(sp.csr_matrix(index_features_labels[:, 1:-1], dtype = np.float32), axis = 1, copy = False)
labels = get_dummies(index_features_labels[:, -1])

print('Building the citation graph ...')
index_map = {j: i for i, j in enumerate(np.array(index_features_labels[:, 0], dtype = np.int32))}
edges_unordered = np.genfromtxt("cora.cites", dtype = np.int32)
edges = np.array(list(map(index_map.get, edges_unordered.flatten())), dtype = np.int32).reshape(edges_unordered.shape)
adjacency_matrix = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape = (labels.shape[0], labels.shape[0]), dtype = np.float32)

print('Calculating the adjacency matrix ...')
adjacency_matrix = adjacency_matrix + adjacency_matrix.T.multiply(adjacency_matrix.T > adjacency_matrix) - adjacency_matrix.multiply(adjacency_matrix.T > adjacency_matrix)
adjacency_matrix = normalize(adjacency_matrix + sp.eye(adjacency_matrix.shape[0]), axis = 1, copy = False)

adjacency_matrix = adjacency_matrix.tocoo().astype(np.float32)
adjacency_matrix_indices = torch.from_numpy(np.vstack((adjacency_matrix.row, adjacency_matrix.col)).astype(np.int64))
adjacency_matrix_values = torch.from_numpy(adjacency_matrix.data)
adjacency_matrix_shape = torch.Size(adjacency_matrix.shape)

adjacency_matrix = torch.sparse.FloatTensor(adjacency_matrix_indices, adjacency_matrix_values, adjacency_matrix_shape)
features, labels = torch.FloatTensor(np.array(features.todense())), torch.LongTensor(np.where(labels)[1])
index_train, index_validation, index_test = torch.LongTensor(range(200)), torch.LongTensor(range(200, 500)), torch.LongTensor(range(500, 1500))

# Model and optimizer

print("Constructing the neural network ...")
model = Graph_Convolutional_Network(in_features = features.shape[1], hidden = 16 , out_features = labels.max().item() + 1)
optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)

# Defining accuracy

def accuracy(output, labels):
    with torch.no_grad():
        predicted = torch.max(output.data, 1)
        correct = labels.eq(predicted[1]).double().sum()
    return correct / len(labels)

# Training and testing the model

total_epochs = 200

print("Training the model ({} epochs) ...".format(total_epochs))
starting_time = time.time()

border = "-"*50
clear_border = _term_move_up() + "\r" + " "*len(border) + "\r"
print(border)
print(border)

progress_bar = trange(total_epochs)
for epoch in progress_bar:
        model.train() #turns dropout on
        optimizer.zero_grad()
        output = model(features, adjacency_matrix)
        loss_train = functional.nll_loss(output[index_train], labels[index_train])
        accuracy_train = accuracy(output[index_train], labels[index_train])

        model.eval() #turns dropout off
        output = model(features, adjacency_matrix)
        loss_validation = functional.nll_loss(output[index_validation], labels[index_validation])
        accuracy_validation = accuracy(output[index_validation], labels[index_validation])

        if epoch%(total_epochs/10) == 0:
            progress_bar.write(clear_border + "epoch:{} loss:{:.9f} accuracy:{:.2f}".format(epoch, loss_validation, accuracy_validation))
            progress_bar.write(border)
        loss_train.backward()
        optimizer.step()

print("Total time elapsed during training: {:.4f}s".format(time.time() - starting_time))

model.eval()
output = model(features, adjacency_matrix)
loss_test= functional.nll_loss(output[index_test], labels[index_test])
accuracy_test = accuracy(output[index_test], labels[index_test])
print("Final test score: loss = {:.4f}".format(loss_test.item()), "accuracy = {:.4f}".format(accuracy_test.item()))