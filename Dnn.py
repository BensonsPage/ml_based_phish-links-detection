"""
File: Dnn.py
Author: Benson Kimani - https://www.linkedin.com/in/benson-kimani-infotech/
Date: 2024-06-15

Description: Script to train a neural network model, machine learning model.
"""

import torch.nn as nn
# import torch

# Neural Network parameters (MLP, DNN)
hiddenLayer1Size=48
hiddenLayer2Size=int(hiddenLayer1Size/2)
p = 0.1
features = 8
n_output = 1


# # MLP
class Dnn(nn.Module):

    def __init__(self):
        super(Dnn, self).__init__()
        self.layer_1 = nn.Linear(features, hiddenLayer1Size)
        self.layer_2 = nn.Linear(hiddenLayer1Size, hiddenLayer2Size)
        self.layer_out = nn.Linear(hiddenLayer2Size, n_output)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p)
        self.batchnorm1 = nn.BatchNorm1d(hiddenLayer1Size)
        self.batchnorm2 = nn.BatchNorm1d(hiddenLayer2Size)

    def forward(self, inputs):
        t = self.relu(self.layer_1(inputs))
        t = self.batchnorm1(t)
        t = self.relu(self.layer_2(t))
        t = self.batchnorm2(t)
        t = self.dropout(t)
        t = self.sigmoid(self.layer_out(t))
        
        return t

# Dnn = Dnn()
# print(Dnn)
