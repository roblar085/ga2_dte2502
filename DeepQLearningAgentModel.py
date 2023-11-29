import os
from pathlib import Path

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from pandas.io import json
from torch import Tensor


class DeepQLearningAgentModel(nn.Module):
    def __init__(self, board_size, n_frames, n_actions, m):
        super(DeepQLearningAgentModel, self).__init__()

        self.board_size = board_size
        self.n_frames   = n_frames
        self.action_values = nn.Sequential(

        )


        in_channels = n_frames

        pre_kernel = 0
        self.n_actions = n_actions

        for layer in m['model']:
            params = m['model'][layer]
            print(layer)
            if (layer.startswith("Conv2D")):
                kernel_size    = params['kernel_size']
                activation_fun = params['activation']
                filters        = params['filters']
                padding = None
                if 'padding' in params:
                    padding = params['padding']

                if padding is  None:
                    self.action_values.append(
                        nn.Conv2d(in_channels, filters, kernel_size=(kernel_size[0], kernel_size[1]), stride=1))

                    pre_kernel = kernel_size[0]
                else:

                    self.action_values.append(nn.Conv2d(in_channels, filters, kernel_size=(kernel_size[0], kernel_size[1]), stride=1,padding=padding))
                    in_channels = in_channels - filters
                    pre_kernel = kernel_size[0]
                if(activation_fun=='relu'):

                    self.action_values.append(nn.ReLU())
                elif(activation_fun=='sigmoid'):

                    self.action_values.append(nn.Sigmoid())
                if(padding=="same"):

                    in_channels =filters
                else:
                    in_channels = filters
            if ('Flatten' in layer):
                self.action_values.append(nn.Flatten())
                print(in_channels)
                in_channels = in_channels*in_channels*pre_kernel*pre_kernel//8


            if (layer.startswith("Dense")):
                in_channels = in_channels
                activation_fun = params['activation']
                units = params['units']
                self.action_values.append( nn.Linear(in_channels,units))
                if (activation_fun == 'relu'):

                    self.action_values.append(nn.ReLU())
                elif (activation_fun == 'sigmoid'):

                    self.action_values.append(nn.Sigmoid())
                in_channels = units


        self.action_values.append(nn.Linear(in_channels, n_actions))


    def get_weights(self):
        weights = []
        for layer in self.action_values.children():
            if 'weight' in layer.state_dict():
                weights.append(layer.state_dict()['weight'])
            if 'bias' in layer.state_dict():
                weights.append(layer.state_dict()['bias'])

        return weights

    def load_weights(self, file):
        state_dict = {}
        with h5py.File(file, 'r') as hdf5_file:
                for key in hdf5_file.keys():

                    state_dict[key] = torch.from_numpy(hdf5_file[key][:])


        self.load_state_dict(state_dict)

    def save_weights(self,file):
        file_path = Path(file)
        if(not file_path.exists()):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            Path(file).touch()

        with h5py.File(file, 'w') as hdf5_file:

            for name, param in self.named_parameters():

                hdf5_file.create_dataset(name, data=Tensor.cpu(param.data).numpy())

    def set_weights(self,weights):
        weight_pos = 0
        for idx,layer in enumerate(self.action_values.children()):
            if 'weight' in layer.state_dict():
                layer.state_dict()['weight'] = weights[weight_pos]
                weight_pos = weight_pos + 1
            if 'bias' in layer.state_dict():
                layer.state_dict()['bias'] = weights[weight_pos]
                weight_pos = weight_pos + 1


    def to_device(self, device):
        # Move the entire model to the specified device
        self.to(device)

    def forward(self, x):
        #last channel -> first channel
        #Model is made in last channel format!
        if torch.cuda.is_available():
            device = torch.device("cuda")

        else:
            device = torch.device("cpu")
        x = x.to(device)
        self.to_device(device)
        x = x.permute(0, 3, 1, 2)
        layers = self.action_values(x)
        return layers