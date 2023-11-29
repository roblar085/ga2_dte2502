from pathlib import Path

import h5py
import torch
import torch.nn as nn
from torch import Tensor

class AdvantageActorCriticAgentModel(nn.Module):
    def __init__(self, board_size, n_frames, n_actions,outputs):
        super(AdvantageActorCriticAgentModel, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(n_frames, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152, 64),
            nn.ReLU()
        )

        self.action_logits =  nn.Linear(64, n_actions)
        self.state_values  =  nn.Linear(64, 1)
        self.outputs = outputs

    def load_weights(self, file):
        state_dict = {}
        with torch.no_grad(), h5py.File(file, 'r') as hdf5_file:
            for name, param in self.named_parameters():

                if name in hdf5_file:
                    state_dict[name] = torch.tensor(hdf5_file[name][:])

        self.load_state_dict(state_dict)

    def load_weights(self, file):
        state_dict = {}
        with torch.no_grad(), h5py.File(file, 'r') as hdf5_file:
            for key in hdf5_file.keys():
                state_dict[key] = torch.from_numpy(hdf5_file[key][:])

        self.load_state_dict(state_dict)

    def save_weights(self, file):
        file_path = Path(file)
        if (not file_path.exists()):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            Path(file).touch()

        with h5py.File(file, 'w') as hdf5_file:

            for name, param in self.named_parameters():
                hdf5_file.create_dataset(name, data=Tensor.cpu(param.data).numpy())

    def to_device(self, device):
        # Move the entire model to the specified device
        self.to(device)

    def set_weights(self, weights):
        weight_pos = 0
        for idx, layer in enumerate(self.model.children()):

            if 'weight' in layer.state_dict():
                layer.state_dict()['weight'] = weights[weight_pos]
                weight_pos = weight_pos + 1
            if 'bias' in layer.state_dict():
                layer.state_dict()['bias'] = weights[weight_pos]
                weight_pos = weight_pos + 1
        for idx, layer in enumerate(self.action_logits.children()):

            if 'weight' in layer.state_dict():
                layer.state_dict()['weight'] = weights[weight_pos]
                weight_pos = weight_pos + 1
            if 'bias' in layer.state_dict():
                layer.state_dict()['bias'] = weights[weight_pos ]
                weight_pos = weight_pos + 1
        for idx, layer in enumerate(self.state_values.children()):

            if 'weight' in layer.state_dict():
                layer.state_dict()['weight'] = weights[weight_pos]
                weight_pos = weight_pos + 1
            if 'bias' in layer.state_dict():
                layer.state_dict()['bias'] = weights[weight_pos ]
                weight_pos = weight_pos + 1

    def get_weights(self):
        weights = []
        for layer in self.model.children():
            if 'weight' in layer.state_dict():
                weights.append(layer.state_dict()['weight'])
            if 'bias' in layer.state_dict():
                weights.append(layer.state_dict()['bias'])
        for layer in self.action_logits.children():
            if 'weight' in layer.state_dict():
                weights.append(layer.state_dict()['weight'])
            if 'bias' in layer.state_dict():
                weights.append(layer.state_dict()['bias'])
        for layer in self.state_values.children():
            if 'weight' in layer.state_dict():
                weights.append(layer.state_dict()['weight'])
            if 'bias' in layer.state_dict():
                weights.append(layer.state_dict()['bias'])

        return weights

    def forward(self, x):
        device = torch.device("cuda")
        if torch.cuda.is_available():
            device = torch.device("cuda")

        else:
            device = torch.device("cpu")
        x = x.to(device)
        self.to_device(device)
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        if 'action_logits' in self.outputs and 'state_values':
            action_logits, state_values =  self.action_logits(x),self.state_values(x)
            return action_logits, state_values
        elif 'action_logits' in self.outputs:
            action_logits = self.action_logits(x)
            return action_logits
        elif 'state_values' in self.outputs:
            state_values = self.state_values(x)
            return state_values
        else:
            return self.action_logits(x),self.state_values(x)

