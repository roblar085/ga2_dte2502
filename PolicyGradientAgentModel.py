from pathlib import Path

import h5py
import torch
import torch.nn as nn
from torch import Tensor


class PolicyGradientAgentModel(nn.Module):
    def __init__(self, board_size, n_frames, n_actions,weight_decay):
        super(PolicyGradientAgentModel, self).__init__()

        self.weight_decay = weight_decay
        self.board_size = board_size
        self.n_frames = n_frames

        self.policy = nn.Sequential(
            nn.Conv2d(n_frames, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def to_device(self, device):
        # Move the entire model to the specified device
        self.to(device)

    def load_weights(self, file):
        state_dict = {}
        with h5py.File(file, 'r') as hdf5_file:
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

    def forward(self, x):
        if torch.cuda.is_available():
            device = torch.device("cuda")

        else:
            device = torch.device("cpu")
        x = x.to(device)
        self.to_device(device)
        x = x.permute(0, 3, 1, 2)
        out = self.policy(x)
        return out
