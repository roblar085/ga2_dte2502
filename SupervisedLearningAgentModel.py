from pathlib import Path

import h5py
import torch
import torch.nn as nn

class SupervisedLearningAgentModule(nn.Module):
    def __init__(self,inputs,outputs):
        super(SupervisedLearningAgentModule, self).__init__()

        self.inputs = inputs
        self.outputs = outputs

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


    def to_device(self, device):
        # Move the entire model to the specified device
        self.to(device)

    def forward(self, x):
        if torch.cuda.is_available():
            device = torch.device("cuda")

        else:
            device = torch.device("cpu")
        x = x.to(device)
        self.to_device(device)
        x = self.inputs(x)
        x = self.outputs(x)
        return x
