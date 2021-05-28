import torch
import numpy as np

class PCDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, pcs, labels, num_pcs):
        'Initialization'
        self.labels = labels
        self.pcs = pcs
        self.num_pcs = num_pcs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        permute_ids = np.arange(len(self.pcs[index]))
        np.random.shuffle(permute_ids)
        point_ids = permute_ids[:self.num_pcs]
        return self.pcs[index][point_ids], self.labels[index][point_ids]
