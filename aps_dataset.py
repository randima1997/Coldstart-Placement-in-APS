import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


class APSSiameseDataset(Dataset):
    def __init__(self, features, aps_vectors):
        super().__init__()
        self.features = features
        self.aps_vectors = aps_vectors

        self.data = []

        for idx1, f_vec1 in enumerate(self.features):
            for idx2, f_vec2 in enumerate(self.features):
                self.data.append((f_vec1, f_vec2, self.aps_vectors[idx1], self.aps_vectors[idx2]))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):


        return                              # (ft1, ft2, aps1, aps2, sim)