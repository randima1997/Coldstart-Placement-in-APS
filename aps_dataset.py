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
                similarity = F.pairwise_distance(self.aps_vectors[idx1], self.aps_vectors[idx2], p= 2)
                self.data.append((f_vec1, f_vec2, self.aps_vectors[idx1], self.aps_vectors[idx2], similarity))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        return self.data[index]                      # (ft1, ft2, aps1, aps2, sim)
    


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.emb_net = nn.Sequential(
            nn.Linear(in_features= 12, out_features= 200),
            nn.ReLU(),
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Linear(512, 100)
        )

    def forward(self, x):

        x = self.emb_net(x)

        return x


class SiameseNet(nn.Module):
    def __init__(self, emb_net):
        super().__init__()
        self.emb_net = emb_net
    
    def forward(self, ft1, ft2, aps1, aps2, sim):

        z1 = self.emb_net(ft1)
        z2 = self.emb_net(ft2)

        return