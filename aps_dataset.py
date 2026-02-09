import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from utils import heatmap, multi_plot


class APSSiameseDataset(Dataset):
    def __init__(self, features, aps_vectors):
        super().__init__()
        self.features = features
        self.aps_vectors = aps_vectors

        self.data = []

        for idx1, f_vec1 in enumerate(self.features):
            for idx2, f_vec2 in enumerate(self.features):
                similarity = F.pairwise_distance(self.aps_vectors[idx1], self.aps_vectors[idx2], p= 2)
                # similarity = F.cosine_similarity(self.aps_vectors[idx1], self.aps_vectors[idx2], dim= 0)
                y = torch.clamp((similarity - 0.9) / 0.1, 0.0, 1.0)
                self.data.append((f_vec1, f_vec2, self.aps_vectors[idx1], self.aps_vectors[idx2], similarity))
                # self.data.append((f_vec1, f_vec2, self.aps_vectors[idx1], y, similarity))                         # Returns the y mapped between 1 and 0

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

        return F.normalize(x, p= 2, dim= 1)


class SiameseNet(nn.Module):
    def __init__(self, emb_net):
        super().__init__()
        self.emb_net = emb_net
    
    def forward(self, ft1, ft2):

        z1 = self.emb_net(ft1)
        z2 = self.emb_net(ft2)

        return z1, z2
    
def contrastive_loss(x1, x2, label, margin=2.0):
    # Calculate L2 distance
    dist = F.pairwise_distance(x1, x2, p=2)
    
    # If label=1 (same), minimize distance. 
    # If label=0 (different), maximize distance up to margin.
    loss = (label * torch.pow(dist, 2) + (1 - label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2))
    
    return torch.mean(loss)

def manual_cosine_loss(x1, x2, y, margin=0.5):
    # Calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(x1, x2)

    y = torch.clamp((y - 0.9) / 0.1, 0.0, 1.0)
    # Loss for similar pairs
    loss_pos = 1 - cos_sim
    
    # Loss for dissimilar pairs (using ReLU to act as the 'max(0, ...)' margin)
    loss_neg = torch.relu(cos_sim - margin)
    
    # Combine based on label y (where y=1 is same, y=0 is different)
    total_loss = y * loss_pos + (1.0 - y) * loss_neg
    return total_loss.mean()



def train(train_dataloader, model, device, loss_fn, optim, loss_data):

    dataset_size = len(train_dataloader.dataset)
    batch_size = train_dataloader.batch_size

    model.train()

    for batch, (ft1, ft2, aps1, aps2, sim) in enumerate(train_dataloader):
        ft1, ft2, sim = ft1.to(device), ft2.to(device), sim.to(device)

        z1, z2 = model(ft1, ft2)
        loss = loss_fn(z1, z2, sim)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (batch%10) == 0:
            train_loss, current = loss.item(), batch * batch_size + len(z1)
            print(f"loss: {train_loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")
            loss_data.append(train_loss)


def validate(val_dataloader, model, device):

    dataset_size = len(val_dataloader.dataset)
    results = []

    model.eval()

    with torch.no_grad():
        for ft1, ft2, aps1, aps2, sim in val_dataloader:
            
            
            ft1, ft2, sim = ft1.to(device), ft2.to(device), sim.to(device)

            z1, z2 = model(ft1, ft2)
            # dist = F.pairwise_distance(z1, z2, p=2)
            dist = torch.sum(z1*z2, dim= 1)

            for similarity, distance in zip(sim, dist):
                results.append((round(similarity.item(), 6), round(distance.item(), 6)))

    
    print("Validation Results : \nAPS \t\t Embedding")
    APSData, EmbeddingData = [] , []
    for i in results:
        APSData.append(i[0])
        EmbeddingData.append(i[1])
        print(i[0], "\t", i[1])
    
    multi_plot(APSData, EmbeddingData)
    # heatmap(APSData, 'APSSim Heatmap')
    # heatmap(EmbeddingData, 'Embedding Heatmap')