import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import torch
import torch.nn as nn
import getpass
from aps_dataset import APSSiameseDataset, EmbeddingNet, SiameseNet, contrastive_loss, train, validate, manual_cosine_loss
from torch.utils.data import Dataset, DataLoader, random_split
from  utils import plot_loss

# password = getpass.getpass("Enter DB Password: ")
password = 'SlsFux11'
engine = create_engine(f"postgresql://randima:{password}@localhost:5432/movielens")


# Queries for accessing Database


query_complete_d = """
SELECT
    performance_results.data_set_name AS dataset,
    COUNT(performance_results.data_set_name) AS dataset_freq
FROM
    performance_results
    
GROUP BY
    performance_results.data_set_name
HAVING
    COUNT(performance_results.data_set_name) = 29;
"""
query_aps_ndcg5 = """
SELECT
    performance_results.data_set_name AS dataset,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name = 'BPR') AS BPR,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'CDAE') AS CDAE,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ConvNCF') AS ConvNCF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DGCF') AS DGCF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DiffRec') AS DiffRec,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DMF') AS DMF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'EASE') AS EASE,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ENMF') AS ENMF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'FISM') AS FISM,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'GCMC') AS GCMC,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ItemKNN') AS ItemKNN,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LDiffRec') AS LDiffRec,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LightGCN') AS LightGCN,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LINE') AS LINE_alg,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MacridVAE') AS MacridVAE,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiDAE') AS MultiDAE,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiVAE') AS MultiVAE,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NAIS') AS NAIS,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCEPLRec') AS NCEPLRec,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCL') AS NCL,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NeuMF') AS NeuMF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NGCF') AS NGCF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NNCF') AS NNCF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'Pop') AS Pop,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'Random') AS Random,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'RecVAE') AS RecVAE,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SGL') AS SGL,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SimpleX') AS SimpleX,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SpectralCF') AS SpectralCF

FROM
    performance_results
GROUP BY
    performance_results.data_set_name;

"""

query_aps_ndcg10 ="""
SELECT
    performance_results.data_set_name AS dataset,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name = 'BPR') AS BPR,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'CDAE') AS CDAE,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'ConvNCF') AS ConvNCF,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'DGCF') AS DGCF,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'DiffRec') AS DiffRec,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'DMF') AS DMF,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'EASE') AS EASE,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'ENMF') AS ENMF,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'FISM') AS FISM,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'GCMC') AS GCMC,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'ItemKNN') AS ItemKNN,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'LDiffRec') AS LDiffRec,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'LightGCN') AS LightGCN,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'LINE') AS LINE_alg,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'MacridVAE') AS MacridVAE,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiDAE') AS MultiDAE,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiVAE') AS MultiVAE,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'NAIS') AS NAIS,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCEPLRec') AS NCEPLRec,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCL') AS NCL,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'NeuMF') AS NeuMF,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'NGCF') AS NGCF,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'NNCF') AS NNCF,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'Pop') AS Pop,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'Random') AS Random,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'RecVAE') AS RecVAE,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'SGL') AS SGL,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'SimpleX') AS SimpleX,
    SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'SpectralCF') AS SpectralCF

FROM
    performance_results
GROUP BY
    performance_results.data_set_name;
"""

query_features = """
SELECT
    aps_merged.data_set_name AS dataset,
    AVG(aps_merged.num_users)::INT AS num_users,
    AVG(aps_merged.num_items)::INT AS num_items,
    AVG(aps_merged.num_interactions)::INT AS num_inter,
    ROUND(AVG(aps_merged.density),6) AS density,
    ROUND(AVG(aps_merged.user_item_ratio),6) AS u_i_ratio,
    ROUND(AVG(aps_merged.item_user_ratio),6) AS i_u_ratio,
    AVG(aps_merged.highest_num_rating_by_single_user)::INT AS highest_num_rating_u,
    AVG(aps_merged.lowest_num_rating_by_single_user)::INT AS lowest_num_rating_u,
    AVG(aps_merged.highest_num_rating_on_single_item)::INT AS highest_num_rating_i,
    AVG(aps_merged.lowest_num_rating_on_single_item)::INT AS lowest_num_rating_i,
    ROUND(AVG(aps_merged.mean_num_ratings_by_user),6) AS mean_num_rating_u,
    ROUND(AVG(aps_merged.mean_num_ratings_on_item),6) AS mean_num_rating_i
FROM
    aps_merged
GROUP BY
    aps_merged.data_set_name;
"""


complete_datasets_df = pd.read_sql(query_complete_d, engine)
aps_df = pd.read_sql(query_aps_ndcg10, engine).set_index("dataset", drop= False)
features_df = pd.read_sql(query_features, engine).set_index("dataset", drop= False)


mask = complete_datasets_df['dataset'].to_list()


aps_vectors = aps_df.loc[mask].reset_index(drop= True)
feature_vectors = features_df.loc[mask].reset_index(drop= True)

tempaps_df = aps_vectors.iloc[:, 1:]
temp_df = feature_vectors.iloc[:, 1:]
t_aps = torch.tensor(tempaps_df.values, dtype=torch.float32)
t_f = torch.tensor(temp_df.values, dtype=torch.float32)

dataset = APSSiameseDataset(t_f, t_aps)

train_set_size = int(0.95*len(dataset))
val_set_size = len(dataset) - train_set_size

train_set, val_set = random_split(dataset, [train_set_size, val_set_size])

train_dataloader = DataLoader(
    train_set,
    batch_size= 16,
    shuffle= True,
    num_workers= 5,
    pin_memory= True
)

val_dataloader = DataLoader(
        val_set,
        batch_size= 64,
        shuffle= False,
        num_workers= 5,
        pin_memory= True
)


device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
print(f"Using {device} device")



epochs = 15

model = SiameseNet(EmbeddingNet()).to(device)
# loss_func = contrastive_loss
# loss_func = nn.CosineEmbeddingLoss(margin= 1.0)
loss_func = manual_cosine_loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loss_data = []

for t in range(epochs):
    print(f"Running Epoch{t}")
    train(train_dataloader, model, device, loss_func, optimizer, loss_data)

plot_loss(loss_data)

validate(val_dataloader, model, device)

# torch.save(model.state_dict(), 'weights/APSSiamNet_weights_4.pth')
# print("Model saved!")