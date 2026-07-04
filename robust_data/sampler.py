import torch
from torch.utils.data import WeightedRandomSampler


def create_weighted_sampler(train_df):
    class_counts = train_df["emotion"].value_counts().to_dict()

    sample_weights = [1.0 / class_counts[e] for e in train_df["emotion"]]

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    return sampler