# src/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class StockDataset(Dataset):
    """Custom Dataset for stock price sequences"""
    
    def __init__(self, features, targets):
        """
        Args:
            features: numpy array of shape (n_samples, sequence_length, n_features)
            targets: numpy array of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
    def get_stats(self):
        """Get statistics about the dataset"""
        return {
            'num_samples': len(self),
            'feature_shape': self.features.shape,
            'target_stats': {
                'mean': self.targets.mean().item(),
                'std': self.targets.std().item(),
                'min': self.targets.min().item(),
                'max': self.targets.max().item()
            }
        }