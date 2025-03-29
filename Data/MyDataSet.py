import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import hot
import pandas as pd
import os 
import pickle
from torch.utils.data import Dataset, DataLoader
import sys



class MultiMaskTimeSeriesDataset(Dataset):
    def __init__(self, data, u,missing_rate, missing_type='random', num_masks=5):
        """
        Initialize the dataset with time series data and missing pattern configuration.

        Parameters:
        data (np.ndarray): The original time series data of shape (num_days, num_time_steps, num_features).
        missing_rate (float): The rate of missing entries (0 < missing_rate < 1).
        missing_type (str): The type of missing pattern. Can be 'random', 'linear', or 'block' or 'mixed'.
        num_masks (int): The number of masks to generate for each day.
        """
        self.data = data
        self.missing_rate = missing_rate
        self.u = u
        self.missing_type = missing_type
        self.num_masks = num_masks
        if missing_type in ['random', 'linear', 'block']:
            self.masks = self.generate_masks(data.shape, missing_rate, missing_type, num_masks)
        elif missing_type == 'mixed':
            mask1= self.generate_masks(data.shape, 1-(1-missing_rate)**(1/3), 'random', num_masks)
            mask2= self.generate_masks(data.shape, 1-(1-missing_rate)**(1/3), 'linear', num_masks)
            mask3= self.generate_masks(data.shape, 1-(1-missing_rate)**(1/3), 'block', num_masks)
            # 将mask1, mask2, mask3对应相乘
            self.masks = mask1 * mask2 * mask3
            
        

    def __len__(self):
        return len(self.data) * self.num_masks

    def __getitem__(self, idx):
        day_idx = idx // self.num_masks
        mask_idx = idx % self.num_masks
        return self.data[day_idx], self.u,self.masks[day_idx, mask_idx]
    
    def get_historical_data(self, idx, history_length):
        """
        Get the historical data of a specific day.

        Parameters:
        idx (int): The index of the day.
        history_length (int): The length of the historical data.

        Returns:
        np.ndarray: The historical data of shape (history_length, num_time_steps, num_features).
        """
        day_idx = idx // self.num_masks
        return self.data[day_idx - history_length:day_idx]

    def generate_masks(self, shape, missing_rate, missing_type, num_masks):
        """
        Generate multiple missing masks for each day.

        Parameters:
        shape (tuple): Shape of the original data (num_days, num_time_steps, num_features).
        missing_rate (float): The rate of missing entries (0 < missing_rate < 1).
        missing_type (str): The type of missing pattern. Can be 'random', 'linear', or 'block'.
        num_masks (int): The number of masks to generate for each day.

        Returns:
        np.ndarray: A binary mask matrix of shape (num_days, num_masks, num_time_steps, num_features). 0: missing, 1: observed.
        """
        num_days, num_time_steps, num_features = shape
        masks = np.zeros((num_days, num_masks, num_time_steps, num_features), dtype=np.float16)

        for day in range(num_days):
            for m in range(num_masks):
                if missing_type == 'random':
                    masks[day, m] = np.random.rand(num_time_steps, num_features) < missing_rate

                elif missing_type == 'linear':
                    for j in range(num_features):
                        if np.random.rand() < missing_rate:
                            masks[day, m, :, j] = 1

                elif missing_type == 'block':
                    total_elements = num_time_steps * num_features
                    num_missing = int(total_elements * missing_rate)
                    current_missing = 0

                    while current_missing < num_missing:
                        block_height = np.random.randint(1, num_time_steps // 4)  # Random block height
                        block_width = np.random.randint(1, num_features // 4)   # Random block width
                        i = np.random.randint(0, num_time_steps - block_height + 1)
                        j = np.random.randint(0, num_features - block_width + 1)

                        masks[day, m, i:i+block_height, j:j+block_width] = 1
                        current_missing = np.sum(masks[day, m])

        return 1 - masks
