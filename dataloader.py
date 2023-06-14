import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class ChestXRayDataset(Dataset):
    def __init__(self, csv_file, image_dir, num_classes):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.num_classes = num_classes
        self.class_mapping = self._create_class_mapping()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = os.path.join(self.image_dir, row['Image Index'])
        labels = row['Finding Labels'].split('|')
        label_vector = self._create_label_vector(labels)
        
    def _create_class_mapping(self):
        unique_labels = set()
        for labels in self.data['Finding Labels'].str.split('|'):
            unique_labels.update(labels)
        class_mapping = {label: i for i, label in enumerate(sorted(unique_labels))}
        return class_mapping
    
    def _create_label_vector(self, labels):
        label_vector = np.zeros(self.num_classes, dtype=np.float32)
        for label in labels:
            if label in self.class_mapping:
                label_index = self.class_mapping[label]
                label_vector[label_index] = 1.0
        return torch.from_numpy(label_vector)

