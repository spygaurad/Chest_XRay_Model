import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import csv

root_dir = '/home/optimus/Downloads/Dataset/ChestXRays/NIH/'

class ChestXRayDataset(Dataset):
    def __init__(self, csv_file, image_dir, num_classes):
        self.data = self._read_csv(csv_file)
        self.image_dir = image_dir
        self.num_classes = num_classes
        self.class_mapping = self._create_class_mapping()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.data[:])
    
    def __getitem__(self, index):
        row = self.data[index]
        image_path = os.path.join(self.image_dir, row[0])
        labels = row[1].split('|')
        label_vector = self._create_label_vector(labels)

        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except (OSError, IOError):
            image = torch.rand(3, 256, 256)

        return image, label_vector
    
    def _create_class_mapping(self):
        unique_labels = set()
        for row in self.data:
            labels = row[1].split('|')
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
    
    def _read_csv(self, csv_file):
        data = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                data.append(row)
        return data


class ChestXRayDataLoader:
    def __init__(self, num_classes=15, train_percent=0.8, val_percent=0.1, batch_size=64, num_workers=4, seed=42):
        self.train_dataset = ChestXRayDataset(csv_file=f'{root_dir}train.csv', image_dir=f'{root_dir}images/', num_classes=num_classes)
        self.test_dataset = ChestXRayDataset(csv_file=f'{root_dir}test.csv', image_dir=f'{root_dir}images/', num_classes=num_classes)
        self.val_dataset = ChestXRayDataset(csv_file=f'{root_dir}val.csv', image_dir=f'{root_dir}images/', num_classes=num_classes)
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()

    def _create_data_loaders(self):
        # Create the data loaders
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader
