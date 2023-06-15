import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(self, csv_file, image_dir, num_classes, transform):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.num_classes = num_classes
        self.class_mapping = self._create_class_mapping()
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        

    def __len__(self):
        return len(self.data[:])

def __getitem__(self, index):
    row = self.data.iloc[index]
    image_path = os.path.join(self.image_dir, row['Image Index'])
    labels = row['Finding Labels'].split('|')
    label_vector = self._create_label_vector(labels)

    try:
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
    except (OSError, IOError):
        return None, None

    return image, label_vector


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


class ChestXRayDataLoader:
    def __init__(self, train_percent=0.8, val_percent=0.1, batch_size=32, num_workers=4, seed=42):
        self.dataset = ChestXRayDataset(csv_file='/home/optimus/Downloads/Dataset/ChestXRays/NIH/Updated_Data_Entry_2017.csv', image_dir='/home/optimus/Downloads/Dataset/ChestXRays/NIH/images/', num_classes=15)
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()

    def _create_data_loaders(self):
        # Calculate the number of samples for each split
        num_samples = len(self.dataset)
        num_train = int(self.train_percent * num_samples)
        num_val = int(self.val_percent * num_samples)
        num_test = num_samples - num_train - num_val

        # Set the random seed for reproducibility
        torch.manual_seed(self.seed)

        # Split the dataset into train, val, and test sets
        train_set, val_set, test_set = random_split(self.dataset, [num_train, num_val, num_test])

        # Create the data loaders
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, )
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader


