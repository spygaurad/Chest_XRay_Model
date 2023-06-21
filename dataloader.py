import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import csv

root_dir = '/home/optimus/Downloads/Dataset/ChestXRays/NIH/'

from torchvision import transforms
import torch
import csv
import os
import numpy as np
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(self, csv_file, image_dir, num_classes):
        self.data = self._read_csv(csv_file)
        self.image_dir = image_dir
        self.num_classes = num_classes
        self.class_mapping = self._create_class_mapping()
        self.mean, self.std = self._calculate_mean_std()
        self.transform = self._create_transform()
        self.weight_tensor = self._calculate_class_weights()

    def __len__(self):
        return len(self.data[:int((len(self.data)) * 0.3)])

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


    def _create_transform(self):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ])
        return transform

    def _calculate_class_weights(self):
        class_counts = np.zeros(self.num_classes, dtype=np.float32)
        total_samples = 0

        for instance in self.data:
            labels = instance[1]
            labels = labels.split('|')
            label_vector = self._create_label_vector(labels)
            class_counts += label_vector.numpy()
            total_samples += 1

        class_weights = total_samples / (class_counts * self.num_classes)
        weight_tensor = torch.from_numpy(class_weights).float()

        return weight_tensor




class ChestXRayDataLoader:
    def __init__(self, batch_size, num_classes=15):
        image_dir = f'{root_dir}/images/'
        self.train_dataset = ChestXRayDataset(f'Datasets/multilabel_classification/train.csv', image_dir, num_classes)
        self.val_dataset = ChestXRayDataset(f'Datasets/multilabel_classification/val.csv', image_dir, num_classes)
        self.test_dataset = ChestXRayDataset(f'Datasets/multilabel_classification/test.csv', image_dir, num_classes)
        self.batch_size = batch_size

    def load_data(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        # Calculate class weights only for the training dataset
        class_weights = self.train_dataset.weight_tensor

        return train_loader, val_loader, test_loader, class_weights