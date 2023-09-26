from libauc.datasets import CheXpert
import torch
import torch.utils.data as data

root = '/home/optimus/Downloads/Datasets/CheXpert-v1.0-small/'

def initialize_datasets():
    def create_dataset(csv_path, mode):
        return CheXpert(
            csv_path=csv_path,
            image_root_path=root,
            use_upsampling=False,
            use_frontal=True,
            image_size=224,
            mode=mode,
            class_index=-1,
            verbose=False
        )
    
    train_set = create_dataset('CNN_Based_Models/Datasets/train.csv', 'train')
    test_set = create_dataset('CNN_Based_Models/Datasets/valid.csv', 'valid')
    
    return train_set, test_set


def initialize_dataloaders(train_set, test_set):
    def create_dataloader(dataset):
        return data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=2,
            shuffle=True if dataset.mode == 'train' else False
        )
    
    train_loader = create_dataloader(train_set)
    test_loader = create_dataloader(test_set)
    
    return train_loader, test_loader


def get_dataloaders_and_properties():
    train_set, test_set = initialize_datasets()
    train_loader, test_loader = initialize_dataloaders(train_set, test_set)
    
    train_class_weights = torch.tensor(train_set.imratio_list)
    classes = train_set.select_cols
    
    return train_loader, test_loader, train_class_weights, classes
