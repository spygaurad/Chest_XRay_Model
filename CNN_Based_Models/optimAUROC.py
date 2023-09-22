from libauc.losses.auc import mAUCMLoss
from libauc.losses import CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from libauc.models import densenet121 as DenseNet121
from libauc.datasets import CheXpert
from libauc.metrics import auc_roc_score # for multi-task

from PIL import Image
import numpy as np
import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F  



def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(y):
    class_counts = np.sum(y, axis=0)
    class_weights = class_counts / np.sum(class_counts)
    return class_counts


root = './CheXpert/CheXpert-v1.0-small/'
# Index=-1 denotes multi-label with 5 diseases
traindSet = CheXpert(csv_path='Datasets/train.csv', image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='train', class_index=-1, verbose=False)
testSet =  CheXpert(csv_path='Datasets/valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='valid', class_index=-1, verbose=False)
trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=64, num_workers=2, shuffle=True)
testloader =  torch.utils.data.DataLoader(testSet, batch_size=64, num_workers=2, shuffle=False)
train_class_weights = calculate_class_weights(traindSet.labels)

# Print the class weights.
print(train_class_weights)

# check imbalance ratio for each task
print (traindSet.imratio_list)

# paramaters
SEED = 123
BATCH_SIZE = 32
lr = 1e-4
epoch_decay = 2e-3
weight_decay = 1e-5
margin = 1.0
total_epochs = 2
set_all_seeds(SEED)
model_name = "Densenet"

model = torch.nn.Sequential(DenseNet121(pretrained=True, last_activation=None, activations='relu', num_classes=768), torch.nn.ReLU(), torch.nn.Linear(768, 128, bias=True), torch.nn.ReLU(), torch.nn.Linear(128, 5, bias=True))
model = model.cuda()

# define loss & optimizer
CELoss = torch.nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# training
best_val_auc = 0 
for epoch in range(1):
    for idx, data in enumerate(trainloader):
        train_data, train_labels = data
        train_data, train_labels  = train_data.cuda(), train_labels.cuda()
        y_pred = model(train_data)
        loss = CELoss(y_pred, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # validation  
    if idx % 400 == 0:
        model.eval()
        with torch.no_grad():    
            test_pred = []
            test_true = [] 
            for jdx, data in enumerate(testloader):
                test_data, test_labels = data
                test_data = test_data.cuda()
                y_pred = model(test_data)
                test_pred.append(y_pred.cpu().detach().numpy())
                test_true.append(test_labels.numpy())
            
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            val_auc_mean =  roc_auc_score(test_true, test_pred) 
            model.train()
            
            if best_val_auc < val_auc_mean:
                best_val_auc = val_auc_mean
                torch.save(model.state_dict(), f'output_dir/{model_name}/ce_pretrained_model_{epoch}.pth')
                
            print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc ))

print("Training With Cross Entropy Loss Complete")


#reinitialize the model
model = torch.nn.Sequential(DenseNet121(pretrained=True, last_activation=None, activations='relu', num_classes=768), torch.nn.ReLU(), torch.nn.Linear(768, 128, bias=True), torch.nn.ReLU(), torch.nn.Linear(128, 5, bias=True))
model = model.cuda()
PATH = 'ce_pretrained_model.pth' 
state_dict = torch.load(PATH)
model.load_state_dict(state_dict)


# define loss & optimizer
loss_fn = mAUCMLoss(num_classes=5)
optimizer = PESG(model, loss_fn=loss_fn,lr=lr, margin=margin, epoch_decay=epoch_decay, weight_decay=weight_decay)


# training
print ('Start Training')
print ('-'*30)

best_val_auc = 0 
for epoch in range(2):
    if epoch > 0:
        optimizer.update_regularizer(decay_factor=10)    

    for idx, data in enumerate(trainloader):
        train_data, train_labels = data
        train_data, train_labels  = train_data.cuda(), train_labels.cuda()
        y_pred = model(train_data)
        y_pred = torch.sigmoid(y_pred)
        loss = loss_fn(y_pred, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # validation  
        if idx % 400 == 0:
            model.eval()
            with torch.no_grad():    
                test_pred = []
                test_true = [] 
                for jdx, data in enumerate(testloader):
                    test_data, test_labels = data
                    test_data = test_data.cuda()
                    y_pred = model(test_data)
                    y_pred = torch.sigmoid(y_pred)
                    test_pred.append(y_pred.cpu().detach().numpy())
                    test_true.append(test_labels.numpy())

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            val_auc_mean = np.mean(auc_roc_score(test_true, test_pred)) 
            model.train()
            if best_val_auc < val_auc_mean:
                best_val_auc = val_auc_mean
                torch.save(model.state_dict(), f'aucm_pretrained_model_{epoch}.pth')

            print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc))



