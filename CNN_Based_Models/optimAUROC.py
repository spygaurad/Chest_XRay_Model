from libauc.losses.auc import mAUCMLoss
from libauc.losses import CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from libauc.models import densenet121 as DenseNet121
from libauc.datasets import CheXpert
from libauc.metrics import auc_roc_score # for multi-task
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F  
from tqdm import tqdm as tqdm



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


root = '/home/optimus/Downloads/Datasets/CheXpert-v1.0-small/'
# Index=-1 denotes multi-label with 5 diseases
traindSet = CheXpert(csv_path='CNN_Based_Models/Datasets/train.csv', image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='train', class_index=-1, verbose=False)
testSet =  CheXpert(csv_path='CNN_Based_Models/Datasets/valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='valid', class_index=-1, verbose=False)
trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=64, num_workers=2, shuffle=True)
testloader =  torch.utils.data.DataLoader(testSet, batch_size=64, num_workers=2, shuffle=False)
train_class_weights = torch.tensor(traindSet.imratio_list)
classes = traindSet.select_cols


class DenseNet(torch.nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.network = DenseNet121(pretrained=True, last_activation=None, activations='relu', num_classes=768)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(768, 128, bias=True)
        self.fc2 = torch.nn.Linear(128, 5, bias=True)

    def forward(self, x):
        feature = self.network(x)
        return self.fc2(self.relu(self.fc1(self.relu(feature))))



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
base_dir = "CNN_Based_Models/"

model = DenseNet()
model = model.cuda()

# define loss & optimizer
CELoss = torch.nn.BCEWithLogitsLoss(pos_weight=train_class_weights.to('cuda'))
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# '''
# training
best_val_auc = 0 
for epoch in range(2):
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader)):
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
                val_auc = roc_auc_score(test_true, test_pred, average="weighted")
                class_names = [f"{i}" for i in classes]
                metrics = []

                for i in range(len(classes)):
                    cm = confusion_matrix(test_true[:, i], (test_pred[:, i] > 0.5).astype(int))
                    tn, fp, fn, tp = cm.ravel()
                    tpr = tp / (tp + fn)
                    fpr = fp / (fp + tn)
                    fnr = fn / (fn + tp)
                    tnr = tn / (tn + fp)
                    precision, recall, f1, _ = precision_recall_fscore_support(test_true[:, i], (test_pred[:, i] > 0.5).astype(int), average="binary")

                    fpr_list, tpr_list, _ = roc_curve(test_true[:, i], test_pred[:, i])
                    class_auroc = roc_auc_score(test_true[:, i], test_pred[:, i])
                    metrics.append([tp, fp, fn, tn, tpr, fpr, fnr, tnr, precision, recall, f1, class_auroc])


                # Create a Markdown table and write it to a file
                with open(f"{base_dir}/output_dir/CE/training_metrics.txt", "a") as file:
                    file.write(f"Epoch: {epoch}, Iteration:{idx} Loss:BCE\n")
                    file.write("| Class | TP | FP | FN | TN | TPR | FPR | FNR | TNR | Precision | Recall | F1 | AUROC |\n")
                    file.write("|-------|-------|----|----|----|----|-----|-----|-----|-----|-----------|--------|----|\n")
                    for i in range(len(classes)):
                        file.write(f"| {class_names[i]} | {metrics[i][0]} | {metrics[i][1]} | {metrics[i][2]} | {metrics[i][3]} | {metrics[i][4]:.4f} | {metrics[i][5]:.4f} | {metrics[i][6]:.4f} | {metrics[i][7]:.4f} | {metrics[i][8]:.4f} | {metrics[i][9]:.4f} | {metrics[i][10]:.4f} | {metrics[i][11]:.4f} |\n")
                    file.write(f"\nAverate AUROC: {val_auc}\n\n\n")
                model.train()

                if best_val_auc < val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), f'{base_dir}/output_dir/CE/ce_pretrained_model_{epoch}.pth')
                    for i, class_name in enumerate(class_names):
                        fpr, tpr, _ = roc_curve(test_true[:, i], test_pred[:, i])
                        plt.figure()
                        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metrics[i][11]:.2f})')
                        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f'ROC Curve for {class_name}')
                        plt.legend(loc="lower right")
                        plt.savefig(f'{base_dir}/output_dir/CE/{class_name}.jpg')
                        plt.close()

                print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc, best_val_auc ))

print("Training With Cross Entropy Loss Complete")
#'''


#reinitialize the model
    
model = DenseNet()
model = model.cuda()
PATH = f'{base_dir}output_dir/CE/ce_pretrained_model_1.pth' 
state_dict = torch.load(PATH)
model.load_state_dict(state_dict)


# define loss & optimizer
loss_fn = mAUCMLoss(num_labels=5)
optimizer = PESG(model.parameters(), loss_fn=loss_fn, lr=lr, margin=margin, epoch_decay=epoch_decay, weight_decay=weight_decay)

output_dir = f"{base_dir}output_dir/auc_max/"
# training
print ('Start Training')
print ('-'*30)

best_val_auc = 0

for epoch in range(2):
    if epoch > 0:
        optimizer.update_regularizer(decay_factor=10)

    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        train_data, train_labels = data
        train_data, train_labels = train_data.cuda(), train_labels.cuda()
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

                # Calculate AUROC across all classes
                val_auc_mean = np.mean(roc_auc_score(test_true, test_pred))

                # Calculate and print per-class metrics
                class_names = [f"{i}" for i in classes]  # Replace num_classes with actual number of classes
                metrics = []

                for i in range(len(classes)):
                    cm = confusion_matrix(test_true[:, i], (test_pred[:, i] > 0.5).astype(int))
                    tn, fp, fn, tp = cm.ravel()
                    tpr = tp / (tp + fn)
                    fpr = fp / (fp + tn)
                    fnr = fn / (fn + tp)
                    tnr = tn / (tn + fp)
                    precision, recall, f1, _ = precision_recall_fscore_support(test_true[:, i], (test_pred[:, i] > 0.5).astype(int), average="binary")

                    fpr_list, tpr_list, _ = roc_curve(test_true[:, i], test_pred[:, i])
                    class_auroc = roc_auc_score(test_true[:, i], test_pred[:, i])
                    metrics.append([tp, fp, fn, tn, tpr, fpr, fnr, tnr, precision, recall, f1, class_auroc])

                # Create a Markdown table and write it to a file
                with open(f"{output_dir}metrics.txt", "a") as file:
                    file.write(f"Epoch: {epoch}, BatchID: {idx} Loss: {loss_fn}\n")
                    file.write("| Class | TP | FP | FN | TN | TPR | FPR | FNR | TNR | Precision | Recall | F1 | AUROC |\n")
                    file.write("|-------|----|----|----|----|-----|-----|-----|-----|-----------|--------|----|-------|\n")
                    for i in range(len(classes)):
                        file.write(f"| {class_names[i]} | {metrics[i][0]} | {metrics[i][1]} | {metrics[i][2]} | {metrics[i][3]} | {metrics[i][4]:.4f} | {metrics[i][5]:.4f} | {metrics[i][6]:.4f} | {metrics[i][7]:.4f} | {metrics[i][8]:.4f} | {metrics[i][9]:.4f} | {metrics[i][10]:.4f} | {metrics[i][11]:.4f} |\n")
                    file.write(f"\nAverage AUROC: {val_auc_mean}\n\n\n")

                model.train()

                # Check if the current model's performance is better than the best so far
                if best_val_auc < val_auc_mean:
                    best_val_auc = val_auc_mean
                    torch.save(model.state_dict(), f'{output_dir}/aucm_pretrained_model_{epoch}.pth')

                    # Generate and save AUROC curve plots for each class
                    for i, class_name in enumerate(class_names):
                        fpr, tpr, _ = roc_curve(test_true[:, i], test_pred[:, i])
                        plt.figure()
                        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metrics[i][11]:.2f})')
                        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f'ROC Curve for {class_name}')
                        plt.legend(loc="lower right")
                        plt.savefig(f'{output_dir}{class_name}.jpg')
                        plt.close()

            print('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f' % (epoch, idx, val_auc_mean, best_val_auc))

print("Training With mAUCM Loss Complete")