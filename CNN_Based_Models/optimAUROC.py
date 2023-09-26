from libauc.losses.auc import mAUCMLoss
from libauc.optimizers import PESG, Adam
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import torch 
from tqdm import tqdm as tqdm

from network import DenseNetWithPCAM as DenseNet
from dataloader import get_dataloaders_and_properties

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# paramaters
SEED = 123
BATCH_SIZE = 32
model_name = "Densenet"
device = "cuda" if torch.cuda.is_available() else 'cpu'
base_dir = "CNN_Based_Models/"
set_all_seeds(SEED)

#model's properties
model = DenseNet()
model = model.to("device")

#dataloader
trainloader, testloader, class_weights, classes = get_dataloaders_and_properties()


def train(model, trainloader, testloader, class_weights, classes, lr=1e-4, epoch_decay=2e-3, weight_decay=1e-5, margin=1.0, total_epochs=2, include_class_weights=False):

    def plot_roc_curves(fpr_list, tpr_list, area, class_name):
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {area:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {class_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'{base_dir}/output_dir/CE/{class_name}.jpg')
        plt.close()


    def calculate_metrics(test_true, test_pred, threshold=0.5):

        def calc_tpr(tp, fn): return tp/(tp+fn)
        def calc_fpr(fp, tn): return fp/(fp+tn)
        def calc_fnr(fn, tp): return fn/(fn+tp)
        def calc_tnr(tn, fp): return tn/(tn+fp)

        metrics = []   

        for i in range(len(classes)):
            fpr_list, tpr_list, _ = roc_curve(test_true[:, i], test_pred[:, i])
            cm = confusion_matrix(test_true[:, i], (test_pred[:, i] > threshold).astype(int))
            tn, fp, fn, tp = cm.ravel()
            tpr = calc_tpr(tp, fn)
            fpr = calc_fpr(fp, tn)
            fnr = calc_fnr(fn, tp)
            tnr = calc_tnr(tn, fp)
            precision, recall, f1, _ = precision_recall_fscore_support(test_true[:, i], (test_pred[:, i] > 0.5).astype(int), average="binary")
            class_auroc = roc_auc_score(test_true[:, i], test_pred[:, i])
            plot_roc_curves(fpr_list, tpr_list, class_auroc, classes[i])
            metrics.append([tp, fp, fn, tn, tpr, fpr, fnr, tnr, precision, recall, f1, class_auroc])
        
        return metrics
    

    def write_metrics_in_file(epoch, idx, CELoss, classes, val_auc, metrics):
        with open(f"{base_dir}/output_dir/CE/training_metrics.txt", "a") as file:
            file.write(f"Epoch: {epoch}, Iteration:{idx} Loss:{CELoss}\n")
            file.write("| Class | TP | FP | FN | TN | TPR | FPR | FNR | TNR | Precision | Recall | F1 | AUROC |\n")
            file.write("|-------|-------|----|----|----|----|-----|-----|-----|-----|-----------|--------|----|\n")
            for i in range(len(classes)):
                file.write(f"| {classes[i]} | {metrics[i][0]} | {metrics[i][1]} | {metrics[i][2]} | {metrics[i][3]} | {metrics[i][4]:.4f} | {metrics[i][5]:.4f} | {metrics[i][6]:.4f} | {metrics[i][7]:.4f} | {metrics[i][8]:.4f} | {metrics[i][9]:.4f} | {metrics[i][10]:.4f} | {metrics[i][11]:.4f} |\n")
            file.write(f"\nAverate AUROC: {val_auc}\n\n\n")


    def save_model(model):
        torch.save(model.state_dict(), f'{base_dir}/output_dir/CE/hinge_pretrained_model_{epoch}.pth')


    def train_epoch(model, loss_fn, device, trainloader):
        for idx, (train_data, train_labels) in tqdm(enumerate(trainloader), total=len(trainloader)):
            train_data, train_labels  = train_data.to(device), train_labels.to(device)
            optimizer.zero_grad()
            y_pred = model(train_data)
            loss = CELoss(y_pred, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    print("First Phase Training...")
    if include_class_weights:
        CELoss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to('cuda'))
    else:
        CELoss = torch.nn.BCEWithLogitsLoss()

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = 0 

    for epoch in range(2):

            
            # validation  
            if idx % 200 == 0 or idx==(len(trainloader)-1):
                model.eval()
                with torch.no_grad():    
                    test_pred = []
                    test_true = [] 
                    for jdx, data in enumerate(testloader):
                        test_data, test_labels = data
                        test_data = test_data.cuda()
                        y_pred = model(test_data)
                        test_pred.append(torch.nn.functional.sigmoid(y_pred).cpu().detach().numpy())
                        test_true.append(test_labels.numpy())

                    test_true = np.concatenate(test_true)
                    test_pred = np.concatenate(test_pred)
                    
                    val_auc = roc_auc_score(test_true, test_pred, average="weighted")

                    if best_val_auc < val_auc:
                        best_val_auc = val_auc
                        save_model(model)
                        metrics = calculate_metrics(test_true, test_pred, 0.5)
                        write_metrics_in_file(epoch, idx, CELoss, classes, val_auc, metrics)
                    model.train()
                    print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc, best_val_auc ))

    print("Training With Cross Entropy Loss Complete")
    #'''


    #reinitialize the model
        
    model = DenseNet()
    model = model.cuda()
    PATH = f'{base_dir}output_dir/auc_max/aucm_pretrained_model_0.pth' 
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

    for epoch in range(5):
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