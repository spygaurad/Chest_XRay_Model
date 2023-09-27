from libauc.losses.auc import mAUCMLoss
from libauc.optimizers import PESG, Adam
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import torch 
from tqdm import tqdm as tqdm
import cv2

from network import DenseNetWithPCAM as DenseNet
from dataloader import get_dataloaders_and_properties
from gradcam_fns import gradcam_pcam, overlay_heatmap

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# paramaters
SEED = 123
BATCH_SIZE = 64
model_name = "Densenet"
device = "cuda" if torch.cuda.is_available() else 'cpu'
base_dir = "CNN_Based_Models/"
set_all_seeds(SEED)

print(device)


def train(lr=1e-4, epoch_decay=2e-3, weight_decay=1e-5, margin=1.0, total_epochs=2, include_class_weights=True):
    
    #model's properties
    model = DenseNet()
    model = model.to(device)

    #dataloader
    trainloader, testloader, class_weights, classes = get_dataloaders_and_properties()


    def calculate_metrics(test_true, test_pred, output_dir, threshold=0.5):

        def calc_tpr(tp, fn): return tp/(tp+fn)
        def calc_fpr(fp, tn): return fp/(fp+tn)
        def calc_fnr(fn, tp): return fn/(fn+tp)
        def calc_tnr(tn, fp): return tn/(tn+fp)

        def plot_roc_curves(fpr_list, tpr_list, area, class_name):
            plt.figure()
            plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label=f'ROC curve (area = {area:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {class_name}')
            plt.legend(loc="lower right")
            plt.savefig(f'{output_dir}/{class_name}.jpg')
            plt.close()

        metrics = []   

        for i in range(len(classes)):
            fpr_list, tpr_list, _ = roc_curve(test_true[:, i], test_pred[:, i])
            cm = confusion_matrix(test_true[:, i], (test_pred[:, i] > threshold).astype(int))
            tn, fp, fn, tp = cm.ravel()
            tpr = calc_tpr(tp, fn)
            fpr = calc_fpr(fp, tn)
            fnr = calc_fnr(fn, tp)
            tnr = calc_tnr(tn, fp)
            precision, recall, f1, _ = precision_recall_fscore_support(test_true[:, i], (test_pred[:, i] > 0.5).astype(int), average="binary", zero_division=1)
            class_auroc = roc_auc_score(test_true[:, i], test_pred[:, i])
            plot_roc_curves(fpr_list, tpr_list, class_auroc, classes[i])
            metrics.append([tp, fp, fn, tn, tpr, fpr, fnr, tnr, precision, recall, f1, class_auroc])
        
        return metrics
    

    def write_metrics_in_file(epoch, idx, CELoss, classes, val_auc, metrics, output_dir):
        with open(f"{output_dir}training_metrics.txt", "a") as file:
            file.write(f"Epoch: {epoch}, Iteration:{idx} Loss:{CELoss}\n\n")
            file.write("| Class | TP | FP | FN | TN | TPR | FPR | FNR | TNR | Precision | Recall | F1 | AUROC |\n")
            file.write("|-------|-------|----|----|----|----|-----|-----|-----|-----|-----------|--------|----|\n")
            for i in range(len(classes)):
                file.write(f"| {classes[i]} | {metrics[i][0]} | {metrics[i][1]} | {metrics[i][2]} | {metrics[i][3]} | {metrics[i][4]:.4f} | {metrics[i][5]:.4f} | {metrics[i][6]:.4f} | {metrics[i][7]:.4f} | {metrics[i][8]:.4f} | {metrics[i][9]:.4f} | {metrics[i][10]:.4f} | {metrics[i][11]:.4f} |\n")
            file.write(f"\nAverate AUROC: {val_auc}\n\n\n")


    def save_model(epoch, output_dir):
        torch.save(model.state_dict(), f'{output_dir}/model_{epoch}.pth')


    def eval_epoch(epoch, idx, loss_fn, output_dir, best_val_auc):
        model.eval()
        with torch.no_grad():    
            test_pred = []
            test_true = [] 
            for jdx, data in enumerate(testloader):
                test_data, test_labels = data
                test_data = test_data.to(device)
                y_pred, _, _ = model(test_data)
                test_pred.append(torch.nn.functional.sigmoid(y_pred).cpu().detach().numpy())
                test_true.append(test_labels.numpy())
                torch.cuda.empty_cache()

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            
            val_auc = roc_auc_score(test_true, test_pred, average="weighted")

            if best_val_auc < val_auc:
                metrics = calculate_metrics(test_true, test_pred, output_dir, 0.5)
                write_metrics_in_file(epoch, idx, loss_fn, classes, val_auc, metrics, output_dir)

        model.train()
        return val_auc


    def train_epoch(epoch, loss_fn, output_dir):
        
        model.train()
        best_val_auc = 0
        for idx, (train_data, train_labels) in tqdm(enumerate(trainloader), total=len(trainloader)):

            train_data, train_labels  = train_data.to(device), train_labels.to(device)
            optimizer.zero_grad()
            y_pred, _, _ = model(train_data)
            loss = loss_fn(y_pred, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
            if idx % 200 == 0 or idx==(len(trainloader)-1):
                val_auc = eval_epoch(epoch, idx, loss_fn, output_dir, best_val_auc)
                if best_val_auc < val_auc:
                    save_model(epoch, output_dir)
                    best_val_auc = val_auc
                    print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc, best_val_auc ))
                

    print("First Phase Training...")
    if include_class_weights:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to('cuda'))
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    output_dir = f"{base_dir}output_dir/CE/"

    for epoch in range(5):
        train_epoch(epoch, loss_fn, output_dir)

    print("Training With Cross Entropy Loss Complete")


    #load the latest parameters
    PATH = f'{base_dir}output_dir/CE/model_1.pth' 
    state_dict = torch.load(PATH)
    model.load_state_dict(state_dict)


    # define loss & optimizer
    loss_fn = mAUCMLoss(num_labels=5)
    optimizer = PESG(model.parameters(), loss_fn=loss_fn, lr=lr, margin=margin, epoch_decay=epoch_decay, weight_decay=weight_decay)
    output_dir = f"{base_dir}output_dir/auc_max/"

    for epoch in range(5):
        train_epoch(epoch, loss_fn, output_dir)

    print("Training With mAUCM Loss Complete")



def infer_a_sample(image):

    def load_model():
        model = DenseNet()
        PATH = f'{base_dir}output_dir/CE/model_1.pth' 
        state_dict = torch.load(PATH)
        model.load_state_dict(state_dict)
        return model
    
    def preprocess_image(image, image_size=224):
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
        image = image / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = image.transpose((2, 0, 1)).astype(np.float32)
        image = torch.tensor(image)
        image = image.unsqueeze(dim=0)
        return image

    model = load_model()
    model = model.to(device)
    _input = preprocess_image(image)
    _input = _input.to(device)

    class_labels = get_dataloaders_and_properties()[3]
    result_images = []
    inference_dir = "/home/wiseyak/saumya/Chest_XRay_Model/CNN_Based_Models/output_dir/inferences/"
    for target_class, class_name in enumerate(class_labels):
        heatmap, _, logits = gradcam_pcam(model, _input, target_class)
        result_image = overlay_heatmap(image, heatmap, class_name)
        result_image.save(f"{inference_dir}{class_name}.png")
        result_images.append(result_image)
        break
    
    return result_images[0], logits


# from PIL import Image
# path = "/home/optimus/Downloads/Datasets/CheXpert-v1.0-small/train/patient00007/study2/view1_frontal.jpg"
# image = Image.open(path)
# infer_a_sample(image)

# infer_a_sample()
# if name == "__main__":
# train()