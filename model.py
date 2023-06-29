from network import EfficientNet
from dataloader import ChestXRayDataLoader
from metrics import DiceLoss, MixedLoss, MultiLabelAccuracy, MultiLabelAUROC, MultiLabelF1, MultiLabelPrecisionRecall

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os, csv, random
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as trF
from sklearn.metrics import multilabel_confusion_matrix, f1_score
from tensorboardX import SummaryWriter


#Model training parameters
# DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE == "cuda":
    torch.cuda.empty_cache()
BATCH_SIZE = 64
MODEL_NAME = "EfficientNet_1_NIH_Small_Sample_GreyScale"
LEARNING_RATE = 1e-3
LEARNING_RATE_SCHEDULE_FACTOR = 0.1
LEARNING_RATE_SCHEDULE_PATIENCE = 5
# large_file_dir = '/mnt/media/wiseyak/Chest_XRay_Model/'
large_file_dir = 'Datasets/'


class Model():
 
    def __init__(self, trained=False):
        self.model = EfficientNet().to(DEVICE)
        if trained: self.model.load_state_dict(torch.load(f'{large_file_dir}/saved_model/EfficientNet_1_55.pth'))
        # self.classes =  {
        #         'Atelectasis': 0, 
        #         'Cardiomegaly': 1, 
        #         'Consolidation': 2, 
        #         'Edema': 3, 
        #         'Effusion': 4, 
        #         'Emphysema': 5, 
        #         'Fibrosis': 6, 
        #         'Hernia': 7, 
        #         'Infiltration': 8, 
        #         'Mass': 9, 
        #         'No Finding': 10, 
        #         'Nodule': 11, 
        #         'Pleural_Thickening': 12, 
        #         'Pneumonia': 13, 
        #         'Pneumothorax': 14
        #     }       


    # def psnr(self, reconstructed, original, max_val=1.0): return 20 * torch.log10(max_val / torch.sqrt(F.mse_loss(reconstructed, original)))        



    def train(self, dataset, loss_func, optimizer, epoch):
        self.model.train()
        running_loss = 0.0
        running_total = 0
        true_labels = []
        predicted_labels = []

        for i, (images, labels) in tqdm(enumerate(dataset), total=len(dataset)):
            optimizer.zero_grad()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, outputs = self.model(images)
            loss = loss_func(outputs, labels)
            running_loss += loss.item()

            outputs = outputs
            predicted = (outputs > 0.6).float()  # Convert probabilities to binary predictions
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            running_total += labels.numel()

            loss.backward()
            optimizer.step()

        # Calculate metrics for the epoch
        epoch_loss = running_loss / len(dataset)

        true_labels_binary = np.array(true_labels)
        predicted_labels_binary = np.array(predicted_labels)
        f1 = f1_score(true_labels_binary, predicted_labels_binary, average='macro', zero_division=0)

        # Calculate confusion matrix
        conf_matrix = multilabel_confusion_matrix(true_labels_binary, predicted_labels_binary)

        # Save confusion matrix to CSV file
        with open(f'{MODEL_NAME}_confusion_matrix_train.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write epoch number
            writer.writerow(['Epoch:' + str(epoch)])

            # Write confusion matrix
            writer.writerow(['True/Predicted'] + [f'Class {i+1}' for i in range(conf_matrix.shape[0])])
            writer.writerow([''] + ['TP', 'FP', 'FN', 'TN'])

            tpr_list = []
            fpr_list = []
            tnr_list = []
            fnr_list = []

            for i in range(conf_matrix.shape[0]):
                class_name = f'Class {i+1}'
                tp = conf_matrix[i][1][1]
                fp = conf_matrix[i][0][1]
                fn = conf_matrix[i][1][0]
                tn = conf_matrix[i][0][0]
                writer.writerow([class_name, tp, fp, fn, tn])

                tpr = tp / (tp + fn)
                fpr = fp / (fp + tn)
                tnr = tn / (tn + fp)
                fnr = fn / (fn + tp)

                tpr_list.append(tpr)
                fpr_list.append(fpr)
                tnr_list.append(tnr)
                fnr_list.append(fnr)

            # Write TPR, FPR, TNR, FNR
            writer.writerow([])
            writer.writerow(['', 'TPR', 'FPR', 'TNR', 'FNR'])
            writer.writerow(['Average'] + [np.mean(tpr_list), np.mean(fpr_list), np.mean(tnr_list), np.mean(fnr_list)])

            # Write empty line for the next epoch
            writer.writerow([])

        return epoch_loss, f1, np.mean(tpr_list), np.mean(fpr_list), np.mean(tnr_list), np.mean(fnr_list)
      



    def validate(self, dataset):
        self.model.eval()
        running_total = 0
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for i, (img, labels) in tqdm(enumerate(dataset), total=len(dataset)):
                img, labels = img.to(DEVICE), labels.to(DEVICE)
                _, outputs = self.model(img)

                outputs = F.sigmoid(outputs)
                predicted = (outputs > 0.6).float()  # Convert probabilities to binary predictions
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

                running_total += labels.numel()

        # Calculate F1 score
        true_labels_binary = np.array(true_labels)
        predicted_labels_binary = np.array(predicted_labels)
        f1 = f1_score(true_labels_binary, predicted_labels_binary, average='macro', zero_division=0)

        return f1



    def test(self, dataset, epoch):
        running_total = 0
        true_labels = []
        predicted_labels = []

        num = random.randint(0, len(dataset) - 1)
        self.model.eval()

        with torch.no_grad():
            for i, (img, labels) in tqdm(enumerate(dataset), total=len(dataset)):
                img, labels = img.to(DEVICE), labels.to(DEVICE)
                _, outputs = self.model(img)

                outputs = F.sigmoid(outputs)
                predicted = (outputs > 0.6).float()  # Convert probabilities to binary predictions
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

                running_total += labels.numel()

        # Convert labels to binary format
        true_labels_binary = np.array(true_labels)
        predicted_labels_binary = np.array(predicted_labels)

        # Calculate confusion matrix
        conf_matrix = multilabel_confusion_matrix(true_labels_binary, predicted_labels_binary)

        f1 = f1_score(true_labels_binary, predicted_labels_binary, average='macro', zero_division=0)

        # Save confusion matrix to CSV file
        with open(f'{MODEL_NAME}_confusion_matrix_test.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write epoch number
            writer.writerow(['Epoch:' + str(epoch)])

            # Write confusion matrix
            writer.writerow(['True/Predicted'] + [f'Class {i+1}' for i in range(conf_matrix.shape[0])])
            writer.writerow([''] + ['TP', 'FP', 'FN', 'TN'])

            tpr_list = []
            fpr_list = []
            tnr_list = []
            fnr_list = []

            for i in range(conf_matrix.shape[0]):
                class_name = f'Class {i+1}'
                tp = conf_matrix[i][1][1]
                fp = conf_matrix[i][0][1]
                fn = conf_matrix[i][1][0]
                tn = conf_matrix[i][0][0]
                writer.writerow([class_name, tp, fp, fn, tn])

                tpr = tp / (tp + fn)
                fpr = fp / (fp + tn)
                tnr = tn / (tn + fp)
                fnr = fn / (fn + tp)

                tpr_list.append(tpr)
                fpr_list.append(fpr)
                tnr_list.append(tnr)
                fnr_list.append(fnr)

            # Write TPR, FPR, TNR, FNR
            writer.writerow([])
            writer.writerow(['', 'TPR', 'FPR', 'TNR', 'FNR'])
            writer.writerow(['Average'] + [np.mean(tpr_list), np.mean(fpr_list), np.mean(tnr_list), np.mean(fnr_list)])

            # Write empty line for the next epoch
            writer.writerow([])

        return conf_matrix, f1, np.mean(tpr_list), np.mean(fpr_list), np.mean(tnr_list), np.mean(fnr_list)





    def fit(self, epochs, lr):
        print(f"Using {DEVICE} device...")

        print("Initializing Parameters...")
        self.model = self.model.to(DEVICE)
        # model.load_state_dict(torch.load('/mnt/media/wiseyak/Chest_XRays/saved_model/EfficientNet_1_25.pth', map_location=DEVICE))
        total_params = sum(p.numel() for p in self.model.parameters())
        print("Total parameters of the model is: {:.2f}{}".format(total_params / 10**(3 * min(len(str(total_params)) // 3, len(["", "K", "M", "B", "T"]) - 1)), ["", "K", "M", "B", "T"][min(len(str(total_params)) // 3, len(["", "K", "M", "B", "T"]) - 1)]))

        print(f"Initializing the Optimizer")
        optimizer = optim.AdamW(self.model.parameters(), lr)

        print("Loading Datasets...")
        data_loader = ChestXRayDataLoader(batch_size=BATCH_SIZE)
        train_data, val_data, test_data, class_weights = data_loader.load_data()
        # train_data, class_weights = data_loader.load_data()
        weight_tensor = class_weights.to(DEVICE)

        print("Dataset Loaded.")
        binaryCrossEntropyLoss = nn.BCELoss(weight=weight_tensor)


        print(f"Beginning to train...")


        # mseloss = nn.MSELoss()
        train_loss_epochs, val_f1_epochs, test_f1_epochs = [], [], []
        writer = SummaryWriter(f'runs/{MODEL_NAME}/')
        os.makedirs("checkpoints/", exist_ok=True)
        os.makedirs(f"{large_file_dir}saved_model/", exist_ok=True)


        for epoch in range(1, epochs+1):

            print(f"Epoch No: {epoch}")

            '''------Training------'''
            train_loss, train_f1, train_tpr, train_fpr, train_tnr, train_fnr  = self.train(dataset=train_data, loss_func=binaryCrossEntropyLoss, optimizer=optimizer, epoch=epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("F1/train", train_f1, epoch)
            writer.add_scalar("TPR/train", train_tpr, epoch)
            writer.add_scalar("FPR/train", train_fpr, epoch)
            writer.add_scalar("TNR/train", train_tnr, epoch)
            writer.add_scalar("FNR/train", train_fnr, epoch)
            train_loss_epochs.append(train_loss)

            '''------Validation------'''
            val_f1 = self.validate(dataset=val_data)
            writer.add_scalar("F1/val", val_f1, epoch)
            val_f1_epochs.append(val_f1)
            
            print(f"Train Loss:{train_loss}, Train F1:{train_f1}, Validation F1:{val_f1}")


            '''------Testing------'''
            if epoch%5==0:
                test_confmtx, test_f1, test_tpr, test_fpr, test_tnr, test_fnr = self.test(dataset=test_data, epoch=epoch)
                test_f1_epochs.append(test_f1)
                print(f"Test F1: {test_f1}")
                writer.add_scalar("F1/test", test_f1, epoch)
                writer.add_scalar("TPR/test", test_tpr, epoch)
                writer.add_scalar("FPR/test", test_fpr, epoch)
                writer.add_scalar("TNR/test", test_tnr, epoch)
                writer.add_scalar("FNR/test", test_fnr,epoch)


            '''------Saving------'''
            print("Saving model")
            torch.save(self.model.state_dict(), f"{large_file_dir}saved_model/{MODEL_NAME}.pth")
            print("Model Saved")


            torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            }, f"checkpoints/{MODEL_NAME}_{epoch}.tar")
            
            
    
            print("Epoch Completed. Proceeding to next epoch...")

        print(f"Training Completed for {epochs} epochs.")





    def infer_a_sample(self, image):

        image = image.to(DEVICE)
        self.model.eval()
        # Forward pass the image through the model.
        prediction = nn.Softmax(dim=1)(self.model(image)).max(1)
        class_prob, class_index = round(prediction.values.item(), 3), prediction.indices.item()
        class_name = self.classes[class_index]
        return f'{class_name}: {class_prob*100}%'



model = Model(trained=False)
model.fit(60, 1e-3)
# data_loader = ChestXRayDataLoader(batch_size=BATCH_SIZE)
# train_data, val_data, test_data, class_weights = data_loader.load_data()
# model.test(dataset=test_data, epoch=0)