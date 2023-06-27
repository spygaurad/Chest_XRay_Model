from network import EfficientNet
from dataloader import ChestXRayDataLoader
from metrics import DiceLoss, MixedLoss

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, csv, random
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as trF
from sklearn.metrics import multilabel_confusion_matrix, f1_score
from tensorboardX import SummaryWriter


# from omnixai.data.image import Image as IM
# from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
BATCH_SIZE = 1
MODEL_NAME = "EfficientNet_1"
large_file_dir = '/mnt/media/wiseyak/Chest_XRays/'


class Model():
 
    def __init__(self, trained=False):
        self.model = EfficientNet().to(DEVICE)
        if trained: self.model.load_state_dict(torch.load('checkpoints/EfficientNet_1_35.tar')['model_state_dict'])
        self.classes =  {
                'Atelectasis': 0, 
                'Cardiomegaly': 1, 
                'Consolidation': 2, 
                'Edema': 3, 
                'Effusion': 4, 
                'Emphysema': 5, 
                'Fibrosis': 6, 
                'Hernia': 7, 
                'Infiltration': 8, 
                'Mass': 9, 
                'No Finding': 10, 
                'Nodule': 11, 
                'Pleural_Thickening': 12, 
                'Pneumonia': 13, 
                'Pneumothorax': 14
            }       


    # def psnr(self, reconstructed, original, max_val=1.0): return 20 * torch.log10(max_val / torch.sqrt(F.mse_loss(reconstructed, original)))        

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
        weight_tensor = class_weights.to(DEVICE)

        print("Dataset Loaded.")
        binaryCrossEntropyLoss = nn.BCEWithLogitsLoss()
        # bceLoss = nn.BCELoss()
        # mseloss = nn.MSELoss()


        print(f"Beginning to train...")


        # mseloss = nn.MSELoss()
        train_loss_epochs, val_f1_epochs, test_f1_epochs = [], [], []
        writer = SummaryWriter(f'runs/{MODEL_NAME}/')
        os.makedirs("checkpoints/", exist_ok=True)
        os.makedirs(f"{large_file_dir}saved_model/", exist_ok=True)


        for epoch in range(1, epochs+1):

            print(f"Epoch No: {epoch}")
            train_loss, train_f1 = self.train(dataset=train_data, loss_func=binaryCrossEntropyLoss, optimizer=optimizer)
            val_f1 = self.validate(dataset=val_data)
            train_loss_epochs.append(train_loss)
            val_f1_epochs.append(val_f1)

            if epoch%5==0:
                test_confmtx, test_f1 = self.test(dataset=test_data, epoch=epoch)
                test_f1_epochs.append(test_f1)
                print(f"Test Accuracy: {test_f1}")
                print("Saving model")
                torch.save(self.model.state_dict(), f"{large_file_dir}saved_model/{MODEL_NAME}_{epoch}.pth")
                print("Model Saved")
                writer.add_scalar("Accuracy/Test", test_f1, epoch)

            print(f"Train Loss:{train_loss}, Train Accuracy:{train_f1}, Validation Accuracy:{val_f1}")

            if max(val_f1_epochs) == val_f1:
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                }, f"checkpoints/{MODEL_NAME}_{epoch}.tar")

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_f1, epoch)
            writer.add_scalar("Accuracy/val", val_f1, epoch)
            
    
            print("Epoch Completed. Proceeding to next epoch...")

        print(f"Training Completed for {epochs} epochs.")



    def train(self, dataset, loss_func, optimizer):
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

            predicted = (outputs > 0.7).float()  # Convert probabilities to binary predictions
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            running_total += labels.numel()

            loss.backward()
            optimizer.step()

        # Calculate metrics for the epoch
        epoch_loss = running_loss / len(dataset)
        f1 = f1_score(true_labels, predicted_labels, average='macro')

        return epoch_loss, f1




    def validate(self, dataset):
        self.model.eval()
        running_total = 0
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for i, (img, labels) in tqdm(enumerate(dataset), total=len(dataset)):
                img, labels = img.to(DEVICE), labels.to(DEVICE)
                _, outputs = self.model(img)

                predicted = (outputs > 0.7).float()  # Convert probabilities to binary predictions
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

                running_total += labels.numel()

        # Calculate F1 score
        f1 = f1_score(true_labels, predicted_labels, average='macro')

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

                predicted = (outputs > 0.7).float()  # Convert probabilities to binary predictions
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

                running_total += labels.numel()

        # Convert labels to binary format
        true_labels_binary = np.array(true_labels)
        predicted_labels_binary = np.array(predicted_labels)

        # Calculate confusion matrix
        conf_matrix = multilabel_confusion_matrix(true_labels_binary, predicted_labels_binary)

        # Calculate F1 score
        f1 = f1_score(true_labels_binary, predicted_labels_binary, average='macro')

        # Save confusion matrix to CSV file
        with open('confusion_matrix.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            if epoch == 0:
                writer.writerow(['Epoch'])
            else:
                writer.writerow([])
            writer.writerow([f'Epoch {epoch + 1}'])
            writer.writerow([''] + [f'Class {i}' for i in range(len(conf_matrix))])
            for i in range(len(conf_matrix)):
                writer.writerow([f'Class {i}'] + list(conf_matrix[i].ravel()))

        return conf_matrix, f1


 



    def infer_a_sample(self, image):

        image = image.to(DEVICE)
        self.model.eval()
        # Forward pass the image through the model.
        prediction = nn.Softmax(dim=1)(self.model(image)).max(1)
        class_prob, class_index = round(prediction.values.item(), 3), prediction.indices.item()
        class_name = self.classes[class_index]
        return f'{class_name}: {class_prob*100}%'



model = Model(trained=True)
data_loader = ChestXRayDataLoader(batch_size=BATCH_SIZE)
train_data, val_data, test_data, class_weights = data_loader.load_data()
model.test(dataset=test_data, epoch=0)