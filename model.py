import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as trF
from tensorboardX import SummaryWriter

from network import EfficientNet
from dataloader import ChestXRayDataLoader
from metrics import DiceLoss, MixedLoss

# from omnixai.data.image import Image as IM
# from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
BATCH_SIZE = 64
MODEL_NAME = "EfficientNet_1"
large_file_dir = '/mnt/media/wiseyak/Chest_XRays/'


class Model():
 
    def __init__(self, trained=False):
        self.model = EfficientNet().to(DEVICE)
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


    def train(self, dataset, loss_func, optimizer):
        self.model.train()
        running_loss = 0.0
        running_correct = 0.0
        running_total = 0

        for i, (images, labels) in tqdm(enumerate(dataset), total=len(dataset)):
            optimizer.zero_grad()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, outputs = self.model(images)
            loss = loss_func(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.7).float()  # Convert probabilities to binary predictions
            correct = torch.sum(predicted == labels).item()
            running_correct += correct
            running_total += labels.numel()

            loss.backward()
            optimizer.step()

        # Calculate metrics for the epoch
        epoch_loss = running_loss / len(dataset)
        epoch_acc = (running_correct / running_total) * 100

        return epoch_loss, epoch_acc




    def validate(self, dataset):

        self.model.eval()
        running_correct = 0.0
        running_total = 0

        with torch.no_grad():
            for i, (img, labels) in tqdm(enumerate(dataset), total=len(dataset)):
                img, labels = img.to(DEVICE), labels.to(DEVICE)
                _, outputs = self.model(img)

                # Calculate accuracy
                predicted = (outputs > 0.7).float()  # Convert probabilities to binary predictions
                correct = torch.sum(predicted == labels).item()
                running_correct += correct
                running_total += labels.numel()

        # loss and accuracy for a complete epoch
        epoch_acc = (running_correct / running_total) * 100
        return epoch_acc



    def test(self, dataset, epoch):

        # self.model.load_state_dict(torch.load('saved_model/TOMATO_LEAF_PLANTVILLAGE_EFFICIENTNET_10CLASSES_V1_3_200.pth'))
        running_correct = 0.0
        running_total = 0

        num = random.randint(0, len(dataset)-1)
        self.model.eval()
        # with torch.no_grad():
        for i, (img, labels) in tqdm(enumerate(dataset), total=len(dataset)):
            img, labels = img.to(DEVICE), labels.to(DEVICE)
            _, outputs = self.model(img)

            # Calculate accuracy
            predicted = (outputs > 0.7).float()  # Convert probabilities to binary predictions
            correct = torch.sum(predicted == labels).item()
            running_correct += correct
            running_total += labels.numel()
            
            # if i == num:
            #     try:
            #         os.makedirs(f"{large_file_dir}saved_samples/{MODEL_NAME}", exist_ok=True)
            #     except:
            #         pass
            #     # sample = random.randint(0, BATCH_SIZE//2)
            #     image = img[0, :, :, :].cpu().numpy().transpose((1, 2, 0))
            #     image = (image * 255).astype('uint8')
            #     image = Image.fromarray(image)
            #     draw = ImageDraw.Draw(image)




            #     #debug this part
            #     real_label = self.classes[labels[0].argmax().item()]
            #     pred_label = self.classes[outputs[0].argmax().item()]
            #     draw.text((image.width - 200, 0), f"Real: {real_label}", fill='red')
            #     draw.text((image.width - 200, 20), f"Predicted: {pred_label}", fill='blue')
            #     image.save(f"{large_file_dir}saved_samples/{MODEL_NAME}/{epoch}.jpg")

        # loss and accuracy for a complete epoch
        epoch_acc = (running_correct / running_total) * 100
    
        return epoch_acc



 
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
        binaryCrossEntropyLoss = nn.BCEWithLogitsLoss(weight=weight_tensor)
        # bceLoss = nn.BCELoss()
        # mseloss = nn.MSELoss()


        print(f"Beginning to train...")


        # mseloss = nn.MSELoss()
        train_loss_epochs, val_acc_epochs, test_acc_epochs = [], [], []
        writer = SummaryWriter(f'runs/{MODEL_NAME}/')
        os.makedirs("checkpoints/", exist_ok=True)
        os.makedirs(f"{large_file_dir}saved_model/", exist_ok=True)


        for epoch in range(1, epochs+1):

            print(f"Epoch No: {epoch}")
            train_loss, train_acc = self.train(dataset=train_data, loss_func=binaryCrossEntropyLoss, optimizer=optimizer)
            val_acc = self.validate(dataset=val_data)
            train_loss_epochs.append(train_loss)
            val_acc_epochs.append(val_acc)

            if epoch%5==0:
                test_acc = self.test(dataset=test_data, epoch=epoch)
                test_acc_epochs.append(test_acc)
                print(f"Test Accuracy: {test_acc}")
                print("Saving model")
                torch.save(self.model.state_dict(), f"{large_file_dir}saved_model/{MODEL_NAME}_{epoch}.pth")
                print("Model Saved")
                writer.add_scalar("Accuracy/Test", test_acc, epoch)

            print(f"Train Loss:{train_loss}, Train Accuracy:{train_acc}, Validation Accuracy:{val_acc}")

            if max(val_acc_epochs) == val_acc:
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                }, f"checkpoints/{MODEL_NAME}_{epoch}.tar")

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            
    
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



model = Model()
model.fit(250, 1e-6)