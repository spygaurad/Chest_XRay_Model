
[[Deep Learning Model]] used is [[CNN]] [[DenseNet121]], trained on [[Stanford CheXPert]]


This Paper uses the setup defined in [[Stanford Model]] except, it doesn't use Ensemble Learning

### Training Setup

- Batch Size : 64
- Pretrained Network : False
- Epochs : 3
- Image Size : 320x320
- Classes: 14
- Learning Rate: 1e-4

### Dataset Preparation
- Uncertainty Policy: U-Ones
- 500 Samples separated for testing, from Training samples


## Training Time
- Loss : BCELoss

Results after 3 epochs on test dataset.

![[Pasted image 20230728132219.png]]

Hasn't calculated the Average AUROC.