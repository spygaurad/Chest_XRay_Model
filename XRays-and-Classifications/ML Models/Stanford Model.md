
A [[CNN]] based [[Deep Learning Model]], trained on the dataset [[Stanford CheXPert]]

## Training Procedure

1. Architecture Selection:
	- [[DenseNet121]] produced the best results, so it was chosen for all experiments.
2. Image Size:
	- Images are resized to 320 × 320 pixels before being fed into the network.
3. Optimizer:
	- Adam optimizer is used with default β-parameters of β1 = 0.9 and β2 = 0.999.
	- Learning rate is set to 1 × 10^-4 and is fixed throughout the training process.
4. Batch Size:
	- Batches are sampled using a fixed batch size of 16 images.
5. Training Epochs:
	- The model is trained for 3 epochs.
6. Checkpoint Saving:
	- Checkpoints are saved every 4800 iterations during training.



## Results and Analysis

#### Validation AUCs for Different Uncertainty Approaches:
- The U-Ones model outperforms the U-Zeros model significantly on Atelectasis (AUC=0.858 vs. AUC=0.811, p = 0.03).
-  No significant difference is found between the best and worst models for Consolidation, Edema, and Pleural Effusion.


#### Effectiveness of Uncertainty Label Handling:

###### U-Ignore approach: 
It not effective in handling uncertainty in the dataset.
Particularly on Cardiomegaly, ignoring uncertainty leads to poor performance due to difficulty in distinguishing borderline cases.

###### U-MultiClass Model
The U-MultiClass approach performs significantly better on Cardiomegaly, indicating the effectiveness of explicitly supervising the model to distinguish between borderline and non-borderline cases.

###### U-Ones Approach
The U-Ones approach performs the best on Atelectasis and Edema, indicating that uncertain phrases are effectively treated as positive findings, suggesting likely presence.

###### U-Zeros Approach for Consolidation
The U-Zeros approach performs the best on Consolidation, suggesting uncertain phrases are better treated as negative findings.


The AUROC results of different approaches are given below:
![[Pasted image 20230727172438.png]]