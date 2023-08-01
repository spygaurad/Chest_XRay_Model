CheXPert is A Large [[Chest Radiography|Chest Radiograph]] [[Dataset]] with Uncertainty Labels and Expert Comparison.

The CheXpert dataset contains 224,316 chest radiographs from 65,240 patients. These radiographs are labeled for the presence of 14 common chest radiographic observations. The 14 common chest observations include:

1. No Finding 
2. Enlarged Cardiomediastinum
3. Cardiomegaly 
4. Lung Opacity 
5. Lung Lesion 
6. Edema
7. Consolidation 
8. Pneumonia
9. Atelectasis 
10. Pneumothorax
11. Pleural Effusion
12. Pleural Other
13. Fracture
14. Support Devices




## Labeler Design

Uncertainty in the dataset refers to the challenge of handling ambiguous or uncertain findings in the radiology reports. The dataset contains 14 observations, including 12 pathologies, "Support Devices," and "No Finding" observations. To arrive at a final label for each observation, the dataset uses aggregation based on mentions in the reports. Here's how uncertainty is handled:

1. Label Assignment:
	- Observations with at least one positively classified mention in the report are assigned a positive (1) label.
	- Observations with no positively classified mentions but at least one uncertain mention are assigned an uncertain (u) label.
	- Observations with at least one negatively classified mention are assigned a negative label.
	- If there is no mention of an observation, it is left blank.

2. "No Finding" Observation:
	- The "No Finding" observation is assigned a positive label (1) if no pathology is classified as positive or uncertain. This indicates a normal or healthy condition.

Positive (1), negative (0), or uncertain (u) labels are considered positive, and blank is considered negative. By handling uncertainty in radiology reports more effectively, the dataset and the new labeling algorithm aim to provide more accurate and reliable data for chest radiograph interpretation.



## Uncertainty Approaches while Training the Model


1. Ignoring (U-Ignore):
	- Approach: Ignore the uncertainty labels during training and optimize the binary cross-entropy loss for positive and negative labels only.
	- Intuition: Similar to complete case deletion in imputation, where cases with missing values are removed. May reduce the effective size of the dataset.
	- Note: This method might ignore a large proportion of labels, especially for observations with prevalent uncertainty labels.

2. Binary Mapping (U-Zeroes and U-Ones models):
	- Approach: Replace uncertain labels (u) with 0 (U-Zeroes) or 1 (U-Ones).
	- Intuition: Similar to zero imputation strategies in statistics, but may distort classifier decision-making and degrade performance if uncertainty labels carry useful information.

3. Self-Training (U-Self Trained):
	- Approach: Treat uncertainty labels as unlabeled examples and perform self-training.
	- Process: Train a model using U-Ignore approach, make predictions to re-label uncertainty labels with model probabilities, and set up loss as the mean of binary cross-entropy losses over the relabeled examples.
	- Note: Inspired by semi-supervised learning, iteratively labels unlabeled examples until convergence.

4. 3-Class Classification (U-MultiClass model):
	- Approach: Treat uncertainty (u) as its own class alongside positive (1) and negative (0) classes for each observation.
	- Hypothesis: By supervising uncertainty, the network can better incorporate information from the image and represent uncertainty for different pathologies.
	- Loss: Mean of multi-class cross-entropy losses over the observations.
	- Test Time: Probability of the positive label is outputted after applying a Soft-max restricted to positive and negative classes.

