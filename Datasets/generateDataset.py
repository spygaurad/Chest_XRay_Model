import os
import random
import csv

root_directory = "/home/optimus/Downloads/Dataset/ChestXRays/NIH/images"
output_directory = "/home/optimus/wiseyak/Chest_XRay_Model/Datasets/autoencoder_pretraining/"

# Get the list of image file names
image_files = os.listdir(root_directory)
random.shuffle(image_files)

# Calculate the number of images for each split
total_images = len(image_files)
train_count = int(total_images * 0.8)
valid_count = int(total_images * 0.1)
test_count = total_images - train_count - valid_count

# Split the image file names into train, valid, and test sets
train_files = image_files[:train_count]
valid_files = image_files[train_count:train_count + valid_count]
test_files = image_files[train_count + valid_count:]

# Function to write file paths to a CSV file
def write_csv(file_paths, csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file_path"])  # Write the header
        for file_path in file_paths:
            writer.writerow([os.path.join(root_directory, file_path)])

# Write the file paths to CSV files
write_csv(train_files, os.path.join(output_directory, "train.csv"))
write_csv(valid_files, os.path.join(output_directory, "valid.csv"))
write_csv(test_files, os.path.join(output_directory, "test.csv"))

print("CSV files generated successfully!")
