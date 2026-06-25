from dataset import DR

# Instantiate the dataset
dataset = DR(csv_file="trainLabels.csv", img_dir="sample")

# Check first sample
img_tensor, label = dataset[0]

print("Image tensor shape:", img_tensor.shape)
print("Label:", label)
print("Data type:", img_tensor.dtype)

"""
test run on 10 images to check if everything is working
fine
"""