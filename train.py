import torch
import pandas as pd
from torch.utils.data import DataLoader,random_split

from dataset import DR
from model import model 

import os

from sklearn.metrics import cohen_kappa_score
#this we need for model evaluation
from sklearn.model_selection import train_test_split
#this for stratified split feature

from torch.utils.data import Subset
# we need this to later make the train,cv,test datasets 
# from the stratified dataset 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

"""
this means that we are importing ResNet50 which we have
placed under the variable model from model.py
"""

df=pd.read_csv("trainLabels.csv")

# Drop rows for images that aren't physically present inside your train directory
existing_images = set([f.removesuffix(".jpeg") for f in os.listdir("/workspace/train")])
df = df[df["image"].isin(existing_images)].reset_index(drop=True)

df.to_csv("trainLabels_clean.csv", index=False)

dataset = DR("trainLabels_clean.csv", "/workspace/train")

counts=df['level'].value_counts().sort_index()
#this returns a series object
total=counts.sum()

class_weights=[total/(len(counts)*i) for i in counts]
class_weights=torch.tensor(class_weights,dtype=torch.float32)

class_weights = class_weights.to(device) # Don't forget to push your loss weights too!

criterion=torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer=torch.optim.Adam(model.fc.parameters(),lr=1e-3)

# valid_images=set(df["image"]) #convert to set for O(1) lookup

# image_ids=[f.removesuffix(".jpeg") for f in os.listdir("sample/") if f.removesuffix(".jpeg") in valid_images]

# sample_df=df[df["image"].isin(image_ids)] 

# """
# df["image"].isin(image_ids) gives Boolean Series 
# those that evaluate to true are retained 
# and hence we get the sampled df
# """

# sample_csv=sample_df.to_csv("sample.csv", index=False)

# dataset=DR("sample.csv","sample/")

# train_dataset, val_dataset, test_dataset = random_split(dataset, [6, 1, 3])
"""
this random_split done was only for sample testing

it won't be justified on a class imbalanced dataset. 
to fix that, we use stratified split
"""
#this is for running on the original dataset
# Point directly to your full dataset and its original labels


# Extract labels instantly from the pandas dataframe instead of loading images!
labels = df['level'].tolist()
indices = list(range(len(df)))

sample_files = os.listdir("/workspace/train")[:5]
print("Sample files in train dir:", sample_files)
print("Sample df['image'] values:", df['image'].head().tolist())
print("Total files found:", len(os.listdir("/workspace/train")))

trainval_idx, test_idx = train_test_split(
    indices, test_size=0.3, stratify=labels, random_state=42
)
#this splits train+crossval and testing into 7:3 ratio
#since we wanted 30% testing data

train_idx, val_idx = train_test_split(
    trainval_idx, test_size=0.142857, stratify=[labels[i] for i in trainval_idx], random_state=42
)

#then we do the same approximately to make it 6:1 ratio

train_dataset=Subset(dataset,train_idx)
val_dataset=Subset(dataset,val_idx)
test_dataset=Subset(dataset,test_idx)

# Leverage the multi-core Xeon CPU and fast NVMe drive on your instance
dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

num_epochs=20 #before it was five since we were running on sample

#and batch_size is changed to 64 from 2 since the dataset is massive

"""
what does .train do ?

below is training and cv part
"""
for epoch in range(num_epochs):

  #training mode
  model.train()
  total_loss=0;
  
  for batch_idx,(images,labels) in enumerate(dataloader):

    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    #this resets the gradient to zero so they don't 
    # accumulate
        
    predicted_labels=model(images)
    loss_value=criterion(predicted_labels,labels)

    total_loss+=loss_value.item()

    loss_value.backward()
    #this is supposed to compute the gradients

    optimizer.step()
    #this updates the weights
  
  print(f"epoch no.{epoch}: avg_loss ={(total_loss/len(dataloader)):.4f}")
  
  #evaluation mode
  model.eval()
  totalval_loss=0;

  with torch.no_grad():
    for batch_idx,(images,labels) in enumerate(val_loader):

      images, labels = images.to(device), labels.to(device) 
      
      predicted_labels=model(images)
      loss_value=criterion(predicted_labels,labels)

      totalval_loss+=loss_value.item()
    
  print(f"CV: epoch no.{epoch}: avg_loss ={(totalval_loss/len(val_loader)):.4f}")

  torch.save(model.state_dict(), f"retina_epoch_{epoch}.pth")


"""
testing part
"""

model.eval()

all_preds=[]
all_labels=[]

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        preds = torch.argmax(model(images), dim=1)
        #here argmax returns a tensor [batch_sz]
        # for eg: if batch_sz is 2, then there are two classes. 
        # one predicted class per image

        all_preds.extend(preds.tolist())
        #tolist() convers a tensor to a list
        all_labels.extend(labels.tolist())



kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

print(f"Test QWK: {kappa:.4f}")

"""
what does .eval mean?
"""



