import torch
import pandas as pd
from torch.utils.data import DataLoader,random_split

from dataset import DR
from model import model 

import os

"""
this means that we are importing ResNet50 which we have
placed under the variable model from model.py
"""

df=pd.read_csv("trainLabels.csv")
counts=df['level'].value_counts().sort_index()
#this returns a series object
total=counts.sum()

class_weights=[total/(len(counts)*i) for i in counts]
class_weights=torch.tensor(class_weights,dtype=torch.float32)

criterion=torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer=torch.optim.Adam(model.fc.parameters(),lr=1e-3)

valid_images=set(df["image"]) #convert to set for O(1) lookup

image_ids=[f.removesuffix(".jpeg") for f in os.listdir("sample/") if f.removesuffix(".jpeg") in valid_images]

sample_df=df[df["image"].isin(image_ids)] 

"""
df["image"].isin(image_ids) gives Boolean Series 
those that evaluate to true are retained 
and hence we get the sampled df
"""

sample_csv=sample_df.to_csv("sample.csv", index=False)

dataset=DR("sample.csv","sample/")
train_dataset, val_dataset, test_dataset = random_split(dataset, [6, 1, 3])

dataloader=DataLoader(train_dataset,batch_size=2,shuffle=True)

val_loader=DataLoader(val_dataset,batch_size=2,shuffle=False)

test_loader=DataLoader(test_dataset,batch_size=2,shuffle=False)

num_epochs=5
"""
what does .train do ?

below is training and cv part
"""
for epoch in range(num_epochs):

  #training mode
  model.train()
  total_loss=0;
  
  for batch_idx,(images,labels) in enumerate(dataloader):

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
        preds = torch.argmax(model(images), dim=1)
        #here argmax


"""
what does .eval mean?
"""



