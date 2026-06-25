import os #to access the filepath of this system
import numpy as np  #to convert the image to array. how does it looks like
from PIL import Image #to load the image

#using the variable below to represent the folder name
sample_folder = "sample" 

#this was done to rummage through the samples and find out their dimensions

# for filename in os.listdir(sample_folder):
#     if filename.endswith(".jpeg"):
#         img = Image.open(os.path.join(sample_folder,filename))
#         img_arr=np.array(img)
#         print(filename, img.size,img_arr.min(),img_arr.max())



import matplotlib.pyplot as plt

img = Image.open(os.path.join(sample_folder, "10_right.jpeg"))
plt.imshow(img)
plt.title("10_right")
plt.show()

#this is for a specific image we wanted to open. although imshow seems to be a new keyword

import pandas as pd
df=pd.read_csv("trainLabels.csv")

print(df['level'].value_counts().sort_index())

"""
running the above lines from 26 to 30:

here value_counts operates on the column and returns a
series where the index is the category number and 
the frequency of each class is mapped to it.

here .sort_index() sorts the series by the keys
"""