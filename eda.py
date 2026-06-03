import os
import numpy as np
from PIL import Image

sample_folder = "sample"



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