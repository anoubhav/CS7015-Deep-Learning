
# In[210]: Load the libraries

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from IPython.display import display, clear_output
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
sns.set()

# In[212]: Create a directory in which results(confidence maps) will be saved.
dirName = 'part2'
 
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory:" , dirName ,  " Created ") 
except FileExistsError:
    print("Directory:" , dirName ,  " already exists")

# In[218]: Enter user defined variables. The location of model weights files and the occlusion window sizes to be tested.
""" Enter the path to MODEL FILE WEIGHTS """

MODEL_FILE = './model-4-Run(lr=0.01, batch_size=128, momentum=0.9, l1_oc=64, filter_multiplier=2, kernel_size=3, stride=1, dropout_rate=0.3)-.pth'
OCCLUSION_WINDOW_SIZES = [3, 5, 7, 9]

# In[214]: Obtain the confidence maps for different occlusion window sizes.

# Load the network architecture from part1 code
""" part1.py should be in same working directory """
from part1 import Network

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class_map = dict(zip(list(range(10)), classes))

# Prepare and load the test set
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=15,
                                        shuffle=True, num_workers=2)

# Define the model parameters (best run of net 4)                       
model = Network(64, 2, 3, 1, (3, 32, 32), 0.3)
model = model.eval().cuda()

# Load the model parameters
checkpoint = torch.load(MODEL_FILE)
model.load_state_dict(checkpoint)

## Run inference of trained model

data = next(iter(testloader))
images, labels = data
images, labels = images.cuda(), labels.cuda() 

# Stores the confidence maps for all windows
all_predictions= dict()

for filter_size in OCCLUSION_WINDOW_SIZES:
    print('Performing inference on occlusion window size:', filter_size)
    
    edge_dist = int((filter_size-1)//2)
    all_pred_for_one_filter = np.zeros(((images.shape[-1]-2*edge_dist), (images.shape[-1]-2*edge_dist), images.shape[0]))
    
    # Loop through pixel locations in the image
    for i in range(edge_dist, images.shape[-1] - edge_dist):
        for j in range(edge_dist, images.shape[-2]- edge_dist):
            occ_imgs = images.clone().detach()

            # Set kxk window to gray
            occ_imgs[:, :, i-edge_dist:i+edge_dist+1, j-edge_dist:j+edge_dist+1] = 128/255

            # Run inference on occluded images
            outputs = model(occ_imgs)

            # Obtain the confidence in prediction of true label
            sm = torch.nn.Softmax()
            probabilities = sm(outputs)

            for img_num in range(0, len(labels)):
                all_pred_for_one_filter[i-edge_dist][j-edge_dist][img_num] = probabilities.data[img_num][labels[img_num]].item()
            
    all_predictions[filter_size] = all_pred_for_one_filter


# In[219]: View the size of the confidence map for each window

for i in OCCLUSION_WINDOW_SIZES:
    print(all_predictions[i].shape)

# In[221]: Plot the images for which occlusion analysis is being performed.

plt.rcParams["axes.grid"] = False
plt.rcParams['figure.facecolor'] = 'white'
grid = torchvision.utils.make_grid(images.cpu().detach(), nrow=10)

plt.figure(figsize=(25, 25))
plt.imshow(np.transpose(grid, (1, 2, 0)))

print([class_map[i.item()] for i in labels])

# In[228]: Plot the batches of 5 images along with their confidence maps.
""" This part of the code is optional. """
fig = plt.figure(figsize=(len(OCCLUSION_WINDOW_SIZES)*6, 24))
rows = 5
cols = len(OCCLUSION_WINDOW_SIZES) + 1
cmap = 'inferno'

for batch in [1, 2, 3]:
    img_num = 0
    for i in range(1, rows*cols+1, cols):
        plt.axis('off')
        fig.add_subplot(rows, cols, i)
        plt.imshow(np.transpose(images.cpu()[img_num+rows*(batch-1)], (1, 2, 0)))
        col_count = 1
        for filter_ind in OCCLUSION_WINDOW_SIZES:
            plt.axis('off')
            img = all_predictions[filter_ind][:,:, img_num+rows*(batch-1)]
            fig.add_subplot(rows, cols, i+col_count)
            ax = sns.heatmap(img, linewidth=0, square='True', vmax = 1, vmin = 0, xticklabels=False                         ,yticklabels=False, cmap=cmap, cbar_kws={'shrink': 0.8})
            col_count += 1
            # Give title to 1st row of heatmap
            if i < 6:
                ax.axes.set_title("{} x {}".format(filter_ind, filter_ind),fontsize=20)

        img_num+=1
        
    plt.axis('off')
    plt.show()
    fig.savefig('./part2/3-batch-{}-{}.png'.format(batch, cmap), bbox_inches = 'tight', facecolor='w')
    plt.close(fig)
    fig = plt.figure(figsize=(32, 24))

#%%
