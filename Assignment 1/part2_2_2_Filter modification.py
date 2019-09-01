
# In[258]:Import libraries. Define utility functions.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import itertools
from torch.utils.data import DataLoader
from IPython.display import display, clear_output
import pandas as pd

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the network architecture from part1 code
from part1 import Network

# User defined; location of model weights.
MODEL_FILE = './model-4-Run(lr=0.01, batch_size=128, momentum=0.9, l1_oc=64, filter_multiplier=2, kernel_size=3, stride=1, dropout_rate=0.3)-.pth'

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class_map = dict(zip(list(range(10)), classes))

def classwise_test(testloader, model):
    """ Returns class wise accuracy for the 10 classes """
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        
    return 100*[class_correct[i] / class_total[i] for i in range(10)]

def get_all_preds(model, loader):
    """ Returns ALL predicted labels """
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()
        preds = model(images)
        preds = preds.cpu().detach()
        all_preds = torch.cat((all_preds, preds), dim = 0)
    return all_preds

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix difference (before - after)', cmap=plt.cm.Blues, fname='confusion_matrix.png'):
    """ Plots a confusion matrix """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(14, 12))
    plt.grid(b=None)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(shrink = 0.95)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=20)
    plt.yticks(tick_marks, classes, size = 16)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", size=17)
    
    plt.tight_layout()
    plt.ylabel('True label', size = 22)
    plt.xlabel('Predicted label', size =22)
    plt.grid(b=None)
    plt.savefig(fname, bbox_inches = 'tight')


# In[259]: Define the test loader. Load the model parameters.

# Prepare and load the test set
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                        shuffle=False, num_workers=2)

# Define the model parameters (best run of net 4)                       
model = Network(64, 2, 3, 1, (3, 32, 32), 0.3)
model = model.eval().cuda()

# Load the model parameters
checkpoint = torch.load(MODEL_FILE)
model.load_state_dict(checkpoint)

# In[261]: Get predicted labels. Plot confusion matrix.
test_preds_before = get_all_preds(model, testloader)

print('Accuracy of the network on the 10000 test images:',      (testset.targets == test_preds_before.argmax(dim = 1).numpy()).sum()/len(testset.targets))

cmt_before = confusion_matrix(testset.targets, test_preds_before.argmax(dim = 1))
plot_confusion_matrix(cmt_before, classes, normalize = False, fname = 'CF_before.png')


# In[237]: Classwise accuracy before setting filter weights to zero.


before_classwise = classwise_test(testloader, model)


# In[262]: Setting 10 filter weights to 0

model.conv1.weight[0] = 0
model.conv1.weight[1] = 0
model.conv2.weight[0] = 0
model.conv2.weight[1] = 0
model.conv3.weight[0] = 0
model.conv3.weight[1] = 0
model.conv4.weight[0] = 0
model.conv4.weight[1] = 0
model.conv5.weight[0] = 0
model.conv5.weight[1] = 0

# In[263]: Get predictions after setting filters to 0. Get class-wise accuracy

test_preds_after = get_all_preds(model, testloader)
after_classwise = classwise_test(testloader, model)

# In[241]: Plot confusion matrix after setting filters to 0.

print('Accuracy of the network on the 10000 test images(after setting to 0):',      (testset.targets == test_preds_after.argmax(dim = 1).numpy()).sum()/len(testset.targets))

cmt_after = confusion_matrix(testset.targets, test_preds_after.argmax(dim = 1))
plot_confusion_matrix(cmt_after, classes, normalize = False, fname = 'CF_after.png')

# In[242]: Plot the confusion matrix representing the difference between before and after setting filter weights to 0.

plot_confusion_matrix(cmt_before-cmt_after, classes, normalize = False, fname = 'CF_before-after.png')


# In[265]: Get first 256 images from testloader and its labels. Find the images which are now missclassified after setting weights to 0.

mask = ((testset.targets == test_preds_before.argmax(dim=1).numpy())&(test_preds_before.argmax(dim=1).numpy()!=test_preds_after.argmax(dim=1).numpy()))[:256]

data = next(iter(testloader))
images, labels = data
data = next(iter(testloader))

images = torch.cat((images, data[0]), dim = 0)
labels = torch.cat((labels, data[1]), dim = 0)
print(images.shape, labels.shape)

# In[271]: Plot the images from the first 256 in the testloader which were initially classified correctly but are misclassified now.
incorrect_images = images[np.where(mask==True)]


plt.rcParams['figure.facecolor'] = 'white'
grid = torchvision.utils.make_grid(incorrect_images.cpu().detach(), nrow=10)
plt.figure(figsize=(25, 25))
plt.grid(b = None)
plt.axis('off')
plt.rcParams["axes.grid"] = False
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.savefig('missclassified.png', bbox_inches = 'tight')
print([class_map[i.item()] for i in labels[np.where(mask == True)]])
print([class_map[i.item()] for i in test_preds_after.argmax(dim = 1)[np.where(mask == True)]])


#%%
