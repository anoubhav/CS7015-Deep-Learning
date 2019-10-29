import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import time
import json
from tqdm import tqdm
import PIL

import random
import numpy as np
import os
from itertools import product
from collections import namedtuple
from collections import OrderedDict
from functools import reduce
from datetime import datetime

# region RUN manager and network class

class RunManager():
    def __init__(self):

        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(log_dir='./runs/assn1',comment=f'-{run}')

        images, labels = next(iter(self.loader))
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)
    

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
    
    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self, testloader):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["train_accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        results['test_accuracy'] = self._test(testloader, self.network)
        results['total parameters'] = sum(p.numel() for p in self.network.parameters())

        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        print(df.tail(1))

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)
    
    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')

    @torch.no_grad()
    def _test(self, testloader, model):
        correct = 0
        total = 0
        model.eval()
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        return correct/total

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

class Network(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained[0])
        # self.model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 10))
        self.model.fc = nn.Sequential(nn.Linear(512, 10))

    def forward(self, x):
        x = self.model(x)
        return x# endregion
# endregion
def sampleFromClass(ds, k):
    """ Creates Tiny-CIFAR-10 dataset """
    class_counts = {}
    train_data = []
    train_label = []
    for data, label in tqdm(ds):
        label = torch.tensor(label)
        c = label.item()
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= k:
            train_data.append(data)
            train_label.append(c)
    train_data = torch.stack(train_data)
    train_label = torch.tensor(train_label)
    return torch.utils.data.TensorDataset(train_data, train_label)

if __name__ == '__main__':

    # Load and Prepare the dataset    

    ## aug 3
    # train_transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.RandomCrop(224),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=.05,hue=.05),
    #         transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    #         transforms.ToTensor()
    #     ])

    ## aug 2
    # train_transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.RandomCrop(224),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=.05,hue=.05),
    #         transforms.ToTensor()
    #     ])

    ## aug 1
    train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=.05,hue=.05),
            transforms.ToTensor()
        ])



    test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    
    trainset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True,
                                            download=True, 
                                            transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=False,
                                        download=True, 
                                        transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=256,
                                            shuffle=False,)  
                        
    
    trainset = sampleFromClass(trainset, 500)
    """ Run the model """
    NUM_EPOCHS = 25 # Vary based on the experiment

    """ Uncomment the parameters according to the experiment. """

    ## For 32x32
    # params = OrderedDict(
    #     lr = [.01]
    #     ,batch_size = [32]
    #     ,momentum = [0.9]
    #     ,weight_decay = [0.01]
    #     ,pretrained=[[False]]
    # )

    ## For 224x224
    params = OrderedDict(
        lr = [.01, .001]
        ,batch_size = [64]
        ,momentum = [0.9]           # ONLY for SGD. Uncomment for other optimizers and modify the code below.
        ,weight_decay = [0.005, 0.001,  0.0005]
        ,pretrained=[[False], [True, 'all'], [True, 40]]
    )

    m = RunManager()
    run_count = 0
    for run in RunBuilder.get_runs(params)[28:]:
        if run.lr == 0.01 and run.batch_size == 64 and run.weight_decay == 0.005:
            print('skipped run')
            continue
        if run.lr == 0.01 and run.batch_size == 64 and run.weight_decay == 0.001 and run.pretrained[0] == False:
            print('skipped run')
            continue
        loader = DataLoader(trainset, batch_size=run.batch_size)

        network = Network(run.pretrained)

        if run.pretrained[0] and run.pretrained[1] != 'all':
            for name, param in list(network.named_parameters())[:-run.pretrained[1]]:
                    print(name, ' frozen')
                    param.requires_grad = False

        network.cuda()

        # Same optimizer as boilerplate code. LR was changed from 0.001 to 0.01 due to slow convergence.
        optimizer = optim.SGD(network.parameters(), lr = run.lr, momentum= run.momentum, weight_decay= run.weight_decay)
                
        # Introduced LR scheduler for network 4 as we are reaching higher accuracy values.
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=4, threshold=1e-5, factor=0.2)

        m.begin_run(run, network, loader)
        for epoch in range(NUM_EPOCHS):
            m.begin_epoch()
            for batch in tqdm(loader):

                images, labels = batch[0], batch[1]
                if torch.cuda.is_available(): images, labels = images.cuda(), labels.cuda()
                
                # Forward pass
                preds = network(images)
                # Loss calculation
                loss = F.cross_entropy(preds, labels.squeeze())
                # Setting gradient vector to 0. To prevent gradient accumulation from previous forward passes.
                optimizer.zero_grad()
                # Calculate the gradient vector.
                loss.backward()
                # Update the parameters
                optimizer.step()

                m.track_loss(loss)
                m.track_num_correct(preds, labels)

            m.end_epoch(testloader)

            # for Net 4 we require loss
            loss = m.epoch_loss / len(m.loader.dataset)
            lr_scheduler.step(loss)
            m.save(f'./csv_results/2.1_2.2_data_aug_tiny')

        m.end_run()
        run_count += 1

    # Save the results into csv file after finishing all runs. 
    m.save(f'2.1_2.2_data_aug_tiny')