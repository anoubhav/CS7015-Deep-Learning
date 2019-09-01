import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import time
import json
from tqdm import tqdm

from itertools import product
from collections import namedtuple
from collections import OrderedDict
from functools import reduce
from datetime import datetime

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

        # with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
        #     json.dump(self.run_data, f, ensure_ascii=False, indent=4)

    @torch.no_grad()
    def _test(self, testloader, model):
        correct = 0
        total = 0
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
    def __init__(self, l1_oc, filter_multiplier, kernel_size, stride, inp_img_shape, dropout_rate = 0):
        super().__init__()
        self.fil_mul = filter_multiplier
        self.l1_oc = l1_oc
        self.kernel_size = kernel_size
        self.num_conv_layers = 2
        self.inp_img_shape = inp_img_shape
        self.stride = stride
        self.dropout_rate = dropout_rate

        # Layers Initialization
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        """Uncomment ONLY for net 1."""
        ## Convolutional Base layers for net 1 ONLY
        # self.conv1 = nn.Conv2d(in_channels=self.inp_img_shape[0], out_channels=self.l1_oc, kernel_size=self.kernel_size, stride=self.stride)
        # self.conv2 = nn.Conv2d(in_channels=self.l1_oc, out_channels=int(self.l1_oc*self.fil_mul), kernel_size=self.kernel_size, stride=self.stride)

        """Uncomment ONLY for net 2 and 3."""

        ## Convolutional Base layers for net 2 and  net 3 ONLY. 
        # self.conv1 = nn.Conv2d(in_channels=self.inp_img_shape[0], out_channels=self.l1_oc, kernel_size=self.kernel_size, stride=self.stride, padding=self.stride)

        # self.conv2 = nn.Conv2d(in_channels=self.l1_oc, out_channels=int(self.l1_oc*self.fil_mul), kernel_size=self.kernel_size, stride=self.stride)

        # self.conv3 = nn.Conv2d(in_channels=int(self.l1_oc*self.fil_mul), out_channels=int(self.l1_oc*(self.fil_mul)**2), kernel_size=self.kernel_size, stride=self.stride, padding=self.stride)

        # self.conv4 = nn.Conv2d(in_channels=int(self.l1_oc*(self.fil_mul)**2), out_channels=int(self.l1_oc*(self.fil_mul)**3), kernel_size=self.kernel_size, stride=self.stride)


        ## Batch Normalization layers ONLY for net 3.
        # self.bn1 = nn.BatchNorm2d(int(self.l1_oc), eps=1e-05, momentum=0.05, affine=True)
        # self.bn2 = nn.BatchNorm2d(int(self.l1_oc*self.fil_mul), eps=1e-05, momentum=0.05, affine=True)
        # self.bn3 = nn.BatchNorm2d(int(self.l1_oc*(self.fil_mul)**2), eps=1e-05, momentum=0.05, affine=True)
        # self.bn4 = nn.BatchNorm2d(int(self.l1_oc*(self.fil_mul)**3), eps=1e-05, momentum=0.05, affine=True)


        """Uncomment ONLY for net 4."""
        ## Convolutional Base layers for net 4 ONLY. 
        self.conv1 = nn.Conv2d(in_channels=self.inp_img_shape[0], out_channels=self.l1_oc, kernel_size=self.kernel_size, stride=self.stride, padding=self.stride)
        self.conv2 = nn.Conv2d(in_channels=self.l1_oc, out_channels=self.l1_oc, kernel_size=self.kernel_size, stride=self.stride, padding=self.stride)

        self.conv3 = nn.Conv2d(in_channels=self.l1_oc, out_channels=int(self.l1_oc*(self.fil_mul)), kernel_size=self.kernel_size, stride=self.stride)

        self.conv4 = nn.Conv2d(in_channels=int(self.l1_oc*(self.fil_mul)), out_channels=int(self.l1_oc*(self.fil_mul)**2), kernel_size=self.kernel_size, stride=self.stride, padding=self.stride)
        self.conv5 = nn.Conv2d(in_channels=int(self.l1_oc*(self.fil_mul)**2), out_channels=int(self.l1_oc*(self.fil_mul)**2), kernel_size=self.kernel_size, stride=self.stride, padding=self.stride)

        self.conv6 = nn.Conv2d(in_channels=int(self.l1_oc*(self.fil_mul)**2), out_channels=int(self.l1_oc*(self.fil_mul)**3), kernel_size=self.kernel_size, stride=self.stride)

        ## Batch Normalization layers ONLY for net 4.
        self.bn1 = nn.BatchNorm2d(int(self.l1_oc), eps=1e-05, momentum=0.05, affine=True)
        self.bn2 = nn.BatchNorm2d(int(self.l1_oc), eps=1e-05, momentum=0.05, affine=True)
        self.bn3 = nn.BatchNorm2d(int(self.l1_oc*self.fil_mul), eps=1e-05, momentum=0.05, affine=True)
        self.bn4 = nn.BatchNorm2d(int(self.l1_oc*(self.fil_mul)**2), eps=1e-05, momentum=0.05, affine=True)
        self.bn5 = nn.BatchNorm2d(int(self.l1_oc*(self.fil_mul)**2), eps=1e-05, momentum=0.05, affine=True)
        self.bn6 = nn.BatchNorm2d(int(self.l1_oc*(self.fil_mul)**3), eps=1e-05, momentum=0.05, affine=True)

        # fl_img_dim is the flattened image dimension. It is determined dynamically using an input tensor of same shape as the transformed input. Required FOR ALL network implementations.
        self.fl_img_dim = list(self.forward(torch.zeros([1, *self.inp_img_shape]), get_shape=True))

        # Fully connected network layers for net 1, 2, 3 and 4. DO NOT COMMENT
        self.fc1 = nn.Linear(in_features=reduce(lambda x, y: x*y, self.fl_img_dim), out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10) 

    def forward(self, x, get_shape = False):

        try:
            """ try-except block ensures the input parameters for the network are valid """
            """ Uncomment ONLY for net 1. """
            ## Conv top 1: conv-pool-conv-pool.  
            # x = self.pool(F.relu(self.conv1(x)))
            # x = self.pool(F.relu(self.conv2(x)))

            """ Uncomment for ONLY net 2. """
            ## Conv top 2: conv(pad)-conv-pool-conv(pad)-conv-pool. Added 2 additional conv layers with 'same' padding. 
            # x = F.relu(self.conv1(x))
            # x = self.pool(F.relu(self.conv2(x)))
            # x = F.relu(self.conv3(x))
            # x = self.pool(F.relu(self.conv4(x)))

            """ Uncomment ONLY for net 3. """
            ## Conv top 3: conv(pad)-bn-conv-pool-conv(pad)-bn-conv-pool. Added batch normalization layers after every convolution operation and before relu. 
            # x = F.relu(self.bn1(self.conv1(x)))
            # x = self.pool(F.relu(self.bn2(self.conv2(x))))
            # x = F.relu(self.bn3(self.conv3(x)))
            # x = self.pool(self.bn4(F.relu(self.conv4(x))))

            """ Uncomment ONLY for net 4. """
            ## Conv top 4: conv(pad)-bn-conv(pad)-bn-conv-pool-conv(pad)-bn-conv(pad)-bn-conv-pool. Added batch normalization layers after every convolution operation and before relu. Adding two more padded conv layers before pooling layers with batch norm preceeding relu. 
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = self.pool(self.bn6(F.relu(self.conv6(x))))

            if get_shape:
                return x.shape

        except Exception as e:   
            print(e)
            print('\n!!!!!!! WRONG/INCOMPATIBLE NETWORK PARAMETERS !!!!!!!\n')
            print('Check csv_results folder for the last saved run\n')
            exit()
        
        """ Uncomment only for net 1 and net 2 """
        ## FCN for net 1 and net 2: flatten-fc1-fc2-fc3. Same as boilerplate code.
        # x = x.view(-1, reduce(lambda x, y: x*y, self.fl_img_dim))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        """  Uncomment ONLY for net 3 and net 4. """
        ## FCN for net 3 and net 4: flatten-DO-fc1-DO-fc2-DO-fc3. 
        x = x.view(-1, reduce(lambda x, y: x*y, self.fl_img_dim))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc3(x)    
        return x

if __name__ == '__main__':
    # Load and Prepare the dataset

    """ Network 1 and 2 have only tensor transformation. Network 3 and 4 have more transformations."""
    
    # transform = transforms.Compose([transforms.ToTensor(),])

    """ Uncomment only for net 3 and net 4 """
    transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
            transforms.ToTensor(),
        ])

    
    trainset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True,
                                            download=False, 
                                            transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=False,
                                        download=False, 
                                        transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=256,
                                            shuffle=False,  
                                            )

    """ Run the model """

    # NUM_EPOCHS = 30 #net1
    # NUM_EPOCHS = 30 #net2
    # NUM_EPOCHS = 60 #net3
    NUM_EPOCHS = 100 #net4

    """ Uncomment the parameters according to the net of choice. """
    # Net 1 params:
    # params = OrderedDict(lr=[0.01], batch_size=[128], momentum=[0.9], l1_oc=[6, 12, 18], filter_multiplier=[1.5, 2], kernel_size=[3, 5], stride=[1, 2])

    # Net 2 params:
    # params = OrderedDict(lr=[0.01], batch_size=[128], momentum=[0.9], l1_oc=[18, 32, 64], filter_multiplier=[2], kernel_size=[3, 5, 7], stride=[1])

    # Net 3 params:
    # params = OrderedDict(lr=[0.01], batch_size=[128], momentum=[0.9], l1_oc=[32, 64, 96], filter_multiplier=[2], kernel_size=[3], stride=[1], dropout_rate=[0.1, 0.3, 0.5])

    ## Net 4 params:  
    params = OrderedDict(
        lr = [.01]
        ,batch_size = [128]
        ,momentum = [0.9]                # ONLY for SGD. Uncomment for other optimizers and modify the code below.
        ,l1_oc = [64]                    # No.of output channels in the first conv layer.
        ,filter_multiplier = [2]         # Multiplicative increase in no. of filters for every conv layer i.e. Width
        ,kernel_size = [3]               # Filter size for each conv layer
        ,stride = [1]                    # stride of each filter
        ,dropout_rate = [0.3, 0.5]       # Dropout probability. (ONLY net 3)
    )

    m = RunManager()
    run_count = 0
    for run in RunBuilder.get_runs(params):

        loader = DataLoader(trainset, batch_size=run.batch_size)

        # input_img_shape = list(trainset[0][0].size())
        input_img_shape =list(next(iter(loader))[0].shape)[1:]

        network = Network(run.l1_oc, run.filter_multiplier, run.kernel_size, run.stride, input_img_shape, run.dropout_rate)
        network.cuda()

        # Same optimizer as boilerplate code. LR was changed from 0.001 to 0.01 due to slow convergence.
        optimizer = optim.SGD(network.parameters(), lr = run.lr, momentum= run.momentum)

        # Introduced LR scheduler for network 4 as we are reaching higher accuracy values.
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=10, threshold=2e-3)

        m.begin_run(run, network, loader)
        for epoch in range(NUM_EPOCHS):
            m.begin_epoch()
            for batch in tqdm(loader):

                images, labels = batch[0], batch[1]
                if torch.cuda.is_available(): images, labels = images.cuda(), labels.cuda()
                
                # Forward pass
                preds = network(images)
                # Loss calculation
                loss = F.cross_entropy(preds, labels)
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

            ## Save model after every epoch
            torch.save(network.state_dict(), './models/model-5'+f'-{run}-'+str(epoch)+'.pth')

        ## Save model after every run
        # torch.save(network.state_dict(), './models/model-5'+f'-{run}-'+'.pth')
        m.end_run()
        run_count += 1

        # Save results of multiple runs to csv file after every X runs.
        X = 1
        if run_count%X == 0:
            m.save(f'./csv_results/{datetime.now().strftime("%d-%m %H-%M")}-results-5 run-{run_count}')
            # overwrites by adding new run results.
            # m.save(f'./csv_results/{datetime.now().strftime("%d-%m %H-%M")}-results-net4')

    # Save the results into csv file after finishing all runs. 
    m.save(f'{datetime.now().strftime("%d-%m %H-%M")}' + '-results-5')



