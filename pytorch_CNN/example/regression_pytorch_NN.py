import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tables
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
from logger import Logger
import os
import shutil
 
torch.manual_seed(1)
 
EPOCH = 1 
BATCH_SIZE = 130
BATCH_SIZE_test = 100
LR = 0.01 #0.001
DOWNLOAD_MNIST = True


class Net(nn.Module):
    def __init__(self, n_feature,  n_output):
        super(Net, self).__init__()
        # define the style of every layer 
        self.hidden1 = torch.nn.Linear(n_feature, 180)   # define hiden layer, liner out put
#        self.drop1   = torch.nn.Dropout(0.5)
        self.hidden2 = torch.nn.Linear(180, 120)   # define hiden layer, liner out put
        self.hidden3 = torch.nn.Linear(120, 90)   # define hiden layer, liner out put
        self.hidden4 = torch.nn.Linear(90, 60)   # define hiden layer, liner out put
        self.hidden5 = torch.nn.Linear(60, 40)   # define hiden layer, liner out put
        self.hidden6 = torch.nn.Linear(40, 20)   # define hiden layer, liner out put
        self.hidden7 = torch.nn.Linear(20, 14)   # define hiden layer, liner out put
        self.hidden8 = torch.nn.Linear(14, 8)   # define hiden layer, liner out put
        self.predict = torch.nn.Linear(8, n_output)   # define output layer, liner out put

 
    def forward(self, x):
        x = self.hidden1(x)
#        x = self.drop1(x)
        x = F.tanh(x) #sigmoid(x) #softplus(x) #relu(x)
        x = self.hidden2(x)
#        x = self.drop2(x)
        x = F.tanh(x) #sigmoid(x) #softplus(x)  #relu(x)
        x = self.hidden3(x)
        x = F.tanh(x) #sigmoid(x) #softplus(x)  #relu(x)
        x = self.hidden4(x)
        x = F.tanh(x) #sigmoid(x) #softplus(x)  #relu(x)
        x = self.hidden5(x)
        x = F.tanh(x) #sigmoid(x) #softplus(x)  #relu(x)
        x = self.hidden6(x)
        x = F.tanh(x) #sigmoid(x) #softplus(x)  #relu(x)
        x = self.hidden7(x)
        x = F.tanh(x) #sigmoid(x) #softplus(x)  #relu(x)
        x = self.hidden8(x)
        x = F.tanh(x) #sigmoid(x) #softplus(x)  #relu(x)
        x = self.predict(x)
        return x
 

def train(pklfile, trainedh5):

# input dataset from h5, then divide it into train dataset and test dataset(10:1)
        
        x = torch.unsqueeze(torch.linspace(-1, 1, 10000), dim=1).double()  # x data (tensor), shape=(100, 1)
        y = torch.acos(x*0.02) + x.pow(5) - x.pow(4)*2 - x.pow(3)*2 + x.pow(2) + 0.002*torch.rand(x.size()).double()                 # noisy y data (tensor), shape=(100, 1)
        x_1 = torch.unsqueeze(torch.linspace(-1, 1, 300), dim=1).double()  # x data (tensor), shape=(100, 1)
        y_1 = torch.acos(x_1*0.02) + x_1.pow(5) - x_1.pow(4)*2 - x_1.pow(3)*2 + x_1.pow(2) + 0.002*torch.rand(x_1.size()).double()                 # noisy y data (tensor), shape=(100, 1)
#        y = x.pow(3)                  # noisy y data (tensor), shape=(100, 1)

        torch_dataset = Data.TensorDataset(x, y)
        torch_dataset_1 = Data.TensorDataset(x_1, y_1)
        concat_dataset = Data.ConcatDataset((torch_dataset, torch_dataset_1))
        print(concat_dataset.datasets)
        loader = Data.DataLoader(dataset=concat_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)

        x_v = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1).double()     # x test data (tensor), shape=(100, 1)
        y_v = torch.acos(x_v*0.02) + x_v.pow(5)  - x_v.pow(4)*2 - x_v.pow(3)*2 +  x_v.pow(2) + 0.002*torch.rand(x_v.size()).double()                 # noisy y test data (tensor), shape=(100, 1)
        y_cacu = torch.acos(x_v*0.02) + x_v.pow(5)  - x_v.pow(4)*2 - x_v.pow(3)*2 +  x_v.pow(2)                 # noisy y test data (tensor), shape=(100, 1)
#        y_v = x_v.pow(3)                  # noisy y test data (tensor), shape=(100, 1)
        
        net = Net(n_feature=1,  n_output=1)
        net = net.double()
        print(net)
        logdir = './NN_logs_' + h5key
        if os.path.isdir(logdir):
            shutil.rmtree(logdir)
        logger = Logger(logdir)
		
        res = net(x_v)	
        writer = SummaryWriter(logdir)
        writer.add_graph(net, res)
        writer.close()
	
        	
#        optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay=0.1)
#        optimizer = torch.optim.SGD(net.parameters(), lr=LR)
#        optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.0001)
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
#        optimizer = torch.optim.Adagrad(net.parameters(), lr=LR, lr_decay=0.001)	
        loss_func = nn.MSELoss()
        
        plt.ion()
        plt.figure(figsize=(13,3.5))	
        loss_list = []
        loss_list_test = []
        #par_np = net.parameters()
        
        Step = 0 
        lri = LR
        for epoch in range(EPOCH):
            print('Epoch: ', epoch)
            for step, (b_x, b_y) in enumerate(loader):
                print('Step: ', step)
                
                prediction = net(b_x)
                loss = loss_func(prediction, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
	
                Step+=1          
                if (Step+1) % 5 == 0:
                    lri = lri/(1 + 0.001)
                    print("lri:  ",lri)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lri
                    pre_y_v = net(x_v)
                    loss_test = loss_func(pre_y_v, y_v)
                    loss_best = loss_func(y_cacu, y_v)
                    loss_list.append(loss.data[0])
                    loss_list_test.append(loss_test.data[0])
                    plt.subplot(131)
                    plt.cla()
                    plt.scatter(b_x.data.numpy(), b_y.data.numpy())
                    plt.scatter(b_x.data.numpy(), prediction.data.numpy(), s = 2, color = 'red')
                    plt.text(-0.3, 0, 'Loss=%.6f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
                    plt.subplot(132)
                    plt.cla()
                    plt.scatter(x_v.data.numpy(), y_v.data.numpy())
                    plt.plot(x_v.data.numpy(), pre_y_v.data.numpy(), 'r-', lw=2)
                    plt.text(-0.3, 0, 'Loss    =%.6f' % loss_test.data.numpy(), fontdict={'size': 10, 'color':  'red'})
                    plt.text(-0.3, 0.2, 'Loss_b=%.6f' % loss_best.data.numpy(), fontdict={'size': 10, 'color':  'red'})
                    plt.subplot(133)
                    plt.cla()
                    plt.plot(loss_list, 'b-', lw=1, label='train')
                    plt.plot(loss_list_test, 'r-', lw=1, label='test')
                    plt.legend(loc = 'best')
                    plt.pause(0.1)
                    
                      
# ================================================================== #
#                        Tensorboard Logging                         #
# ================================================================== #

                    # 1. Log scalar values (scalar summary)
                    info = { 'loss': loss.item()}
                    
                    for tag, value in info.items():
                        print(tag, value)
                        logger.scalar_summary(tag, value, Step+1)
                    
                    # 2. Log values and gradients of the parameters (histogram summary)
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, value.data.cpu().numpy(), Step+1)
                        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), Step+1)
                    
                    # 3. Log training images (image summary)
                    info = { 'images': x.view(-1, 5, 5)[:10].cpu().numpy() }
                    
                    for tag, images in info.items():
                        logger.image_summary(tag, images, Step+1)

        
        plt.ioff()
        plt.show()
        	
        
	
ID = 'regression' #'2927360606' # '3975284924' 
trainh5file = 'B2APi0selection_' + ID + '_crystal_modified_1.h5'
h5key = 'crystal_'+ ID
trainpklfile =  'NN_train_params_' + ID + '.pkl'
trainedh5 = 'NN_train_test_' + ID +'.h5'

train(trainpklfile, trainedh5)
#draw_result(trainedh5, h5key)
#application(trainh5file, h5key, trainpklfile)

