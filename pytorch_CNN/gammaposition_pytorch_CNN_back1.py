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
from logger import Logger

 
torch.manual_seed(1)
 
EPOCH = 1
BATCH_SIZE = 100
BATCH_SIZE_test = 100
LR = 0.0001
DOWNLOAD_MNIST = True


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # (3,5,5)
                     nn.Conv2d(in_channels=3, out_channels=18, kernel_size=3,
                               stride=1, padding=1), # (18,5,5)
        # want to don't change the size of picture convoluted by con2d, padding=(kernel_size-1)/2
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=3,stride=1) # (18,3,3)
                     )
        self.conv2 = nn.Sequential( # (18,3,3)
                     nn.Conv2d(18, 36, 3, 1, 1), # (36,3,3)
                     nn.ReLU(),
                     nn.MaxPool2d(2,1) # (36,2,2)
		     )
        self.fc1 = nn.Linear(36*2*2,36)
        self.fc2 = nn.Linear(36,24)
        self.out = nn.Linear(24,1)

 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # flat batch 32,7,7 to (batch_size, 32*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.out(x)
        return output
 





def train(h5file, h5key, pklfile, trainedh5):

# input dataset from h5, then divide it into train dataset and test dataset(4:1)
        mydf_readd5 = pd.read_hdf(h5file, h5key)

        mydf_train = mydf_readd5.iloc[ : int(mydf_readd5.shape[0]*9/10)]
        mydf_test  = mydf_readd5.iloc[int(mydf_readd5.shape[0]*9/10) : ]
        print(mydf_train.shape, mydf_test.shape)
        
        # train dataset
        train_data_np = mydf_train.iloc[ : , 4 : ].replace(np.nan, 0.0).values.reshape((mydf_train.shape[0],3,5,5))
        train_labels_phi_np = mydf_train.mcPhi.values.reshape((mydf_train.shape[0],1))
        train_labels_theta_np = mydf_train.mcTheta.values.reshape((mydf_train.shape[0],1))
        
        
        train_data_tensor = torch.from_numpy(train_data_np).double()
        train_labels_phi_tensor = torch.from_numpy(train_labels_phi_np).double()
        train_labels_theta_tensor = torch.from_numpy(train_labels_theta_np).double()
        
        train_phi_dataset   = Data.TensorDataset(train_data_tensor, train_labels_phi_tensor)
        train_theta_dataset = Data.TensorDataset(train_data_tensor, train_labels_theta_tensor)
        
        train_phi_loader = Data.DataLoader(train_phi_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_theta_loader = Data.DataLoader(train_theta_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        #test dataset	
        test_data_np = mydf_test.iloc[:,4:].replace(np.nan, 0.0).values.reshape((mydf_test.shape[0],3,5,5))
        test_labels_phi_np = mydf_test.mcPhi.values.reshape((mydf_test.shape[0],1))
        test_labels_theta_np = mydf_test.mcTheta.values.reshape((mydf_test.shape[0],1))
        
        test_data_tensor = torch.from_numpy(test_data_np).double()
        test_labels_phi_tensor = torch.from_numpy(test_labels_phi_np).double()
        test_labels_theta_tensor = torch.from_numpy(test_labels_theta_np).double()
        
        test_phi_dataset   = Data.TensorDataset(test_data_tensor, test_labels_phi_tensor)
        test_theta_dataset = Data.TensorDataset(test_data_tensor, test_labels_theta_tensor)

        test_phi_loader   = Data.DataLoader(test_phi_dataset, batch_size=BATCH_SIZE_test)
        test_theta_loader = Data.DataLoader(test_theta_dataset, batch_size=BATCH_SIZE_test)
        
         
        cnn = CNN()
        cnn = cnn.cuda()
        cnn = cnn.double()
        
        print(cnn)
        logger = Logger('./CNN_logs_' + h5key)
        
#        optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=5e-1)
        optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, weight_decay=1e-1)
        	
        #loss_function = nn.CrossEntropyLoss()
        loss_func = nn.MSELoss()
         
        plt.ion()
        step_list = []
        loss_list = []
        loss_list_test = []
        
        #for name, param in cnn.named_parameters():
        #	if param.requires_grad: 
        #		print(name, param.data)
        
        for epoch in range(EPOCH):
            for step, data in enumerate(train_phi_loader):
                b_X, b_Y = data
                b_x = b_X.cuda()
                b_y = b_Y.cuda()
        #        b_x, b_y = data
         
                output = cnn(b_x).cuda()
                loss = loss_func(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
         
                if (step+1) % 100 == 0:
#                    np.vstack([par_np, cnn.parameters()])
                    test_output = cnn(test_data_tensor[:20000].cuda())
                    pred_y = test_output.cpu().data.numpy()
                    accuracy = sum(pred_y - test_labels_phi_np[:20000])
                    loss_test = loss_func(test_output, test_labels_phi_tensor[:20000].cuda())
                    print('Epoch:', epoch, '|Step:', step,
                          '|train loss:%.4f'%loss.data[0] )
                    step_list.append(step)	
                    loss_list.append(loss.data[0])	
                    loss_list_test.append(loss_test.data[0])	
                    
                    plt.cla()
                    plt.plot(step_list, loss_list, 'b-', lw=1, label='train')	
                    plt.plot(step_list, loss_list_test, 'r-', lw=3, label='test')	
                    plt.xlabel('step')	
                    plt.ylabel('loss')	
                    plt.text(10000, 2.1, 'Loss_train=%.4f' % loss.cpu().data.numpy(), fontdict={'size': 10, 'color':  'blue'})
                    plt.text(10000, 2.0, 'Loss_test =%.4f' % loss_test.cpu().data.numpy(), fontdict={'size': 10, 'color':  'red'})
                    plt.legend(loc = 'best')	
                    plt.pause(0.1)

# ================================================================== #
#                        Tensorboard Logging                         #
# ================================================================== #

                    # 1. Log scalar values (scalar summary)
                    info = { 'loss': loss.item(), 'accuracy': accuracy.item() }
                    
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, step+1)
                    
                    # 2. Log values and gradients of the parameters (histogram summary)
                    for tag, value in cnn.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
                        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)
                    
                    # 3. Log training images (image summary)
                    info = { 'images': b_x.view(-1, 5, 5)[:10].cpu().numpy() }
                    
                    for tag, images in info.items():
                        logger.image_summary(tag, images, step+1)

        
        plt.ioff()
        plt.show()

# The model after train        
        for name, param in cnn.state_dict().items():
             print(name, param.size())

        print('**save the whole model') 	     
# save the whole model
        #torch.save(model_object, 'model.pkl')
# only save the parameters ((recommended))
        torch.save(cnn.state_dict(), pklfile)
        
        print('**save phi')
        pred_y = np.empty((0,1)) 
        for step, data in enumerate(test_phi_loader):
            t_X,  t_Y = data
            t_x = t_X.cuda()
            test_output = cnn(t_x).cuda()        
            pred_y = np.vstack([pred_y, test_output.cpu().data.numpy()])
            print("pred_y shapes:  ", pred_y.shape)
        
#        pred_y = np.delete(pred_y, 0, 0)
        print("shapes:  ", pred_y.shape)
        pred_df = pd.DataFrame(mydf_test[['mcPhi','phi']])
        print("shapes:  ", pred_y.shape, pred_df.shape)
        pred_df['prePhi'] = pred_y
        pred_df.to_hdf(trainedh5, key=h5key, mode='w')	
        
        
        

def application(h5file, h5key, pklfile):
        # load the model
        #the_model = TheModelClass(*args, **kwargs)
        #model = torch.load('model.pkl')
        
        cnn = CNN 
        cnn.load_state_dict(torch.load(pklfile))

def draw_result(trainedh5, h5key):
#  load the h5 file
        trained_df = pd.read_hdf(trainedh5, h5key)
#        plt.figure()
        trained_df.plot.hist(bins=100, range=[-3.2, 3.2], alpha=0.5, fill=False, histtype='step')
        plt.xlabel(r'$\phi$')
        plt.title('')
        plt.show()
        
ID = '3975284924' #'2927360606'
trainh5file = 'B2APi0selection_' + ID +'_crystal_modified_1.h5'
h5key = 'crystal_' + ID
trainpklfile =  'CNN_train_params_' + ID + '.pkl'
trainedh5 = 'CNN_train_test_' + ID + '.h5'

#train(trainh5file, h5key, trainpklfile, trainedh5)
draw_result(trainedh5, h5key)	
#application(trainh5file, h5key, trainpklfile)
