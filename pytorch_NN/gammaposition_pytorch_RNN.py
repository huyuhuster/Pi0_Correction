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
 
torch.manual_seed(1)
 
EPOCH = 1
BATCH_SIZE = 20
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 1      # rnn input size
LR = 0.001 #0.001
DOWNLOAD_MNIST = True

# input dataset from h5, then divide it into train dataset and test dataset(4:1)
mydf_readd5 = pd.read_hdf('B2APi0selection_2927360606_crystal_modified.h5','crystal_2927360606')
#print(mydf_readd5.iloc[0:1,54:])
#for column in mydf_readd5.columns:
#	print(column)

mydf_train = mydf_readd5.iloc[: int(mydf_readd5.shape[0]*4/5)]
mydf_test  = mydf_readd5.iloc[int(mydf_readd5.shape[0]*4/5):]

# train dataset
train_data_np = mydf_train.iloc[:,54:].replace(np.nan, 0.0).values
#train_data_np -= np.mean(train_data_np, axis=0)
#train_data_np /= np.std(train_data_np, axis=0)
train_labels_phi_np = mydf_train.mcPhi.values.reshape((mydf_train.shape[0],1))
train_labels_theta_np = mydf_train.mcTheta.values.reshape((mydf_train.shape[0],1))

#print(train_data_np,         train_data_np.shape,'\n')
#print(train_labels_phi_np,   train_labels_phi_np.shape,'\n' )
#print(train_labels_theta_np, train_labels_theta_np.shape, '\n')


train_data_tensor = torch.from_numpy(train_data_np).double()
train_labels_phi_tensor = torch.from_numpy(train_labels_phi_np).double()
train_labels_theta_tensor = torch.from_numpy(train_labels_theta_np).double()

#print(train_labels_phi_tensor)

train_phi_dataset   = Data.TensorDataset(train_data_tensor, train_labels_phi_tensor)
train_theta_dataset = Data.TensorDataset(train_data_tensor, train_labels_theta_tensor)

#print(train_phi_dataset[0])
#print(train_phi_dataset[1])

 
# The format of the dataset get via torchvision.datasets cab directly put in DataLoader
train_phi_loader = Data.DataLoader(dataset=train_phi_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_theta_loader = Data.DataLoader(dataset=train_theta_dataset, batch_size=BATCH_SIZE, shuffle=True)

#test dataset	
test_data_np = mydf_test.iloc[:,54:].replace(np.nan, 0.0).values
test_data_np -= np.mean(test_data_np, axis=0)
test_data_np /= np.std(test_data_np, axis=0)
test_labels_phi_np = mydf_test.mcPhi.values.reshape((mydf_test.shape[0],1))
test_labels_theta_np = mydf_test.mcTheta.values.reshape((mydf_test.shape[0],1))


test_data_tensor = torch.from_numpy(test_data_np).cuda()
test_labels_phi_tensor = torch.from_numpy(test_labels_phi_np).cuda()
test_labels_theta_tensor = torch.from_numpy(test_labels_theta_np).cuda()

print("Let's use", torch.cuda.device_count(), "GPUs!")

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)
 
    def forward(self, x, h_state):
            # x (batch, time_step, input_size)
            # h_state (n_layers, batch, hidden_size)
            # r_out (batch, time_step, hidden_size)
            r_out, h_state = self.rnn(x, h_state)

            outs = []    # save all predictions
            for time_step in range(r_out.size(1)):    # calculate output for each time step
                outs.append(self.out(r_out[:, time_step, :]))
            return torch.stack(outs, dim=1), h_state

            # instead, for simplicity, you can replace above codes by follows
            # r_out = r_out.view(-1, 32)
            # outs = self.out(r_out)
            # return outs, h_state
 
rnn = RNN()
print(rnn)
	
optimizer = torch.optim.SGD(rnn.parameters(), lr=LR, weight_decay=0.1)
#optimizer = torch.optim.RMSprop(rnn.parameters(), lr=LR, weight_decay=5e-5)
loss_func = nn.MSELoss()
 
h_state = None      # for initial hidden state

plt.ion()
step_list = []
loss_list = []

for epoch in range(EPOCH):
    for step, data in enumerate(train_phi_loader):
        b_X, b_Y = data
        b_x = b_X.cuda()
        b_y = b_Y.cuda()	       
#        b_x, b_y = data
 
        prediction = rnn(b_x, h_state).cuda()
        h_state = h_state.data

#L2 regularization        
        reg_lambda = torch.tensor(0.2)
        l2_reg = torch.tensor(0.)
        for param in rnn.parameters(): 
            l2_reg += param.cpu().float().norm(2)
        
        loss = loss_func(prediction, b_y)
#        loss +=  (reg_lambda*l2_reg).cuda().double()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if step % 100 == 0:
            test_output = rnn(test_data_tensor)
            pred_y = test_output.cpu().data.numpy()
#           pred_y = test_output.data.numpy()
            accuracy = sum(pred_y - test_labels_phi_np)
            print('Epoch:', epoch, '|Step:', step,
                  '|train loss:%.4f'%loss.data[0])
            step_list.append(step)
            loss_list.append(loss.data[0])
            plt.cla()
            plt.plot(step_list, loss_list, 'r-', lw=5)	
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.text(1000, 1.5, 'Loss=%.4f' % loss.cpu().data.numpy(), fontdict={'size': 20, 'color':  'blue'})
#           plt.text(1000, 1.5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'blue'})
            plt.pause(0.1)

plt.ioff()
plt.show()
	
test_output = rnn(test_data_tensor[:10])
pred_y = test_output.cpu().data.numpy()
#pred_y = test_output.data.numpy()
print(pred_y, 'prediction number')
print(test_labels_phi_np[:10], 'real number')

