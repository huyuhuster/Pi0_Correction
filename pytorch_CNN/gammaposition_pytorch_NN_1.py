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
import os, sys
 
torch.manual_seed(1)
 
EPOCH = 500
BATCH_SIZE = 8192  #32768   #8192 #32768  #16384
BATCH_SIZE_test = 100
LR = 0.001 #0.001


class Net(nn.Module):
    def __init__(self, n_feature,  n_output):
        super(Net, self).__init__()
        # define the style of every layer 
        self.hidden1 = torch.nn.Linear(n_feature, 64)   # define hiden layer, liner out put
#        self.drop1   = torch.nn.Dropout(0.5)
        self.hidden2 = torch.nn.Linear(64, 40)   # define hiden layer, liner out put
        self.hidden3 = torch.nn.Linear(40, 20)   # define hiden layer, liner out put
        self.hidden4 = torch.nn.Linear(20, 14)   # define hiden layer, liner out put
        self.predict = torch.nn.Linear(14, n_output)   # define output layer, liner out put

 
    def forward(self, x):
        x = self.hidden1(x)
#        x = self.drop1(x)
        x = F.tanh(x)  #sigmoid(x) #softplus(x) #relu(x)
        x = self.hidden2(x)
        x = F.tanh(x)  #sigmoid(x) #softplus(x) #relu(x)
        x = self.hidden3(x)
        x = F.tanh(x)  #sigmoid(x) #softplus(x) #relu(x)
        x = self.hidden4(x)
        x = F.tanh(x)  #sigmoid(x) #softplus(x) #relu(x)
        x = self.predict(x)
        return x
 

def train(h5file, h5key, pklfile, trainedh5, trainedlossplot, train_target):

# input dataset from h5, then divide it into train dataset and test dataset(10:1)
#        mydf_readd5 = pd.read_hdf(h5file, h5key, start=0, stop= 19500000)
        reader = pd.read_hdf(h5file, h5key, chunksize=200000)
        train_dataset_list   = [] 
        test_dataset_list   = []
        for mydf_readd5 in reader:
           mydf_train = mydf_readd5.iloc[: int(mydf_readd5.shape[0]*15/16)]
           mydf_test  = mydf_readd5.iloc[int(mydf_readd5.shape[0]*15/16):]
           print(mydf_train.shape)
           

           # train dataset
           train_data_np = mydf_train.iloc[:,4:].replace(np.nan, 0.0).values
           train_data_tensor = torch.from_numpy(train_data_np).double()
           if train_target == 'phi':
               train_labels_np = mydf_train.mcPhi.values.reshape((mydf_train.shape[0],1))
           elif train_target == 'theta':
               train_labels_np = mydf_train.mcTheta.values.reshape((mydf_train.shape[0],1))
           else:
               print("Wrong train target!")
           
           train_labels_tensor = torch.from_numpy(train_labels_np).double()
           train_dataset_list.append(Data.TensorDataset(train_data_tensor, train_labels_tensor))
           
           #test dataset	
           test_data_np = mydf_test.iloc[:,4:].replace(np.nan, 0.0).values
           test_data_tensor = torch.from_numpy(test_data_np).double()
           
           if train_target == 'phi':
               test_labels_np = mydf_test.mcPhi.values.reshape((mydf_test.shape[0],1))
               test_rec_np = mydf_test.phi.values.reshape((mydf_test.shape[0],1))	
           elif train_target == 'theta':
               test_labels_np = mydf_test.mcTheta.values.reshape((mydf_test.shape[0],1))
               test_rec_np = mydf_test.theta.values.reshape((mydf_test.shape[0],1))	
           else:
               print("Wrong train target!")
           
           test_labels_tensor = torch.from_numpy(test_labels_np).double()
           test_rec_tensor = torch.from_numpy(test_rec_np).double()	
           test_dataset_list.append(Data.TensorDataset(test_data_tensor, test_labels_tensor))
        
        train_dataset = Data.ConcatDataset(train_dataset_list)
        test_dataset = Data.ConcatDataset(test_dataset_list)
        train_loader = Data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
        test_loader   = Data.DataLoader(test_dataset, batch_size = BATCH_SIZE_test )

        print("Let's use", torch.cuda.device_count(), "GPUs!")
        
        net = Net(n_feature=75,  n_output=1)
#        pklfile6 = 'train6/NN_train_params_3975284924_2.pkl'	    
#        net.load_state_dict(torch.load(pklfile6))
        net.cuda()
        net = net.double()
        print(net)
        logger = Logger('./NN_logs_' + h5key)
        	
#        optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay=0.01,momentum=0.9)
#        optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.5)
#        optimizer = torch.optim.Adagrad(net.parameters(), lr=LR, lr_decay=0.01)
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
#        optimizer = torch.optim.RMSprop(net.parameters(), lr=LR, weight_decay=5e-2)
        loss_func = nn.MSELoss()
        
        plt.ion()
        plt.figure(figsize=(10,3))	
        loss_list = []
        loss_list_test = []
        #par_np = net.parameters()
        
        Step = 0 
        lri = LR
        for epoch in range(EPOCH):
            print(epoch,"  th round:")
            for step, data in enumerate(train_loader):
                b_X, b_Y = data
                b_x = b_X.cuda()
                b_y = b_Y.cuda()	       
        #        b_x, b_y = data
         
              
        #L2 regularization        
                reg_lambda = torch.tensor(0.2)
                l2_reg = torch.tensor(0.)
                for param in net.parameters(): 
                    l2_reg += param.cpu().float().norm(2)
                
                prediction = net(b_x).cuda()
                loss = loss_func(prediction, b_y)
        #       loss +=  (reg_lambda*l2_reg).cuda().double()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                Step+=1      

                if (Step+1) % 100 == 0:
                    lri = lri/(1 + 0.005)
                    print("lri:  ",lri)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lri
                    test_output = net(test_data_tensor.cuda())
                    test_pred_y = test_output.cpu().data.numpy()
        #           test_pred_y = test_output.data.numpy()
                    accuracy_test = sum(test_pred_y - test_labels_np)
                    loss_test = loss_func(test_output, test_labels_tensor.cuda())
#                    loss_rec = loss_func(test_rec_tensor.cuda(), test_labels_tensor.cuda())
                    print('Epoch:', epoch, '|step:', Step,
                          '|train loss:%.8f'%loss.data[0], '|test loss:%.8f'%loss_test.data[0])
                    loss_list.append(loss.data[0])
                    loss_list_test.append(loss_test.data[0])
                    
                    plt.subplot(131)
                    plt.cla()
                    plt.plot(loss_list, 'b-', lw=1, label='train')
                    plt.plot(loss_list_test, 'r-', lw=3, label='test')
                    plt.xlabel('step')
                    plt.ylabel('loss')
                    plt.text(10, 0.027, 'Loss_train=%.8f' % loss.data[0], fontdict={'size': 10, 'color':  'blue'})
                    plt.text(10, 0.025, 'Loss_test=%.8f' % loss_test.data[0], fontdict={'size': 10, 'color':  'red'})
#                    plt.text(10, 0.023, 'Loss_rec=%.8f' % loss_rec.data[0], fontdict={'size': 10, 'color':  'red'})
                    legend = plt.legend(loc="best")#(loc="best")
                    frame = legend.get_frame()
                    frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑

                    if train_target == 'phi':
                        Range = [-3.2, 3.2]
                    elif train_target == 'theta':
                        Range = [0.4, 2.4]

                    plt.subplot(133)
                    plt.cla() 
                    plt.hist(test_labels_np, bins=200,range=Range, color='red',alpha=0.7, fill=False,histtype='step', label='test_truth') 
                    plt.hist(test_pred_y,    bins=200,range=Range, color='blue',alpha=0.7, fill=False,histtype='step', label='test_pre') 
                    plt.hist(test_rec_np,    bins=200,range=Range, color='green',alpha=0.7, fill=False,histtype='step', label='test_rec') 
                    plt.xlabel(r'$\phi$')
                    legend = plt.legend(loc="best")#(loc="best")
                    frame = legend.get_frame()
                    frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑
                    
                    plt.subplot(132)
                    plt.cla() 
                    plt.hist(b_y.cpu().data.numpy(),        bins=200,range=Range, color='red',alpha=0.7, fill=False,histtype='step', label='train_truth') 
                    plt.hist(prediction.cpu().data.numpy(), bins=200,range=Range, color='blue',alpha=0.7, fill=False,histtype='step', label='train_pre') 
                    plt.xlabel(r'$\phi$')
                    legend = plt.legend(loc="best")#(loc="best")
                    frame = legend.get_frame()
                    frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑
                    plt.pause(0.1)
                      
                    # ================================================================== #
                    #                        Tensorboard Logging                         #
                    # ================================================================== #

                    # 1. Log scalar values (scalar summary)
                    info = { 'loss': loss.item(), 'accuracy': accuracy_test.item() }
                    
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, Step+1)
                    
                    # 2. Log values and gradients of the parameters (histogram summary)
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, value.data.cpu().numpy(), Step+1)
                        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), Step+1)
                    
                    # 3. Log training images (image summary)
                    info = { 'images': b_x.view(-1, 5, 5)[:10].cpu().numpy() }
                    
                    for tag, images in info.items():
                        logger.image_summary(tag, images, Step+1)


        
        plt.ioff()
        plt.savefig(trainedlossplot,dpi=300)
        plt.show()
        	
        
        test_output = net(test_data_tensor[:10].cuda())
        test_pred_y = test_output.cpu().data.numpy()
        #test_pred_y = test_output.data.numpy()
        print('prediction number:  ', test_pred_y )
        print( 'real number:  ', test_labels_np[:10])
        
#       The model after train        
        for name, param in net.state_dict().items():
            print(name, param.size())


        # save the whole model
        #torch.save(model_object, 'model.pkl')
        # only save the parameters ((recommended))
        torch.save(net.state_dict(), pklfile)

        test_pred_y = np.empty((0,1))
        for step, data in enumerate(test_loader): 	
            t_X,  t_Y = data
            t_x = t_X.cuda()
            t_y = t_Y.cuda()
#            print(t_x)
            test_output = net(t_x).cuda()
#            print(test_output)
            test_pred_y = np.vstack([test_pred_y, test_output.cpu().data.numpy()])
        #            print("test_pred_y shapes:  ", test_pred_y.shape)
        
        #        test_pred_y = np.delete(test_pred_y, 0, 0)
        print("shapes:  ", test_pred_y.shape)
        pred_df = pd.DataFrame(mydf_test[['mcPhi','phi', 'mcTheta', 'theta']])
        print("shapes:  ", test_pred_y.shape, pred_df.shape)
        if train_target == 'phi':
           pred_df['prePhi'] = test_pred_y
        elif train_target == 'theta':
           pred_df['preTheta'] = test_pred_y
        pred_df.to_hdf(trainedh5, key=h5key, mode='w')

        
def application(h5file, testh5, h5key, pklfile):	
	reader = pd.read_hdf(h5file, h5key, chunksize=200000)
#	mydf_test = pd.read_hdf(h5file, h5key, start =0, stop = 2000000)
#        mydf_readd5 = pd.concat([mydf_readd5_1, mydf_readd5_2])
#	chunks = []
	if os.path.exists(testh5):
                os.remove(testh5)
	else:
                print("The file",  testh5, "does not exist")
	print('Begain test....')
	for mydf_test in reader:
#		chunks.append(mydf_readd5)
#		mydf_test = pd.concat(chunks)
#		print(mydf_test)
		
	#	mydf_readd5 = pd.read_hdf(h5file, h5key, start =0, stop = 10000000)
	#	mydf_test  = mydf_readd5.iloc[int(mydf_readd5.shape[0]*3/4):]
		
		#test dataset	
		test_data_np = mydf_test.iloc[:,4:].replace(np.nan, 0.0).values
		test_labels_phi_np = mydf_test.mcPhi.values.reshape((mydf_test.shape[0],1))
		test_rec_phi_np = mydf_test.phi.values.reshape((mydf_test.shape[0],1))
		test_labels_theta_np = mydf_test.mcTheta.values.reshape((mydf_test.shape[0],1))
		
		test_data_tensor = torch.from_numpy(test_data_np).double()
		test_labels_phi_tensor = torch.from_numpy(test_labels_phi_np).double()
		
#		test_phi_dataset   = Data.TensorDataset(test_data_tensor, test_labels_phi_tensor)
		
#		test_phi_loader   = Data.DataLoader(test_phi_dataset, batch_size=BATCH_SIZE_test )
	    
	# ***** load the model
		net = Net(n_feature=75,  n_output=1)
		net.load_state_dict(torch.load(pklfile, map_location='cpu'))
		net = net.double()
	
#		for name, param in net.state_dict().items():
#		    print(name, param.size())
		
		print('test calculat....')
		test_output = net(test_data_tensor)
		print('test calculat successfully!')
		test_pred_y = test_output.data.numpy()
#		test_pred_y = np.empty((0,1))
#		for step, data in enumerate(test_phi_loader): 	
#		    t_x,  t_y = data
#		    test_output = net(t_x)
#		    test_pred_y = np.vstack([test_pred_y, test_output.data.numpy()])
		
		pred_df = pd.DataFrame(mydf_test[['mcPhi','phi', 'mcTheta', 'theta']])
		pred_df['prePhi'] = test_pred_y
		pred_df.to_hdf(testh5, key=h5key, append = True, mode='a')
	#	print(pred_df.head())
	print('end of test, begain plot....')
	draw_result(testh5,h5key)


#def draw_result(trainedh5, h5key, trained_dist, trained_dist_res):
def draw_result(trainedh5, h5key):
#       load the h5 file	
        trained_df = pd.read_hdf(trainedh5, h5key)
#        trained_df = trainedh5
        trained_df['res_rec'] = trained_df.phi-trained_df.mcPhi
        trained_df['res_pre'] = trained_df.prePhi-trained_df.mcPhi
        print(trained_df[0:20])
#        fig1 = plt.figure() 
#        plt.style.use('ggplot')
        Phi1 = 1.44836 
        Phi2 = 1.47945
        Theta1 =  0.833649
        Theta2 =  0.864727
        Phi_c = 1.46411
        Theta_c = 2.08468

        trained_df[['prePhi','phi','mcPhi']][(trained_df.mcPhi>=Phi1)&(trained_df.mcPhi<=Phi2)&(trained_df.mcTheta>=Theta1)&(trained_df.mcTheta<=Theta2)].plot.hist(bins=50, range=[Phi1*0.99,Phi2*1.01], alpha=1.0, linewidth=1, fill=False, histtype='step')
        plt.plot([Phi1, Phi1],[0, 100], 'k--')
        plt.text(Phi1, 0, 'A', fontdict={'size': 10, 'color':  'blue'})
        plt.plot([Phi2, Phi2],[0, 100], 'k--')
        plt.text(Phi2, 0, 'B', fontdict={'size': 10, 'color':  'blue'})
        plt.plot([Phi_c, Phi_c],[0, 100], 'k--')
        plt.text(Phi_c, 0, 'C', fontdict={'size': 10, 'color':  'blue'})
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'')
        legend = plt.legend(loc="best")#(loc="best")
        frame = legend.get_frame()
        frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑
        plt.title('')	
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_dist_c, dpi=300)

        trained_df_sub = trained_df[['prePhi','phi', 'mcPhi']][(trained_df.mcPhi>=Phi1)&(trained_df.mcPhi<=Phi2)&(trained_df.mcTheta>=Theta1)&(trained_df.mcTheta<=Theta2)]
        fig1 = plt.figure() 
        plt.hist2d(trained_df_sub['phi'],trained_df_sub['mcPhi'],(50,50), range=[[Phi1*0.99,Phi2*1.01],[Phi1*0.99,Phi2*1.01]],cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot([Phi1, Phi1],[Phi2, Phi2], 'k--')
        plt.xlabel(r'$\phi_{rec}$')
        plt.ylabel(r'$\phi_{Truth}$')
        frame = legend.get_frame()
        frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑
        plt.title('')	
        plt.grid(True, alpha=0.8)
#        plt.savefig(trained_dist_c, dpi=300)

        fig2 = plt.figure() 
        plt.hist2d(trained_df_sub['prePhi'],trained_df_sub['mcPhi'],(50,50), range=[[Phi1*0.99,Phi2*1.01],[Phi1*0.99,Phi2*1.01]],cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot([Phi1, Phi1],[Phi2, Phi2], 'k--')
        plt.xlabel(r'$\phi_{pre}$')
        plt.ylabel(r'$\phi_{Truth}$')
        frame = legend.get_frame()
        frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑
        plt.title('')	
        plt.grid(True, alpha=0.8)
#        plt.savefig(trained_dist_c, dpi=300)

        trained_df[['prePhi','phi','mcPhi']].plot.hist(bins=288, range=[-3.2, 3.2], alpha=1., linewidth=0.6, fill=False, histtype='step')
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'')
        legend = plt.legend(loc="best")#(loc="best")
        frame = legend.get_frame()
        frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑
        plt.title('')	
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_dist, dpi=300)

        trained_df[['res_rec', 'res_pre']].plot.hist(bins=200, range=[-0.05, 0.05], alpha=1.0, fill=False, histtype='step')
        plt.xlabel(r'$\phi$- $\phi_{truth}$')
        plt.ylabel(r'')
        plt.legend(loc = 'best')
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_dist_res, dpi=300)

        trained_df[['res_rec', 'res_pre']][(trained_df.mcPhi>=Phi1)&(trained_df.mcPhi<=Phi2)&(trained_df.mcTheta>=Theta1)&(trained_df.mcTheta<=Theta2)].plot.hist(bins = 50, range=[-0.05, 0.05], alpha=1.0, fill=False, histtype='step')
        plt.xlabel(r'$\phi$- $\phi_{truth}$')
        plt.ylabel(r'')
        plt.legend(loc = 'best')
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_dist_res, dpi=300)
        
        plt.show()

train_target = 'theta'
Dir = 'train8/'  # + time.strftime( "%Y%m%d%H%M%S",time.localtime())
if not os.path.isdir(Dir):
   os.mkdir(Dir)
   if not os.path.isdir(Dir+ train_target + '/'):
      os.mkdir(Dir+ train_target + '/')
ID = '7151069798' #'2927360606' # '3975284924' 
trainh5file = 'B2APi0selection_' + ID + '_crystal_addmcMatchWeight_modified.h5'

h5key = 'crystal_'+ ID

trainpklfile = Dir + 'NN_train_params_' + ID + '.pkl'

trainedh5 = Dir + 'NN_train_test_' + ID +'.h5'
testh5    = Dir + 'NN_test_' + ID +'.h5'

trainedlossplot   = Dir + 'NN_train_test_' + ID + '_' + train_target + 'loss.png'
trained_dist      = Dir + 'NN_train_test_' + ID + '_' + train_target + '.png'
trained_dist_c    = Dir + 'NN_train_test_' + ID + '_' + train_target + '_crystal.png'
trained_dist_res  = Dir + 'NN_train_test_' + ID + '_' + train_target + '_res.png'

train(trainh5file, h5key, trainpklfile, trainedh5, trainedlossplot, train_target)
#draw_result(trainedh5, h5key, trained_dist, trained_dist_res)
#draw_result(testh5, h5key)
#application(trainh5file, testh5, h5key, trainpklfile)

