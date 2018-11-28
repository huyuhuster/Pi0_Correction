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
import os, sys, shutil
import calculateCrystalPhiTheta
from calculateCrystalPhiTheta import getCrystalPhiAndTheta
from matplotlib.ticker import NullFormatter
from matplotlib.colors import Colormap
import matplotlib.colors as colors
import time 

import Draw_result
from Draw_result import draw_result

#torch.manual_seed(1)
 
EPOCH = 100
BATCH_SIZE = 2000  #8192  #32768   #8192 #32768  #16384
BATCH_SIZE_test = 100
LR = 0.0001 #0.00001


class Net(nn.Module):
    def __init__(self, n_feature,  n_output):
        super(Net, self).__init__()
        # ****** define the style of every layer 
        self.hidden1 = torch.nn.Linear(n_feature, 128)   # define hiden layer, liner out put
        # self.drop1   = torch.nn.Dropout(0.5)
        self.hidden2 = torch.nn.Linear(128, 128)   # define hiden layer, liner out put
        # self.hidden3 = torch.nn.Linear(400, 200)   # define hiden layer, liner out put
        # self.hidden4 = torch.nn.Linear(200, 140)   # define hiden layer, liner out put
        self.predict = torch.nn.Linear(128, n_output)   # define output layer, liner out put

 
    def forward(self, x):
        x = self.hidden1(x)
        # x = self.drop1(x)
        x = torch.relu(x)  #tanh(x)  #sigmoid(x) #softplus(x) #relu(x)
        x = self.hidden2(x)
        x = torch.relu(x)  #sigmoid(x) #softplus(x) #relu(x)
        # x = self.hidden3(x)
        # x = torch.relu(x)  #sigmoid(x) #softplus(x) #relu(x)
        # x = self.hidden4(x)
        # x = torch.relu(x)  #sigmoid(x) #softplus(x) #relu(x)
        x = self.predict(x)
        return x
 

def train(h5file, h5key, pklfile, validationh5, trainedlossplot, train_target, train_lossh5, n_cpu):

        # ******* input dataset from h5, then divide it into train dataset and test dataset(16:1)

        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = Net(n_feature=75,  n_output=1)
        # pklfile6 = 'train6/NN_train_params_3975284924_2.pkl'	    
        # net.load_state_dict(torch.load(pklfile6))
        net.cuda()
        net = net.double()
        print(net)

        # optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay=0.01,momentum=0.9)
        # optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.5)
        # optimizer = torch.optim.Adagrad(net.parameters(), lr=LR, lr_decay=0.01)
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        # optimizer = torch.optim.RMSprop(net.parameters(), lr=LR, weight_decay=5e-2)
        loss_func = nn.MSELoss()
              
        train_mode_file = Dir_training + "train_mode.txt"	
        train_mode = open(train_mode_file, "w") 	
        train_mode.write(str(net) + '\n')
        train_mode.write("Activation:  " + "Relu"  + '\n')
        train_mode.write("Optimizer:  " + str(optimizer)  + '\n')
        train_mode.write("EPOCH:  "+ str(EPOCH) + '\n')
        train_mode.write("BATCH_SIZE:  "+ str(BATCH_SIZE) + '\n')
        train_mode.write("Leaning rate:  "+ str(LR) + '\n')
        train_mode.write("Training data set  :  "+ h5file + '\n')
        train_mode.write("Test data size  :  "+ "1000" + '\n')
        train_mode.write("Additional  :  "+ "For crystal 2626. And wide layer." + '\n')
        train_mode.close()		

        logdir = Dir_training + 'NN_logs_' + h5key
        if os.path.isdir(logdir):
            shutil.rmtree(logdir)
        logger = Logger(logdir)

        if os.path.exists(train_lossh5):
                print("The file",  train_lossh5, " exist, will remove it!")
                os.remove(train_lossh5)
        else:
                print("The file",  train_lossh5, "does not exist!")
        	
        loss_list_train = []
        loss_list_test  = []
        step_list = []
        # par_np = net.parameters()
        
        Step = 0 
        lri = LR

        # ****** test dataset	
        mydf_test = pd.read_hdf(h5file, h5key, start=0, stop = 1000) #stop= 400)
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
        test_dataset   = Data.TensorDataset(test_data_tensor, test_labels_tensor)
        test_loader   = Data.DataLoader(test_dataset, batch_size=BATCH_SIZE_test )

        res = test_data_tensor.cuda()
        #res = Variable(torch.rand(75,640))	
        writer = SummaryWriter(logdir)
        writer.add_graph(net, (res, ))
        writer.close()			

        runTimeh5 = Dir_training    + 'NN_train_runTime' + ID_train + '.h5'
        total_start_time = time.clock()			
        for epoch in range(EPOCH):
           epoch_start_time = time.clock()			
           print('EPOCH:  ', epoch)
           loss_df_EPOCH_i = pd.DataFrame(columns =  ['step', 'train', 'test' ])
           reader = pd.read_hdf(h5file, h5key, chunksize=BATCH_SIZE*2, start = 1000, stop = 101000) #, start = 400)
           for mydf_readd5 in  reader: 	

              mydf_train = mydf_readd5
              # mydf_train = mydf_readd5.iloc[: int(mydf_readd5.shape[0]*15/16)]
              # mydf_test  = mydf_readd5.iloc[int(mydf_readd5.shape[0]*15/16):]
              # print(mydf_train.iloc[:,54:].head())
              # print(mydf_test.iloc[:,54:].head())
              # print(mydf_train.shape)
              

              # ****** train dataset
              train_data_np = mydf_train.iloc[:,4:].replace(np.nan, 0.0).values
              train_data_tensor = torch.from_numpy(train_data_np).double()
              if train_target == 'phi':
                  train_labels_np = mydf_train.mcPhi.values.reshape((mydf_train.shape[0],1))
              elif train_target == 'theta':
                  train_labels_np = mydf_train.mcTheta.values.reshape((mydf_train.shape[0],1))
              else:
                  print("Wrong train target!")
              
              train_labels_tensor = torch.from_numpy(train_labels_np).double()
              train_dataset   = Data.TensorDataset(train_data_tensor, train_labels_tensor)
              train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=n_cpu)
              
              
              
              for step, data in enumerate(train_loader):
                  # b_x, b_y = data
                  b_X, b_Y = data
                  b_x = b_X.cuda()
                  b_y = b_Y.cuda()	       
              
                
                  # ****** L2 regularization        
                  reg_lambda = torch.tensor(0.2)
                  l2_reg = torch.tensor(0.)
                  for param in net.parameters(): 
                      l2_reg += param.cpu().float().norm(2)
                  
                  prediction = net(b_x).cuda()
                  loss = loss_func(prediction, b_y)
                  # loss +=  (reg_lambda*l2_reg).cuda().double()
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
                  Step+=1      

                  if (Step+1) % 100 == 0:
                      test_output = net(test_data_tensor.cuda())
                      test_pred_y = test_output.cpu().data.numpy()
                      # test_pred_y = test_output.data.numpy()
                      accuracy_test = sum(test_pred_y - test_labels_np)
                      loss_test = loss_func(test_output, test_labels_tensor.cuda())
                      # loss_rec = loss_func(test_rec_tensor.cuda(), test_labels_tensor.cuda())
                      print('Epoch:', epoch, '|step:', Step,
                            '|train loss:%.8f'%loss.item(), '|test loss:%.8f'%loss_test.item())
                      step_list.append(Step)
                      loss_list_train.append(loss.item())
                      loss_list_test.append(loss_test.item())

                      loss_df = pd.DataFrame.from_dict({'step' : [Step], 'train' : [loss.item()], 'test' : [loss_test.item()]})
                      loss_df.to_hdf(train_lossh5, key=h5key+'step', append = True, mode='a')
                      loss_df_EPOCH_i = pd.DataFrame.from_dict({'epoch' : [epoch], 'train' : [loss.item()], 'test' :  [loss_test.item()]})
                                         

           #lri = lri/(1 + 0.005)
           # print("lri:  ",lri)
           # for param_group in optimizer.param_groups:
           #     param_group['lr'] = lri
           epoch_end_time = time.clock()
           runTime = epoch_end_time - epoch_start_time
           print("Epoch RunTime: ", runTime)
           time_EPOCH_i = pd.DataFrame.from_dict({'epoch' : [epoch], 'time' : [runTime]})
           time_EPOCH_i.to_hdf(runTimeh5, key=h5key+'_CPU_' + str(n_cpu) + 't', append = True, mode='a')
           loss_df_EPOCH_i.to_hdf(train_lossh5, key=h5key+'epoch', append = True, mode='a')
           if (epoch+1) % 50 == 0:		   
              pklfile_epoch = Dir_pkl + 'NN_train_params_epoch' + str(epoch) + '.pkl'
              torch.save(net.state_dict(), pklfile_epoch)

        total_end_time = time.clock()
        total_runTime = total_end_time - total_start_time
        print("Total RunTime: ", total_runTime)

        test_output = net(test_data_tensor[:10].cuda())
        test_pred_y = test_output.cpu().data.numpy()
        print('prediction number:  ', test_pred_y )
        print( 'real number:  ', test_labels_np[:10])
        
        # ****** The model after train        
        for name, param in net.state_dict().items():
            print(name, param.size())


        # ****** save the whole model
        # torch.save(model_object, 'model.pkl')
        # only save the parameters ((recommended))
        torch.save(net.state_dict(), pklfile)

        test_pred_y = np.empty((0,1))
        for step, data in enumerate(test_loader): 	
            t_X,  t_Y = data
            t_x = t_X.cuda()
            t_y = t_Y.cuda()
            test_output = net(t_x).cuda()
            test_pred_y = np.vstack([test_pred_y, test_output.cpu().data.numpy()])
        
        # test_pred_y = np.delete(test_pred_y, 0, 0)
        print("shapes:  ", test_pred_y.shape)
        pred_df = pd.DataFrame(mydf_test[['mcPhi','phi', 'mcTheta', 'theta']])
        print("shapes:  ", test_pred_y.shape, pred_df.shape)
        if train_target == 'phi':
           pred_df['prePhi'] = test_pred_y
        elif train_target == 'theta':
           pred_df['preTheta'] = test_pred_y
        pred_df.to_hdf(validationh5, key=h5key, mode='w')

        
def application(h5file, testh5, h5key_test, pklfile, train_target):
	print(h5file)
	reader = pd.read_hdf(h5file, h5key_test, chunksize=BATCH_SIZE*2, start = 400)
#	reader = pd.read_hdf(h5file, h5key_test, chunksize=1000000,  stop = 2000000)
	if os.path.exists(testh5):
		print("The file",  testh5, " exist,  and deleted!")
		os.remove(testh5)
	else:
		print("The file",  testh5, "does not exist!")
	print('Begain test....')
	for mydf_test in reader:
		# ****** test dataset	
		test_data_np = mydf_test.iloc[:,4:].replace(np.nan, 0.0).values

		if train_target == 'phi':
		    test_labels_np = mydf_test.mcPhi.values.reshape((mydf_test.shape[0],1))
		    test_rec_np = mydf_test.phi.values.reshape((mydf_test.shape[0],1))	
		elif train_target == 'theta':
		    test_labels_np = mydf_test.mcTheta.values.reshape((mydf_test.shape[0],1))
		    test_rec_np = mydf_test.theta.values.reshape((mydf_test.shape[0],1))	
		else:
		    print("Wrong train target!")
		
		test_data_tensor = torch.from_numpy(test_data_np).double()
		test_labels_tensor = torch.from_numpy(test_labels_np).double()
		
	    
	    # ****** load the model
		net = Net(n_feature=75,  n_output=1)
        #pklfile9 = 'Energy1000MeV/train9/theta/train_4055541376/train_pkl_4055541376/NN_train_params_epoch1379.pkl' 
		net.load_state_dict(torch.load(pklfile, map_location='cpu'))
		net = net.double()
		
		print('test calculat....')
		test_output = net(test_data_tensor)
		print('test calculat successfully!')
		test_pred_y = test_output.data.numpy()
		
		pred_df = pd.DataFrame(mydf_test[['mcPhi','phi', 'mcTheta', 'theta']])
		if train_target == 'phi':
		   pred_df['prePhi'] = test_pred_y
		elif train_target == 'theta':
		   pred_df['preTheta'] = test_pred_y
		pred_df.to_hdf(testh5, key=h5key_test, append = True, mode='a')
	
	print('end of test, begain plot....')
    # draw_result(testh5, h5key_test, train_target, ID_crystal, definition)



def draw_threshold(testh5_thresholds, h5key, train_target, ID_crystal, definition):
        # ****** load the h5 file	
        trained_df_0 = pd.read_hdf(testh5_thresholds[0], h5key)
        trained_df_1 = pd.read_hdf(testh5_thresholds[1], h5key)
        trained_df_2 = pd.read_hdf(testh5_thresholds[2], h5key)

        Phi1, Phi2,  Phi_c, Theta1, Theta2, Theta_c = getCrystalPhiAndTheta(ID_crystal, definition)

        if train_target == 'phi':
            trained_df_0['res_rec'] = trained_df_0.phi    - trained_df_0.mcPhi
            trained_df_0['res_pre'] = trained_df_0.prePhi - trained_df_0.mcPhi
            trained_df_1['res_rec'] = trained_df_1.phi    - trained_df_1.mcPhi
            trained_df_1['res_pre'] = trained_df_1.prePhi - trained_df_1.mcPhi
            trained_df_2['res_rec'] = trained_df_2.phi    - trained_df_2.mcPhi
            trained_df_2['res_pre'] = trained_df_2.prePhi - trained_df_2.mcPhi
            mc_col  = 'mcPhi'
            rec_col = 'phi'
            pre_col = 'prePhi'
            angle1 = Phi1
            angle2 = Phi2
            angle_c = Phi_c
            Range_Cry = [Phi1*0.99,Phi2*1.01]
            Range = [-3.2, 3.2]
        elif train_target == 'theta':
            trained_df_0['res_rec'] = trained_df_0.theta    - trained_df_0.mcTheta
            trained_df_0['res_pre'] = trained_df_0.preTheta - trained_df_0.mcTheta
            trained_df_1['res_rec'] = trained_df_1.theta    - trained_df_1.mcTheta
            trained_df_1['res_pre'] = trained_df_1.preTheta - trained_df_1.mcTheta
            trained_df_2['res_rec'] = trained_df_2.theta    - trained_df_2.mcTheta
            trained_df_2['res_pre'] = trained_df_2.preTheta - trained_df_2.mcTheta
            mc_col  = 'mcTheta'
            rec_col = 'theta'
            pre_col = 'preTheta'
            angle1 = Theta1
            angle2 = Theta2
            angle_c = Theta_c
            Range_Cry = [Theta1*0.99,Theta2*1.01]
            Range = [0.4, 2.4]
        else:
            print("Wrong train target!")

        print(trained_df_0[0:10])
        
        trained_df_sub_0 = trained_df_0[(trained_df_0.mcPhi>=Phi1)&(trained_df_0.mcPhi<=Phi2)&(trained_df_0.mcTheta>=Theta1)&(trained_df_0.mcTheta<=Theta2)]
        trained_df_sub_1 = trained_df_1[(trained_df_1.mcPhi>=Phi1)&(trained_df_1.mcPhi<=Phi2)&(trained_df_1.mcTheta>=Theta1)&(trained_df_1.mcTheta<=Theta2)]
        trained_df_sub_2 = trained_df_2[(trained_df_2.mcPhi>=Phi1)&(trained_df_2.mcPhi<=Phi2)&(trained_df_2.mcTheta>=Theta1)&(trained_df_2.mcTheta<=Theta2)]

        plt.hist(trained_df_sub_0.res_pre, bins=50, range=[-0.03, 0.03], alpha=1.0, linewidth=1, fill=False, histtype='step', label = 'Threshold 0.0 MeV')
        plt.hist(trained_df_sub_1.res_pre, bins=50, range=[-0.03, 0.03], alpha=1.0, linewidth=1, fill=False, histtype='step', label = 'Threshold 1.0 MeV')
        plt.hist(trained_df_sub_2.res_pre, bins=50, range=[-0.03, 0.03], alpha=1.0, linewidth=1, fill=False, histtype='step', label = 'Threshold 2.5 MeV')
        legend = plt.legend(loc="best")#(loc="best")
        plt.xlabel(r'$' + '\\' + train_target + '_{NN}$ - $' + '\\'+ train_target +'_{truth}$')
        plt.ylabel(r'')
        frame = legend.get_frame()
        frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑
        plt.title('')	
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_dist_res_c_thresholds, dpi=300)

        print(trained_df_sub_0[0:10])
        plt.figure() 
        plt.hist(trained_df_sub_0.preTheta, bins=50, range=Range_Cry, alpha=1.0, linewidth=1, fill=False, histtype='step', label = 'Threshold 0.0 MeV')
        plt.hist(trained_df_sub_1.preTheta, bins=50, range=Range_Cry, alpha=1.0, linewidth=1, fill=False, histtype='step', label = 'Threshold 1.0 MeV')
        plt.hist(trained_df_sub_2.preTheta, bins=50, range=Range_Cry, alpha=1.0, linewidth=1, fill=False, histtype='step', label = 'Threshold 2.5 MeV')
        legend = plt.legend(loc="best")#(loc="best")
        plt.xlabel(r'$' + '\\' + train_target + '_{NN}$')
        plt.ylabel(r'')
        frame = legend.get_frame()
        frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑
        plt.title('')	
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_dist_c_thresholds, dpi=300)


        plt.show()


def draw_phase(testh5_phases, h5key, train_target, ID_crystal, definition):
        # ****** load the h5 file	
        print('testh5_phases[0] :  ', testh5_phases[0])
        trained_df_phase2 = pd.read_hdf(testh5_phases[0], h5key_phases[0])
        trained_df_phase3 = pd.read_hdf(testh5_phases[1], h5key_phases[1])

        Phi1, Phi2,  Phi_c, Theta1, Theta2, Theta_c = getCrystalPhiAndTheta(ID_crystal, definition)

        if train_target == 'phi':
            trained_df_phase2['res_rec'] = trained_df_phase2.phi    - trained_df_phase2.mcPhi
            trained_df_phase2['res_pre'] = trained_df_phase2.prePhi - trained_df_phase2.mcPhi
            trained_df_phase3['res_rec'] = trained_df_phase3.phi    - trained_df_phase3.mcPhi
            trained_df_phase3['res_pre'] = trained_df_phase3.prePhi - trained_df_phase3.mcPhi
            mc_col  = 'mcPhi'
            rec_col = 'phi'
            pre_col = 'prePhi'
            angle1 = Phi1
            angle2 = Phi2
            angle_c = Phi_c
            Range_Cry = [Phi1*0.99,Phi2*1.01]
            Range = [-3.2, 3.2]
        elif train_target == 'theta':
            trained_df_phase2['res_rec'] = trained_df_phase2.theta    - trained_df_phase2.mcTheta
            trained_df_phase2['res_pre'] = trained_df_phase2.preTheta - trained_df_phase2.mcTheta
            trained_df_phase3['res_rec'] = trained_df_phase3.theta    - trained_df_phase3.mcTheta
            trained_df_phase3['res_pre'] = trained_df_phase3.preTheta - trained_df_phase3.mcTheta
            mc_col  = 'mcTheta'
            rec_col = 'theta'
            pre_col = 'preTheta'
            angle1 = Theta1
            angle2 = Theta2
            angle_c = Theta_c
            Range_Cry = [Theta1*0.99,Theta2*1.01]
            Range = [0.4, 2.4]
        else:
            print("Wrong train target!")

        print(trained_df_phase2[0:10])
        
        trained_df_sub_phase2 = trained_df_phase2[(trained_df_phase2.mcPhi>=Phi1)&(trained_df_phase2.mcPhi<=Phi2)&(trained_df_phase2.mcTheta>=Theta1)&(trained_df_phase2.mcTheta<=Theta2)]
        trained_df_sub_phase3 = trained_df_phase3[(trained_df_phase3.mcPhi>=Phi1)&(trained_df_phase3.mcPhi<=Phi2)&(trained_df_phase3.mcTheta>=Theta1)&(trained_df_phase3.mcTheta<=Theta2)]
        print(trained_df_sub_phase2[0:10])

        plt.hist(trained_df_sub_phase2.res_pre, bins=50, range=[-0.03, 0.03], alpha=1.0, linewidth=1, fill=False, histtype='step', label = 'Phase 2', normed = 100)
        plt.hist(trained_df_sub_phase3.res_pre, bins=50, range=[-0.03, 0.03], alpha=1.0, linewidth=1, fill=False, histtype='step', label = 'Phase 3', normed = 100)
        legend = plt.legend(loc="best")#(loc="best")
        plt.xlabel(r'$' + '\\' + train_target + '_{NN}$ - $' + '\\'+ train_target +'_{truth}$')
        plt.ylabel(r'')
        frame = legend.get_frame()
        frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑
        plt.title('')	
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_dist_res_c_phases, dpi=300)

        plt.figure() 
        plt.hist(trained_df_sub_phase2.preTheta, bins=50, range=Range_Cry, alpha=1.0, linewidth=1, fill=False, histtype='step', label = 'Phase 2', normed = 100)
        plt.hist(trained_df_sub_phase3.preTheta, bins=50, range=Range_Cry, alpha=1.0, linewidth=1, fill=False, histtype='step', label = 'Phase 3', normed = 100)
        legend = plt.legend(loc="best")#(loc="best")
        plt.xlabel(r'$' + '\\' + train_target + '_{NN}$')
        plt.ylabel(r'')
        frame = legend.get_frame()
        frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑
        plt.title('')	
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_dist_c_phases, dpi=300)

        plt.show()


energy = '1000MeV'
Dir    = 'Energy' + energy + '/'
if not os.path.isdir(Dir):
   os.mkdir(Dir)

Dir_train = Dir + 'train_runtime' + '/'
if not os.path.isdir(Dir_train):
   os.mkdir(Dir_train)

train_target = 'theta'
Dir_target = Dir_train + train_target + '/'  # + time.strftime( "%Y%m%d%H%M%S",time.localtime())
if not os.path.isdir(Dir_target):
   os.mkdir(Dir_target)

ID_train = '7151069798'  # '7151069798'   # '2927360606'   # '3975284924' 
ID_test  = '7151069798' # '9242450162'  #'7151069798'  # '7245655874'   # '4908190819'   # '0010973571' # '3975284924' #'2927360606' # '7151069798' 

Dir_training = Dir_target + 'train_' + ID_train + '/'
if not os.path.isdir(Dir_training):
   os.mkdir(Dir_training)

Dir_pkl = Dir_training + 'train_pkl_' + ID_train + '/'
if not os.path.isdir(Dir_pkl):
   os.mkdir(Dir_pkl)

Dir_test = Dir_target + 'test_' + ID_test + '/'
if not os.path.isdir(Dir_test):
   os.mkdir(Dir_test)

threshold_train = 0.0 
threshold       = 0.0   #0.0  #0.001  #0.0025

Dir_threshold = Dir_test + 'threshold' + str(threshold) + '/'
if not os.path.isdir(Dir_threshold):
   os.mkdir(Dir_threshold)

phase = '' # '_phase3'	
crystal = '' #'_c2626'
trainh5file = 'B2APi0selection_' + ID_train + '_crystal_addmcMatchWeight_modified_threshold' + str(threshold_train)  + crystal + '.h5'
testh5file  = 'B2APi0selection_' + ID_test  + '_crystal_addmcMatchWeight_modified_threshold' + str(threshold) + phase  + crystal + '.h5'

h5key_train = 'crystal_'+ ID_train
h5key_test  = 'crystal_'+ ID_test

#trainpklfile = Dir_target + 'NN_train_params_' + ID_train + '.pkl'
trainpklfile = Dir_pkl + 'NN_train_params_epoch99.pkl'

validationh5 = Dir_training    + 'NN_train_test_' + ID_train + '.h5'
train_lossh5 = Dir_training    + 'NN_train_loss_' + ID_train + '.h5'
testh5       = Dir_threshold   + 'NN_test_'       + ID_test  + '.h5'


trainedlossplot   = Dir_training + 'NN_train_test_' + ID_train + '_' + train_target + 'loss.png'

#train(trainh5file, h5key_train, trainpklfile, validationh5, trainedlossplot, train_target, train_lossh5)

for i in range(8):
	train(trainh5file, h5key_train, trainpklfile, validationh5, trainedlossplot, train_target, train_lossh5, i+1)

ID_crystal = '2626'  #'7090' # '5218' # '2626'
definition = '2'
trained_dist_res_c_thresholds  = Dir_test + 'NN_test_' + ID_test + '_' + train_target + '_' + ID_crystal + '_' + definition + '_res_crystal_thresholds.png'
trained_dist_c_thresholds      = Dir_test + 'NN_test_' + ID_test + '_' + train_target + '_' + ID_crystal + '_' + definition + '_crystal_thresholds.png'
trained_dist_res_c_phases      = Dir_test + 'NN_test_' + ID_test + '_' + train_target + '_' + ID_crystal + '_' + definition + '_res_crystal_phases.png'
trained_dist_c_phases          = Dir_test + 'NN_test_' + ID_test + '_' + train_target + '_' + ID_crystal + '_' + definition + '_crystal_phases.png'

#draw_result(validationh5, h5key_train, train_target, ID_crystal, definition)
#draw_result(testh5, h5key_test, train_target, ID_test, ID_crystal, definition, Dir_threshold, train_lossh5, h5key_train)

#application(testh5file, testh5,  h5key_test, trainpklfile, train_target)

	
testh5_thresholds = []
testh5_thresholds.append('/home/huyu/Pi0_correction/pytorch_CNN/Energy1000MeV/train7/theta/test_3975284924/threshold0.0/NN_test_3975284924.h5')
testh5_thresholds.append('/home/huyu/Pi0_correction/pytorch_CNN/Energy1000MeV/train7/theta/test_3975284924/threshold0.001/NN_test_3975284924.h5')
testh5_thresholds.append('/home/huyu/Pi0_correction/pytorch_CNN/Energy1000MeV/train7/theta/test_3975284924/threshold0.0025/NN_test_3975284924.h5')
#draw_threshold(testh5_thresholds, h5key_test, train_target, ID_crystal, definition)

	
testh5_phases = []
testh5_phases.append('/home/huyu/Pi0_correction/pytorch_CNN/Energy1000MeV/train7/theta/test_3975284924/threshold0.0/NN_test_3975284924.h5')
testh5_phases.append('/home/huyu/Pi0_correction/pytorch_CNN/Energy1000MeV/train7/theta/test_4908190819/threshold0.0/NN_test_4908190819.h5')
h5key_phases = ['crystal_3975284924', 'crystal_4908190819']
#draw_phase(testh5_phases, h5key_phases, train_target, ID_crystal, definition)


#Phi1, Phi2,  Phi_c, Theta1, Theta2, Theta_c = getCrystalPhiAndTheta(ID_crystal, definition)
