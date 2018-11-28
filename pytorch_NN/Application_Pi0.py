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
import datetime

import Draw_result
#from Draw_result import draw_result
 
#torch.manual_seed(1)
 
EPOCH = 10000
BATCH_SIZE = 128  #8192  #32768   #8192 #32768  #16384
BATCH_SIZE_test = 100
LR = 0.0001 #0.00001


class Net(nn.Module):
    def __init__(self, n_feature,  n_output):
        super(Net, self).__init__()
        # ****** define the style of every layer 
        self.hidden1 = torch.nn.Linear(n_feature, 640)   # define hiden layer, liner out put
        # self.drop1   = torch.nn.Dropout(0.5)
        self.hidden2 = torch.nn.Linear(640, 400)   # define hiden layer, liner out put
        self.hidden3 = torch.nn.Linear(400, 200)   # define hiden layer, liner out put
        self.hidden4 = torch.nn.Linear(200, 140)   # define hiden layer, liner out put
        self.predict = torch.nn.Linear(140, n_output)   # define output layer, liner out put

 
    def forward(self, x):
        x = self.hidden1(x)
        # x = self.drop1(x)
        x = torch.tanh(x)  #tanh(x)  #sigmoid(x) #softplus(x) #relu(x)
        x = self.hidden2(x)
        x = torch.tanh(x)  #sigmoid(x) #softplus(x) #relu(x)
        x = self.hidden3(x)
        x = torch.tanh(x)  #sigmoid(x) #softplus(x) #relu(x)
        x = self.hidden4(x)
        x = torch.tanh(x)  #sigmoid(x) #softplus(x) #relu(x)
        x = self.predict(x)
        return x
 
index_list_CSE0 = []
index_list_CSCTheta0 = []
index_list_CSCPhi0 = []
index_list_CSE1 = []
index_list_CSCTheta1 = []
index_list_CSCPhi1 = []
for i in range(25):
      index_list_CSE0.append('CSE0_'+str(i))
      index_list_CSCTheta0.append('CSCTheta0_'+str(i))
      index_list_CSCPhi0.append('CSCPhi0_'+str(i))
      index_list_CSE1.append('CSE1_'+str(i))
      index_list_CSCTheta1.append('CSCTheta1_'+str(i))
      index_list_CSCPhi1.append('CSCPhi1_'+str(i))


        
def application(h5file, testh5, h5key_test, pklfile, train_target, gamma_id):
	print(h5file)
	if os.path.exists(testh5):
	        print("The file",  testh5, " exist, will remove it!")
	        os.remove(testh5)
	else:
	        print("The file",  testh5, "does not exist! Will create it.")

	index_list = []

	if gamma_id == '0':
		index_list =  index_list_CSE0 + index_list_CSCTheta0 + index_list_CSCPhi0 
		
	elif gamma_id == '1':
		index_list =  index_list_CSE1 + index_list_CSCTheta1 + index_list_CSCPhi1 
	else:
	    print("Wrong gamma ID!")

	reader = pd.read_hdf(h5file, h5key_test, chunksize=100000)
	if os.path.exists(testh5):
		print("The file",  testh5, " exist,  and deleted!")
		os.remove(testh5)
	else:
		print("The file",  testh5, "does not exist!")
	print('Begain test....')
	for mydf_test in reader:
		# ****** test dataset	
		test_data_np = mydf_test[index_list].replace(np.nan, 0.0).values
		test_data_tensor = torch.from_numpy(test_data_np).double()
		
	    
	    # ****** load the model
		net = Net(n_feature=75,  n_output=1)
		pklfile9 = 'Energy1000MeV/train9/theta/train_4055541376/train_pkl_4055541376/NN_train_params_epoch1379.pkl' 
		net.load_state_dict(torch.load(pklfile, map_location='cpu'))
		net = net.double()
		
		print('test calculat....')
		test_output = net(test_data_tensor)
		print('test calculat successfully!')
		test_pred_y = test_output.data.numpy()
		
		pred_df = pd.DataFrame(mydf_test[['mcPhi'+gamma_id,'phi'+gamma_id, 'mcTheta'+gamma_id, 'theta'+gamma_id]])
		if train_target == 'phi':
		   pred_df['prePhi'+gamma_id] = test_pred_y
		elif train_target == 'theta':
		   pred_df['preTheta'+gamma_id] = test_pred_y
		pred_df.to_hdf(testh5, key=h5key_test, append = True, mode='a')
	
	print('end of test, begain plot....')

def draw_Mpi0(rec_h5file, gamma0_phi_h5, gamma1_phi_h5, gamma0_theta_h5, gamma1_theta_h5, Mpi0h5file):
	rec_df = pd.read_hdf(rec_h5file, h5key_test, columns=['Mpi0', 'Epi0', 'E0', 'E1', 'mcE0', 'mcE1','mcPhi0','mcTheta0','phi0','theta0', 'mcPhi1','mcTheta1','phi1','theta1']) 
	gamma0_phi_df =  pd.read_hdf(gamma0_phi_h5, h5key_test,  columns=['mcPhi0', 'prePhi0']) 
	gamma1_phi_df =  pd.read_hdf(gamma1_phi_h5, h5key_test,  columns=['mcPhi1', 'prePhi1']) 
	gamma0_theta_df =  pd.read_hdf(gamma0_theta_h5, h5key_test,  columns=['mcTheta0', 'preTheta0']) 
	gamma1_theta_df =  pd.read_hdf(gamma1_theta_h5, h5key_test,  columns=['mcTheta1', 'preTheta1']) 
	rec_df['prePhi0'] = gamma0_phi_df['prePhi0']
	rec_df['prePhi1'] = gamma1_phi_df['prePhi1']
	rec_df['preTheta0'] = gamma0_theta_df['preTheta0']
	rec_df['preTheta1'] = gamma1_theta_df['preTheta1']
    #print(pd.DataFrame({'recmcPhi0':rec_df['mcPhi0'], 'premcPhi0':gamma0_phi_df['mcPhi0']}))
	rec_df.to_hdf(Mpi0h5file, key=h5key_test,  mode='w')
	


energy = '1000MeV'
Dir    = 'Energy' + energy + '/'

Dir_train = Dir + 'train8' + '/'

train_target =  'theta' #'phi' #'theta'
Dir_target = Dir_train + train_target + '/'  # + time.strftime( "%Y%m%d%H%M%S",time.localtime())

ID_train = '7151069798'   
ID_test  = '5083269204Pi0' #'7245655874'   

Dir_training = Dir_target + 'train_' + ID_train + '/'

Dir_pkl = Dir_training + 'train_pkl_' + ID_train + '/'

Dir_test = Dir_target + 'test_' + ID_test + '/'
if not os.path.isdir(Dir_test):
   os.mkdir(Dir_test)	

threshold_train = 0.0
threshold       = 0.0   #0.0  #0.001  #0.0025

Dir_threshold = Dir_test + 'threshold' + str(threshold) + '/'
if not os.path.isdir(Dir_threshold):
   os.mkdir(Dir_threshold)

phase = '' # '_phase3'  
testh5file  = 'B2APi0selection_' + ID_test  + '_crystal_addmcMatchWeight_modified_threshold' + str(threshold) +    phase  + '.h5'

h5key_test  = 'crystal_'+ ID_test

#trainpklfile = Dir_train + 'NN_train_params_' + ID_train + '.pkl'
trainpklfile = Dir_pkl + 'NN_train_params_epoch999.pkl'
#trainpklfile = Dir_pkl + 'NN_train_params_epoch79.pkl'


gamma_id = '1' # '0' #'1'
testh5       = Dir_threshold   + 'NN_test_'       + ID_test  + '_gamma' + gamma_id + '.h5'


#application(testh5file, testh5,  h5key_test, trainpklfile, train_target, gamma_id)

gamma0_phi_h5   =  'Energy1000MeV/train8/phi/test_5083269204Pi0/threshold0.0/NN_test_5083269204Pi0_gamma0.h5'
gamma1_phi_h5   =  'Energy1000MeV/train8/phi/test_5083269204Pi0/threshold0.0/NN_test_5083269204Pi0_gamma1.h5'
gamma0_theta_h5 =  'Energy1000MeV/train8/theta/test_5083269204Pi0/threshold0.0/NN_test_5083269204Pi0_gamma0.h5'
gamma1_theta_h5 =  'Energy1000MeV/train8/theta/test_5083269204Pi0/threshold0.0/NN_test_5083269204Pi0_gamma1.h5'

Dir_Pi0rec = Dir_train + 'Pi0_M_rec' + '/'
if not os.path.isdir(Dir_Pi0rec):
   os.mkdir(Dir_Pi0rec)

Mpi0h5file = Dir_Pi0rec + 'Pi0_M.h5'

draw_Mpi0(testh5file, gamma0_phi_h5, gamma1_phi_h5, gamma0_theta_h5, gamma1_theta_h5, Mpi0h5file)

