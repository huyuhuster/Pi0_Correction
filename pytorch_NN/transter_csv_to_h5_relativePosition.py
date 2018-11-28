import pandas as pd
from pandas import HDFStore
import numpy as np
import matplotlib.pyplot as plt
import tables
import os

# def csv_h5(csvfile,hdffile, h5key):
#    mydf = pd.read_csv(csvfile)
#    mydf = mydf[(mydf.theta>0.561996) & (mydf.theta<2.2463)]
#    mydf.to_hdf(hdffile, key=h5key, mode='w')
#    print(mydf.head())
#    for column in mydf.columns:
#    	print(column)

def csv_h5(csvfile,hdffile, h5key):
	reader = pd.read_csv(csvfile, chunksize = 300000)
    # chunk5 = reader.get_chunk(5)
    # print(chunk5)
	i = 0
	for chunk in reader:
		print('chunk:',i)
		i = i+1
		chunk.to_hdf(hdffile, key=h5key, append = True, mode='a')
        # if i>10:
        #   break

def read_h5(hdffile, h5key):
	i = 0
	reader = pd.read_hdf(hdffile, h5key, chunksize=100000)
	for chunk in reader:
		print('chunk:',i)
		i = i+1
		print(chunk.columns)
		if i>0:
			break

def modified_h5(hdffile, hdffile_m, h5key, threshold):
	if os.path.exists(hdffile_m):
		os.remove(hdffile_m)
	else:
		print("The file",  hdffile_m, "does not exist")
	reader = pd.read_hdf(hdffile, h5key, chunksize=100000)
	ichunk = 0
	for mydf_readd5 in reader:
		print('chunk: ', ichunk)
		ichunk = ichunk + 1
        # mydf_readd5 = pd.read_hdf(hdffile, h5key, start =0, stop = 10000000)
		Phi1 =  1.4483643081842303 
		Phi2 =  1.4794450818866633  
		Theta1 =   0.8336485385269553
		Theta2 =   0.8647267287924316
        # Theta1 = 0.561996,   Theta2 = 2.2463	  	
		mydf_readd5 = mydf_readd5[(mydf_readd5.mcTheta>Theta1) & (mydf_readd5.mcTheta<Theta2)]
		mydf_readd5 = mydf_readd5[(mydf_readd5.mcPhi>Phi1) & (mydf_readd5.mcPhi<Phi2)]
		mydf_readd5 = mydf_readd5.replace([np.nan,-1.],0.0)
        # print(mydf_readd5.head())
        # for column in mydf_readd5.columns:
        #	print(column)
		file_pos = open("digits_position_dirction/digits_position.txt")
		id_theta = {}
		id_phi = {}
		line = file_pos.readline()
		line = file_pos.readline()
		while line:
			line_list = line.rstrip('\n').split('\t')
			id_theta[line_list[0]]= float(line_list[1])
			id_phi[line_list[0]] = float(line_list[2])
			line = file_pos.readline()

        mydf_readd5['gamma_CScentralCellid'] 

        # print(id_theta)
		index_list = ['mcPhi','mcTheta','phi','theta']
		index_list_CSCid = []
        #index_list_CSW = []
		index_list_CSE = []
		for i in range(25):
			index_list_CSCid.append('CSCid_'+str(i))
			#index_list_CSW.append('CSW_'+str(i))
			index_list_CSE.append('CSE_'+str(i))
       
        # ****** Set the threshold of crystal energy	
		if threshold>0 :
			mydf_readd5[mydf_readd5[index_list_CSE]<threshold] = 0.0
        # print(mydf_readd5[0:20][index_list_CSE])
        
        # ****** Standardization 
		mydf_readd5[index_list_CSE] = ((mydf_readd5[index_list_CSE].T - mydf_readd5[index_list_CSE].T.mean()) / mydf_readd5[index_list_CSE].T.std()).T
        # mydf_readd5[index_list_CSE] = ((mydf_readd5[index_list_CSE].T - mydf_readd5[index_list_CSE].T.min()) / (mydf_readd5[index_list_CSE].T.max()-mydf_readd5[index_list_CSE].T.min())).T

		index_list = index_list +  index_list_CSE 
        # print(index_list)
		mydf_readd5 = mydf_readd5[index_list] 
        # print(mydf_readd5.loc[:,['CSCPhi_1','CSCid_1']])
        # print(mydf_readd5.head())
		mydf_readd5.to_hdf(hdffile_m, key=h5key, append = True, mode='a')


	
ID = '7151069798'  #'4908190819'  #'0010973571'  #'7151069798'  #'3975284924' # '2927360606'   #'3975284924'	
phase = '' # '_pahse3'
threshold = 0.0  #0.001 0.0025 # GeV
csvfile = 'B2APi0selection_' + ID + '_crystal_addmcMatchWeight' + phase + '.csv'
hdffile = 'B2APi0selection_' + ID + '_crystal_addmcMatchWeight' + phase + '.h5'
hdffile_m = 'B2APi0selection_' + ID + '_crystal_addmcMatchWeight_modified_threshold' + str(threshold) + phase + '_c2626' + '.h5'
h5key = 'crystal_' + ID
print('hdffile_m:  ', hdffile_m)

#csv_h5(csvfile,hdffile, h5key)
#read_h5(hdffile_m, h5key)
modified_h5(hdffile, hdffile_m, h5key, threshold)
