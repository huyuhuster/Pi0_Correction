import pandas as pd
from pandas import HDFStore
import numpy as np
import matplotlib.pyplot as plt
import tables
import os

#def csv_h5(csvfile,hdffile, h5key):
#	mydf = pd.read_csv(csvfile)
#	mydf = mydf[(mydf.theta>0.561996) & (mydf.theta<2.2463)]
#	mydf.to_hdf(hdffile, key=h5key, mode='w')
#	print(mydf.head())
#	for column in mydf.columns:
#		print(column)

def csv_h5(csvfile,hdffile, h5key):
	reader = pd.read_csv(csvfile, chunksize = 300000)
#	chunk5 = reader.get_chunk(5)
#	print(chunk5)
	i = 0
	for chunk in reader:
		print('chunk:',i)
		i = i+1
		chunk.to_hdf(hdffile, key=h5key, append = True, mode='a')
#		if i>10:
#			break

def read_h5(hdffile, h5key):
	i = 0
	reader = pd.read_hdf(hdffile, h5key, chunksize=100000)
	for chunk in reader:
		print('chunk:',i)
		i = i+1
		print(chunk.columns)
		if i>0:
			break

def modified_h5(hdffile, hdffile_m, h5key):
	if os.path.exists(hdffile_m):
		os.remove(hdffile_m)
	else:
		print("The file",  hdffile_m, "does not exist")
	reader = pd.read_hdf(hdffile, h5key, chunksize=100000)
	ichunk = 0
	for mydf_readd5 in reader:
		print('chunk: ', ichunk)
		ichunk = ichunk + 1
#		mydf_readd5 = pd.read_hdf(hdffile, h5key, start =0, stop = 10000000)
#		mydf_readd5 = pd.read_hdf(hdffile, h5key, start = 10000000)
#		mydf_readd5 = pd.read_hdf(hdffile, h5key)
		mydf_readd5 = mydf_readd5[(mydf_readd5.theta>0.561996) & (mydf_readd5.theta<2.2463)]
		mydf_readd5 = pd.DataFrame(mydf_readd5.replace([np.nan,-1.],0.0))
#		print(mydf_readd5.head())
#		for column in mydf_readd5.columns:
#			print(column)
		file_pos = open("digits_position_dirction/digits_position.txt")
#		all_lines = file_pos.readlines()
#		for line in all_lines:
		id_theta = {}
		id_phi = {}
		line = file_pos.readline()
		line = file_pos.readline()
		while line:
			line_list = line.rstrip('\n').split('\t')
			id_theta[line_list[0]]= float(line_list[1])
			id_phi[line_list[0]] = float(line_list[2])
			line = file_pos.readline()

#		print(id_theta)
		index_list = ['mcPhi','mcTheta','phi','theta']
		index_list_CSCid = []
		index_list_CSW = []
		index_list_CSE = []
		index_list_CSCTheta = []
		index_list_CSCPhi = []
		for i in range(25):
			index_list_CSCid.append('CSCid_'+str(i))
			index_list_CSW.append('CSW_'+str(i))
			index_list_CSE.append('CSE_'+str(i))
			index_list_CSCTheta.append('CSCTheta_'+str(i))
			index_list_CSCPhi.append('CSCPhi_'+str(i))
			mydf_readd5['CSCTheta_'+str(i)] = mydf_readd5['CSCid_'+str(i)].map(str).map(id_theta)
			mydf_readd5['CSCPhi_'+str(i)] = mydf_readd5['CSCid_'+str(i)].map(str).map(id_phi)
       
#   Standardization 
		mydf_readd5[index_list_CSE] = ((mydf_readd5[index_list_CSE].T - mydf_readd5[index_list_CSE].T.mean()) / mydf_readd5[index_list_CSE].T.std()).T
		mydf_readd5[index_list_CSCTheta] = (mydf_readd5[index_list_CSCTheta] - mydf_readd5[index_list_CSCTheta].stack().mean())/ mydf_readd5[index_list_CSCTheta].stack().std()
		mydf_readd5[index_list_CSCPhi] = (mydf_readd5[index_list_CSCPhi] - mydf_readd5[index_list_CSCPhi].stack().mean())/ mydf_readd5[index_list_CSCPhi].stack().std()
#		mydf_readd5[index_list_CSE] = ((mydf_readd5[index_list_CSE].T - mydf_readd5[index_list_CSE].T.min()) / (mydf_readd5[index_list_CSE].T.max()-mydf_readd5[index_list_CSE].T.min())).T
#		mydf_readd5[index_list_CSCTheta] = (mydf_readd5[index_list_CSCTheta] - mydf_readd5[index_list_CSCTheta].stack().min())/ (mydf_readd5[index_list_CSCTheta].stack().max() - mydf_readd5[index_list_CSCTheta].stack().min())
#		mydf_readd5[index_list_CSCPhi] = (mydf_readd5[index_list_CSCPhi] - mydf_readd5[index_list_CSCPhi].stack().min())/ (mydf_readd5[index_list_CSCPhi].stack().max() - mydf_readd5[index_list_CSCPhi].stack().min())

		index_list = index_list +  index_list_CSE + index_list_CSCTheta + index_list_CSCPhi
#		print(index_list)
		mydf_readd5 = mydf_readd5[index_list] 
#		print(mydf_readd5.loc[:,['CSCPhi_1','CSCid_1']])
#		print(mydf_readd5.head())
		mydf_readd5.to_hdf(hdffile_m, key=h5key, append = True, mode='a')


	
ID = '3975284924'  #'7151069798'  #'3975284924' # '2927360606'   #'3975284924'	
csvfile = 'B2APi0selection_' + ID + '_crystal_addmcMatchWeight.csv'
hdffile = 'B2APi0selection_' + ID + '_crystal_addmcMatchWeight.h5'
hdffile_m = 'B2APi0selection_' + ID + '_crystal_addmcMatchWeight_modified.h5'
h5key = 'crystal_' + ID

#csv_h5(csvfile,hdffile, h5key)
#read_h5(hdffile_m, h5key)
modified_h5(hdffile, hdffile_m, h5key)
