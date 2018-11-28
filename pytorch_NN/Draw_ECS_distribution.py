import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import root_pandas
import tables
import os, sys, shutil
from matplotlib.ticker import NullFormatter
from matplotlib.colors import Colormap
import matplotlib.colors as colors
 
#torch.manual_seed(1)


def draw_result(testh5, h5key_test = None, gamma = True, Range = [-0.01, 1.0]):
        # ****** load the file	


	if h5key_test:
		columns_CSE = []
		for i in range(25):
			if gamma:
				columns_CSE.append('CSE_'+str(i))
			else:
				columns_CSE.append('CSE0_'+str(i))
		Gamma = pd.read_hdf(testh5, h5key_test, columns=columns_CSE)

		Gamma.plot.hist( sort_columns=False, subplots=True, layout=(5, 5),sharex=True, sharey=True, legend=False, range=Range, bins=20)

	else:
		if gamma:
			mykey = 'gamma'
			Col = 'gamma_CSE'
		else:
			mykey = 'Pi0'
			Col = 'pi0_gamma0_CSE'
		print("Col = ", Col)
		Gamma = root_pandas.read_root(testh5, mykey, columns=Col)

		pd.DataFrame( Gamma[Col].values.tolist()).replace(-9999.0,0).plot.hist( sort_columns=False, subplots=True, layout=(5, 5),sharex=True, sharey=True, legend=False, range=Range,bins=20)

	#plt.show()



rootfile_100 = '/nfs/dust/belle2/user/huyu/Pi0_Correction/particlegun/B2APi0selection_0350861857_crystal_addmcMatchWeight.root'
rootfile_1000 = '/nfs/dust/belle2/user/huyu/Pi0_Correction/particlegun/B2APi0selection_7245655874_crystal_addmcMatchWeight.root'
rootfile_Pi0 = '/nfs/dust/belle2/user/huyu/Pi0_Correction/particlegun/B2APi0selection_5083269204Pi0_crystal_addmcMatchWeight.root'


h5file_100 = '/nfs/dust/belle2/user/huyu/Pi0_Correction/particlegun/B2APi0selection_0350861857_crystal_addmcMatchWeight_modified_threshold0.0.h5' 
h5file_1000 = '/nfs/dust/belle2/user/huyu/Pi0_Correction/particlegun/B2APi0selection_7245655874_crystal_addmcMatchWeight_modified_threshold0.001.h5' 
h5file_Pi0 = '/nfs/dust/belle2/user/huyu/Pi0_Correction/particlegun/B2APi0selection_5083269204Pi0_crystal_addmcMatchWeight_modified_threshold0.0.h5' 


#Range = [-0.01, 0.10]
#draw_result(rootfile_100, Range=Range)
#draw_result(rootfile_1000)
#Range = [-0.01, 1.1]
#draw_result(rootfile_Pi0, gamma=False, Range=Range)


Range = [-3.01, 7.]
draw_result(h5file_100, h5key_test = 'crystal_0350861857',  Range=Range)
draw_result(h5file_1000, h5key_test = 'crystal_7245655874',  Range=Range )
draw_result(h5file_Pi0, h5key_test = 'crystal_5083269204Pi0', gamma=False, Range=Range)

plt.show()
