import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tables
from logger import Logger
import os, sys, shutil
import calculateCrystalPhiTheta
from calculateCrystalPhiTheta import getCrystalPhiAndTheta
from matplotlib.ticker import NullFormatter
from matplotlib.colors import Colormap
import matplotlib.colors as colors
 
#torch.manual_seed(1)


def draw_result(timeh5, h5key_time):
        time_1CPU_df  = pd.read_hdf(timeh5, h5key_time + '_CPU_1t')
        time_2CPU_df  = pd.read_hdf(timeh5, h5key_time + '_CPU_2t')
        time_3CPU_df  = pd.read_hdf(timeh5, h5key_time + '_CPU_3t')
        time_4CPU_df  = pd.read_hdf(timeh5, h5key_time + '_CPU_4t')
        time_5CPU_df  = pd.read_hdf(timeh5, h5key_time + '_CPU_5t')
        time_6CPU_df  = pd.read_hdf(timeh5, h5key_time + '_CPU_6t')
        time_7CPU_df  = pd.read_hdf(timeh5, h5key_time + '_CPU_7t')
        time_8CPU_df  = pd.read_hdf(timeh5, h5key_time + '_CPU_8t')
        time_bacth1_df = pd.read_hdf(timeh5, h5key_time + '_batch64')
        time_bacth2_df = pd.read_hdf(timeh5, h5key_time + '_batch512_t')
        time_bacth3_df = pd.read_hdf(timeh5, h5key_time + '_batch1024')

        time_CPU_df = pd.DataFrame({'Epoch':time_1CPU_df['epoch'], '1CPU':time_1CPU_df['time'], '2CPU':time_2CPU_df['time'], '3CPU':time_3CPU_df['time'], '4CPU':time_4CPU_df['time'], '4CPU':time_4CPU_df['time'], '5CPU':time_5CPU_df['time'], '6CPU':time_6CPU_df['time'], '7CPU':time_7CPU_df['time'], '8CPU':time_8CPU_df['time']})
        time_bacth_df = pd.DataFrame({'Epoch':time_bacth1_df['epoch'], 'batch64': time_bacth1_df['time'], 'batch512': time_bacth2_df['time'], 'batch1024': time_bacth3_df['time']})
	
        print(time_CPU_df)
        
        time_CPU_df.plot(x='Epoch',y=['1CPU','2CPU','3CPU','4CPU','5CPU','6CPU','7CPU','8CPU'],  lw = 2)
        plt.xlabel(r'Epoch')
        plt.ylabel(r'Time')
        plt.legend(loc = 'best')
        #plt.ylim(0.2,0.6) 
        plt.grid(True, alpha=0.8)
        # plt.savefig(trained_loss_step, dpi=300)

        time_bacth_df.plot(x='Epoch',y=['batch64','batch512', 'batch1024'],  lw = 2)
        plt.xlabel(r'Epoch')
        plt.ylabel(r'Time')
        plt.legend(loc = 'best')
        plt.grid(True, alpha=0.8)
        # plt.savefig(trained_loss_epoch, dpi=300)

        plt.show()


timeh5 = '/home/huyu/Pi0_correction/pytorch_NN/Energy1000MeV/train_runtime/theta/train_7151069798/NN_train_runTime7151069798.h5'
h5key_time = 'crystal_7151069798'
draw_result(timeh5, h5key_time)


