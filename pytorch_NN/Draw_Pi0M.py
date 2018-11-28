import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tables
import os, sys, shutil
from matplotlib.ticker import NullFormatter
from matplotlib.colors import Colormap
import matplotlib.colors as colors
import ROOT
import math
 
#torch.manual_seed(1)


def get_M(df0):
	E0 = df0['E0']
	E1 = df0['E1']
	p0 = df0['E0']
	p1 = df0['E1']
	pt0 = p0*math.sin(df0['preTheta0'])
	pt1 = p1*math.sin(df0['preTheta1'])
	px0 = pt0*math.cos(df0['phi0'])
	py0 = pt0*math.sin(df0['phi0'])
	pz0 = p0*math.cos(df0['preTheta0'])
	px1 = pt1*math.cos(df0['phi1'])
	py1 = pt1*math.sin(df0['phi1'])
	pz1 = p1*math.cos(df0['preTheta1'])
	gamma0P4 = ROOT.TLorentzVector(px0,py0,pz0,E0)
	gamma1P4 = ROOT.TLorentzVector(px1,py1,pz1,E1)
	return (gamma0P4+gamma1P4).M()


def get_M_rec(df0):
	E0 = df0['E0']
	E1 = df0['E1']
	p0 = df0['E0']
	p1 = df0['E1']
	pt0 = p0*math.sin(df0['theta0'])
	pt1 = p1*math.sin(df0['theta1'])
	px0 = pt0*math.cos(df0['phi0'])
	py0 = pt0*math.sin(df0['phi0'])
	pz0 = p0*math.cos(df0['theta0'])
	px1 = pt1*math.cos(df0['phi1'])
	py1 = pt1*math.sin(df0['phi1'])
	pz1 = p1*math.cos(df0['theta1'])
	gamma0P4 = ROOT.TLorentzVector(px0,py0,pz0,E0)
	gamma1P4 = ROOT.TLorentzVector(px1,py1,pz1,E1)
	return (gamma0P4+gamma1P4).M()


def draw_result(Pi0Mh5,h5key):
        # ****** load the h5 file	
        trained_df = pd.read_hdf(Pi0Mh5, h5key)
        print(trained_df[0:5])
        trained_df['Mpi0_c'] = trained_df.apply(get_M_rec, axis=1)
        trained_df['Mpi0_NN'] = trained_df.apply(get_M, axis=1)
        print(trained_df[['Mpi0_NN','Mpi0','Mpi0_c']])
        

        trained_df[['Mpi0_NN', 'Mpi0']].plot.hist(bins=50, range=[0.08, 0.18], alpha=1.0, fill=False, histtype='step')
        plt.xlabel(r'$M(\pi^{0}) [GeV] $')
        plt.ylabel(r'')
        plt.legend(loc = 'best', labels=('NN', 'rec', 'Calculate'))
        plt.grid(True, alpha=0.8)
        plt.savefig('Mpi0_rec_and_NN.png', dpi=300)


        plt.show()






Pi0Mh5 = '/pnfs/desy.de/belle/local/user/huyu/Pi0_correction/pytorch_CNN/Energy1000MeV/train8/Pi0_M_rec/Pi0_M.h5'
h5key = 'crystal_5083269204Pi0'
draw_result(Pi0Mh5,h5key)
