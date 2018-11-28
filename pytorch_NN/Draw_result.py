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


def draw_result(testh5, h5key_test, train_target, ID_test, ID_crystal, definition, Dir_threshold, train_lossh5, h5key_train):
        trained_dist           = Dir_threshold + 'NN_test_' + ID_test + '_' + train_target + '_' + ID_crystal + '_' + definition + '.png'
        trained_dist_c         = Dir_threshold + 'NN_test_' + ID_test + '_' + train_target + '_' + ID_crystal + '_' + definition + '_crystal.png'
        trained_dist_c_RecVsMC = Dir_threshold + 'NN_test_' + ID_test + '_' + train_target + '_' + ID_crystal + '_' + definition + '_crystal_RecVsMC.png'
        trained_dist_c_PreVsMC = Dir_threshold + 'NN_test_' + ID_test + '_' + train_target + '_' + ID_crystal + '_' + definition + '_crystal_PreVsMC.png'
        trained_dist_res       = Dir_threshold + 'NN_test_' + ID_test + '_' + train_target + '_' + ID_crystal + '_' + definition + '_res.png'
        trained_dist_res_c     = Dir_threshold + 'NN_test_' + ID_test + '_' + train_target + '_' + ID_crystal + '_' + definition + '_res_crystal.png'
        trained_loss_epoch     = Dir_threshold + 'NN_test_' + ID_test + '_' + train_target + '_' + ID_crystal + '_' + definition + '_loss_epoch.png'
        trained_loss_step      = Dir_threshold + 'NN_test_' + ID_test + '_' + train_target + '_' + ID_crystal + '_' + definition + '_loss_step.png'
        # ****** load the h5 file	
        trained_df = pd.read_hdf(testh5, h5key_test)
        loss_step_df  = pd.read_hdf(train_lossh5, h5key_train + 'step')
        loss_epoch_df = pd.read_hdf(train_lossh5, h5key_train + 'epoch')

        Phi1, Phi2,  Phi_c, Theta1, Theta2, Theta_c = getCrystalPhiAndTheta(ID_crystal, definition)
        print("Phi1:  ", Phi1, "  Phi2:  ", Phi2)
        print("Theta1:  ", Theta1, "  Theta2:  ", Theta2)

        if train_target == 'phi':
            trained_df['res_rec'] = trained_df.phi-trained_df.mcPhi
            trained_df['res_pre'] = trained_df.prePhi-trained_df.mcPhi
            mc_col  = 'mcPhi'
            rec_col = 'phi'
            pre_col = 'prePhi'
            angle1 = Phi1
            angle2 = Phi2
            angle_c = Phi_c
            Range_Cry = [Phi1*0.99,Phi2*1.01]
            Range = [-3.2, 3.2]
        elif train_target == 'theta':
            trained_df['res_rec'] = trained_df.theta-trained_df.mcTheta
            trained_df['res_pre'] = trained_df.preTheta-trained_df.mcTheta
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

        print(trained_df[0:10])
        # fig1 = plt.figure() 
        # plt.style.use('ggplot')
        
        col_list = [rec_col,pre_col,mc_col]	
        trained_df_sub = trained_df[col_list][(trained_df.mcPhi>=Phi1)&(trained_df.mcPhi<=Phi2)&(trained_df.mcTheta>=Theta1)&(trained_df.mcTheta<=Theta2)]

        trained_df_sub.plot.hist(bins=50, range=Range_Cry, alpha=1.0, linewidth=1, fill=False, histtype='step')
        legend = plt.legend(loc="best", labels=(r'$\theta_{rec}$', r'$\theta_{NN}$', r'$\theta_{Truth}$'))#(loc="best")
        plt.plot([angle1, angle1],[0, 50], 'k--')
        plt.text(angle1, 0, 'A', fontdict={'size': 10, 'color':  'blue'})
        plt.plot([angle2, angle2],[0, 50], 'k--')
        plt.text(angle2, 0, 'B', fontdict={'size': 10, 'color':  'blue'})
        plt.plot([angle_c, angle_c],[0, 50], 'k--')
        plt.text(angle_c, 0, 'C', fontdict={'size': 10, 'color':  'blue'})
        plt.xlabel(r'$' + '\\'+ train_target + '$')
        plt.ylabel(r'')
        frame = legend.get_frame()
        frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑
        plt.title('')	
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_dist_c, dpi=300)

	
        # definitions for the axes
        left, width    = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02
        
        rect_scatter = [left,   bottom,   width, height]
        rect_histx   = [left,   bottom_h, width, 0.2]
        rect_histy   = [left_h, bottom,   0.2,   height]

        	
        fig1 = plt.figure() 
        axScatter = plt.axes(rect_scatter, xlabel=r'$'+ '\\' + train_target  + '_{rec}$ [rad]', ylabel=r'$'+ '\\' + train_target + '_{Truth}$ [rad]')
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        
        # no labels
        nullfmt = NullFormatter()         # no labels	
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        # the 2d histgram plot	
        axScatter.hist2d(trained_df_sub[rec_col], trained_df_sub[mc_col],(50,50), range=[Range_Cry, Range_Cry], norm=colors.LogNorm(), cmap = plt.cm.Blues)
        axScatter.grid(True, alpha=0.8)
        
        axHistx.hist(trained_df_sub[rec_col], bins=50, range=Range_Cry, alpha=1.0, linewidth=1, fill=False, histtype='step') 	
        axHisty.hist(trained_df_sub[mc_col], bins=50, range=Range_Cry, alpha=1.0, linewidth=1, fill=False, histtype='step', orientation='horizontal') 	
        axHistx.grid(True, alpha=0.8)
        axHisty.grid(True, alpha=0.8)
	     
        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())
        plt.savefig(trained_dist_c_RecVsMC, dpi=300)


        fig2 = plt.figure() 
        axScatter = plt.axes(rect_scatter, xlabel=r'$'+ '\\' + train_target  + '_{NN}$ [rad]', ylabel=r'$'+ '\\' + train_target + '_{Truth}$ [rad]')
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        # no labels
        nullfmt = NullFormatter()         # no labels	
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        # the 2d histgram plot	
        current_cmap = plt.cm.get_cmap('GnBu')
        current_cmap.set_bad(color='blue')
        current_cmap.set_under(color='white', alpha=1.0)
        axScatter.hist2d(trained_df_sub[pre_col],trained_df_sub[mc_col],(50,50), range = [Range_Cry, Range_Cry], norm=colors.LogNorm(), cmap = plt.cm.Blues)
        axScatter.grid(True, alpha=0.8)
        
        axHistx.hist(trained_df_sub[pre_col], bins=50, range=Range_Cry, alpha=1.0, linewidth=1, fill=False, histtype='step') 	
        axHisty.hist(trained_df_sub[mc_col], bins=50, range=Range_Cry, alpha=1.0, linewidth=1, fill=False, histtype='step', orientation='horizontal') 	
        axHistx.grid(True, alpha=0.8)
        axHisty.grid(True, alpha=0.8)
	     
        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())
        plt.savefig(trained_dist_c_PreVsMC, dpi=300)

	
        fig3 = plt.figure() 
        trained_df[col_list].plot.hist(bins=288, range = Range_Cry, alpha=1., linewidth=0.6, fill=False, histtype='step')
        plt.xlabel(r'$'+'\\'+ train_target + '$')
        plt.ylabel(r'')
        legend = plt.legend(loc="best", labels=(r'$\theta_{rec}$', r'$\theta_{NN}$', r'$\theta_{Truth}$'))
        frame = legend.get_frame()
        frame.set_facecolor('none') # 璁剧疆鍥句緥legend鑳屾櫙閫忔槑
        plt.title('')	
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_dist, dpi=300)


        fig4 = plt.figure() 
        trained_df[['res_rec', 'res_pre']].plot.hist(bins=200, range=[-0.03, 0.03], alpha=1.0, fill=False, histtype='step')
        plt.xlabel(r'$' + '\\' + train_target + '$ - $' + '\\'+ train_target +'_{truth}$')
        plt.ylabel(r'')
        plt.legend(loc = 'best', labels=('res_rec', 'res_NN'))
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_dist_res, dpi=300)


        fig5 = plt.figure() 
        trained_df[['res_rec', 'res_pre']][(trained_df.mcPhi>=Phi1)&(trained_df.mcPhi<=Phi2)&(trained_df.mcTheta>=Theta1)&(trained_df.mcTheta<=Theta2)].plot.hist(bins = 50, range=[-0.03, 0.03], alpha=1.0, fill=False, histtype='step')
        plt.xlabel(r'$' + '\\'+ train_target + '$- $'+'\\'+train_target+'_{truth}$')
        plt.ylabel(r'')
        plt.legend(loc = 'best', labels=('res_rec', 'res_NN'))
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_dist_res_c, dpi=300)
        
        fig6 = plt.figure(8.4,5) 
        plt.plot(loss_step_df.step, loss_step_df.train,  lw = 2, label = 'train')
        plt.plot(loss_step_df.step, loss_step_df.test,   lw = 2, label = 'test')
        plt.xlabel(r'Step')
        plt.ylabel(r'Loss')
        plt.legend(loc = 'best')
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_loss_step, dpi=300)

        fig7 = plt.figure(figsize=(8.4,5)) 
        plt.plot(loss_epoch_df.epoch, loss_epoch_df.train,  lw = 2, label = 'train')
        plt.plot(loss_epoch_df.epoch, loss_epoch_df.test,   lw = 2, label = 'test')
        plt.xlabel(r'Epoch')
        plt.ylabel(r'Loss')
        plt.ylim(0,0.0006) 
        plt.legend(loc = 'best')
        plt.grid(True, alpha=0.8)
        plt.savefig(trained_loss_epoch, dpi=300)

        plt.show()






