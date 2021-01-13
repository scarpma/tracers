#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from params import *
import stats
from glob import glob
import os.path as osp



# COMMAND LINE PARSING

if "-h" in sys.argv:
    print("usage: sf_analysis.py <run> <number> [--no_plot]")
    exit()

run = int(sys.argv.pop(1))
number = int(sys.argv.pop(1))
plot = True

while len(sys.argv) > 1:
    if sys.argv[1] == '--no_plot':
        gen_renorm = plot = False
        sys.argv.pop(1)
    else:
        raise NameError('Parsing error, exiting.')
        exit()



# DEFINE SAVE FILES

save_path = f'runs/{run}/HighStat/TOT_{number}'
if not osp.exists(save_path):
    os.makedirs(save_path)

save_file_sfr     = osp.join(save_path, 'sfr.dat')
save_file_sfg     = osp.join(save_path, 'sfg.dat')
save_file_dlr     = osp.join(save_path, 'dlr.dat')
save_file_dlg     = osp.join(save_path, 'dlg.dat')
save_file_ftr     = osp.join(save_path, 'ftr.dat')
save_file_ftg     = osp.join(save_path, 'ftg.dat')
save_file_dlr_ess = osp.join(save_path, 'dlr_ess.dat')
save_file_dlg_ess = osp.join(save_path, 'dlg_ess.dat')



# FIND SF.DAT FILES

read_file = f'runs/{run}/HighStat/sf_{number}_me_*.dat'
read_file_real = '../data/real/sf.dat'
read_file = glob(read_file)
print(f'found {len(read_file)} sf.dat files:')
for file in read_file:
    print('\t', file)



# READ ALREADY COMPUTED SF

sfr = np.loadtxt(read_file_real)

for ii, fl in enumerate(read_file):
    if ii == 0:
        sfg = np.loadtxt(fl)
    else:
        temp = np.loadtxt(fl)
        assert all(temp[:,0] == sfg[:,0])
        sfg[:,1:] = sfg[:,1:] + temp[:,1:]

sfg[:,1:] = sfg[:,1:] / len(read_file)



# COMPUTE LOG DERIVATIVES

dlg       = np.zeros(shape=sfg.shape)
dlg[:,0]  = sfg[:,0]
dlg[:,1:] =  np.gradient( np.log(sfg[:,1:]), np.log(sfg[:,0]), axis=0 )

dlr       = np.zeros(shape=sfr.shape)
dlr[:,0]  = sfr[:,0]
dlr[:,1:] = np.gradient( np.log(sfr[:,1:]), np.log(sfr[:,0]), axis=0 )



# COMPUTE FLATNESS

ftg       = np.zeros(shape=(sfg.shape[0], sfg.shape[1]-1))
ftg[:,0]  = sfg[:,0]
for ii in range(1, sfg.shape[1]-1):
    ftg[:,ii] = sfg[:,ii+1] / (sfg[:,1])**(ii+1)

ftr       = np.zeros(shape=(sfr.shape[0], sfr.shape[1]-1))
ftr[:,0]  = sfr[:,0]
for ii in range(1, sfg.shape[1]-1):
    ftr[:,ii] = sfr[:,ii+1] / (sfr[:,1])**(ii+1)



# COMPUTE LOG DERIVATIVES ESS

dlg_ess       = np.zeros(shape=(dlg.shape[0], dlg.shape[1]-1))
dlg_ess[:,0]  = dlg[:,0]
for ii in range(1, sfg.shape[1]-1):
    dlg_ess[:,ii] = dlg[:,ii+1] / dlg[:,1]

dlr_ess       = np.zeros(shape=(dlr.shape[0], dlr.shape[1]-1))
dlr_ess[:,0]  = dlr[:,0]
for ii in range(1, sfg.shape[1]-1):
    dlr_ess[:,ii] = dlr[:,ii+1] / dlr[:,1]



# SAVE FILES

np.savetxt(save_file_sfr, sfr)
np.savetxt(save_file_sfg, sfg)
np.savetxt(save_file_dlr, dlr)
np.savetxt(save_file_dlg, dlg)
np.savetxt(save_file_ftr, ftr)
np.savetxt(save_file_ftg, ftg)
np.savetxt(save_file_dlr_ess, dlr_ess )
np.savetxt(save_file_dlg_ess, dlg_ess )



# PLOT

if plot:
    op_gen = {'marker':'.','lw':0.4,'ms':25,'markeredgewidth':1 ,
              'markeredgecolor':"black"}
    op_real = {'marker':'^','lw':0.4,'ms':14}
    op_leg = {'markerscale':1, 'ncol':2}
    op_leg_1 = {'markerscale':1, 'ncol':3}

    plt.rcParams['font.size'] = 24
    #plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 17
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams['figure.figsize'] = (17, 12)
    plt.rcParams['legend.markerscale'] = 2

    fig, ax = plt.subplots(2,2)
    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('log')
    ax[0,0].set_xlabel("$\\tau$")
    ax[0,0].set_ylabel('$S_n(\\tau)$')
    ax[0,1].set_xscale('log')
    #ax[0,1].set_yticks([3,6,9,12,15,18])
    #ax[0,1].set_ylim([1,1e6])
    ax[0,1].set_xlabel("$\\tau$")
    ax[0,1].set_ylabel("$F_n(\\tau)$")
    ax[0,1].set_yscale('log')
    ax[0,1].grid(which='minor', alpha=0.2)
    ax[0,1].grid(which='major', alpha=0.9)
    ax[1,0].set_xscale('log')
    #ax[1,0].set_ylim([-0.3,7])
    ax[1,0].set_xlabel("$\\tau$")
    ax[1,0].set_ylabel("$\\xi_n(\\tau)$")
    ax[1,1].set_ylim([0,7.5])
    ax[1,1].set_xscale('log')
    ax[1,1].set_xlabel("$\\tau$")
    ax[1,1].set_ylabel("$\\xi_n(\\tau)/ \\xi_2(\\tau)$")



    # PLOT SF

    for ii in range(1,7):
        ax[0,0].plot(sfr[:,0],sfr[:,ii],label="DNS n="+str((ii)*2),
                     color='C'+str(ii-1),**op_real)
    for ii in range(1,7):
        ax[0,0].plot(sfg[:,0],sfg[:,ii],#label="GAN n="+str((ii)*2),
                     color='C'+str(ii-1),**op_gen)



    # PLOT FLATNESS

    ax[0,1].plot(ftr[:,0], ftr[:,1], label="DNS $F_4$",
                 color='C1',**op_real)
    ax[0,1].plot(ftr[:,0], ftr[:,2], label="DNS $F_6$",
                 color='C2',**op_real)
    ax[0,1].plot(ftr[:,0], ftr[:,3], label="DNS $F_8$",
                 color='C3',**op_real)
    ax[0,1].plot(ftr[:,0], ftr[:,4], label="DNS $F_{10}$",
                 color='C4',**op_real)
    ax[0,1].plot(ftr[:,0], ftr[:,5], label="DNS $F_{12}$",
                 color='C5',**op_real)

    ax[0,1].plot(ftg[:,0], ftg[:,1],# label="GAN $F_4$",
                 color='C1',**op_gen)
    ax[0,1].plot(ftg[:,0], ftg[:,2],# label="GAN $F_6$",
                 color='C2',**op_gen)
    ax[0,1].plot(ftg[:,0], ftg[:,3],# label="GAN $F_8$",
                 color='C3',**op_gen)
    ax[0,1].plot(ftg[:,0], ftg[:,4],# label="GAN $F_{10}$",
                 color='C4',**op_gen)
    ax[0,1].plot(ftg[:,0], ftg[:,5],# label="GAN $F_{12}$",
                 color='C5',**op_gen)



    # PLOT LOGARITMIC DERIVATIVES

    for ii in range(1,7):
        ax[1,0].plot(*(dlr[:,[0,ii]].T), label="DNS n="+str((ii)*2),
                     color='C'+str(ii-1),**op_real)
    for ii in range(1,7):
        ax[1,0].plot(*(dlg[:,[0,ii]].T),# label="GAN n= "+str((ii)*2),
                     color='C'+str(ii-1),**op_gen)



    # PLOT CONTANT VALUES FOR ESS REFERENCE

    ax[1,1].plot(dlr[:,0], (lambda x: [4/2]*len(x))(dlr[:,0]),ls='--',
                 color="C1")
    ax[1,1].plot(dlr[:,0], (lambda x: [6/2]*len(x))(dlr[:,0]),ls='--',
                 color="C2")
    ax[1,1].plot(dlr[:,0], (lambda x: [8/2]*len(x))(dlr[:,0]),ls='--',
                 color="C3")
    ax[1,1].plot(dlr[:,0], (lambda x: [10/2]*len(x))(dlr[:,0]),ls='--',
                 color="C4")
    ax[1,1].plot(dlr[:,0], (lambda x: [12/2]*len(x))(dlr[:,0]),ls='--',
                 color="C5")



    # PLOT LOGARITMIC DERIVATIVES IN ESS

    for ii in range(5):
        ax[1,1].plot(dlr_ess[:,0], dlr_ess[:,ii+1],label="DNS n="+str((ii+2)*2),
                     color='C'+str(ii+1),**op_real)
    for ii in range(5):
        ax[1,1].plot(dlg_ess[:,0], dlg_ess[:,ii+1],#label="GAN n="+str((ii+2)*2),
                     color='C'+str(ii+1),**op_gen)

    ax[0,0].legend(**op_leg)
    ax[0,1].legend(**op_leg_1, loc='upper right')
    ax[1,0].legend(**op_leg)
    ax[1,1].legend(**op_leg)


    fig.tight_layout()
    img_file = osp.join(save_path, 'sf.png')
    fig.savefig(img_file, fmt='png', dpi=60)
