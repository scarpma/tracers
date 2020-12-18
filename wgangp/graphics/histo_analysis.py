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
    print("usage: histo_analysis.py <run> <number> [--no_plot]")
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

save_path = f'../data/{WGAN_TYPE}/runs/{run}/TOT_{number}'
if not osp.exists(save_path):
    os.makedirs(save_path)

save_file = osp.join(save_path, 'pdf.dat')



# FIND SF.DAT FILES

read_file = f'../data/{WGAN_TYPE}/runs/{run}/pdf_{run}_{number}_*.dat'
read_file_real = ['../data/real/pdf0.dat', '../data/real/pdf1.dat',
                    '../data/real/pdf2.dat']
read_file = glob(read_file)
print(f'found {len(read_file)} sf.dat files:')
for file in read_file:
    print('\t', file)



# READ ALREADY COMPUTED SF

vr, vm, vstd = stats.make_hist(read_path_r[0], std=True, out=True)
ar, am, astd = stats.make_hist(read_path_r[1], std=True, out=True)
rr, rm, rstd = stats.make_hist(read_path_r[2], std=True, out=True)

for ii, fl in enumerate(read_file):
    if ii == 0:
        # DA RISCRIVERE IN BASE AL NUOVO FORMAT
        #vg = np.loadtxt(fl)[0:2,:]
        #ag = np.loadtxt(fl)[2:4,:]
        #rg = np.loadtxt(fl)[4:6,:]
        temp = np.loadtxt(fl)
        vg = temp[:,0:2]
        ag = temp[:,2:4]
        rg = temp[:,4:6]
    else:
        # DA RISCRIVERE IN BASE AL NUOVO FORMAT
        #vg_temp = np.loadtxt(fl)[0:2,:]
        #ag_temp = np.loadtxt(fl)[2:4,:]
        #rg_temp = np.loadtxt(fl)[4:6,:]
        #assert all(vg_temp[:,0] == vg[:,0]), 'different bin for velo'
        #assert all(ag_temp[:,0] == ag[:,0]), 'different bin for acce'
        #assert all(rg_temp[:,0] == rg[:,0]), 'different bin for sder'
        #vg[:,1] = vg[:,1] + vg_temp[:,1]
        #ag[:,1] = ag[:,1] + ag_temp[:,1]
        #rg[:,1] = rg[:,1] + rg_temp[:,1]
        temp = np.loadtxt(fl)
        assert all(temp[:,0] == vg[:,0]), 'different bin for velo'
        assert all(temp[:,2] == ag[:,0]), 'different bin for acce'
        assert all(temp[:,4] == rg[:,0]), 'different bin for sder'
        vg[:,1] = vg[:,1] + temp[:,0:2]
        ag[:,1] = ag[:,1] + temp[:,2:4]
        rg[:,1] = rg[:,1] + temp[:,4:6]



# SUM HISTOS AND COMPUTE STANDARDIZED PDF

# purtroppo bisogna riscrivere da capo i bins perché
# ho perso l'informazione sui dx. Si potrebbe approssimare
v_bin = stats.create_log_bins(-12,12,600,1.e-2) #VAR
a_bin = stats.create_log_bins(-6,6,600,5.e-4) #VAR
r_bin = stats.create_log_bins(-6,6,600,1.e-4) #VAR
v_for_norm = []
a_for_norm = []
r_for_norm = []
for jj in range(v_bin.shape[0]-1):
    v_for_norm.append(v_bin[jj+1]-v_bin[jj])
    a_for_norm.append(a_bin[jj+1]-a_bin[jj])
    r_for_norm.append(r_bin[jj+1]-r_bin[jj])
v_for_norm = np.array(v_for_norm)
a_for_norm = np.array(a_for_norm)
r_for_norm = np.array(r_for_norm)

# from histogram to density
vg[1,:] = vg[1,:] / v_for_norm
ag[1,:] = ag[1,:] / a_for_norm
rg[1,:] = rg[1,:] / r_for_norm

v_norm = 0
a_norm = 0
r_norm = 0
for jj in range(vg.shape[1]):
    v_norm += vg[1,jj] * v_for_norm[jj]
    a_norm += ag[1,jj] * a_for_norm[jj]
    r_norm += rg[1,jj] * r_for_norm[jj]

# normalize
vg[1,:] = vg[1,:] / v_norm
ag[1,:] = ag[1,:] / a_norm
rg[1,:] = rg[1,:] / r_norm

# standardize
mean_v = 0.
std_v = 0.
mean_a = 0.
std_a = 0.
mean_r = 0.
std_r = 0.
for jj in range(vg.shape[1]):
    mean_v += v_for_norm[jj]*vg[1,jj]*vg[0,jj]
    mean_a += a_for_norm[jj]*ag[1,jj]*ag[0,jj]
    mean_r += r_for_norm[jj]*rg[1,jj]*rg[0,jj]
for jj in range(vg.shape[1]):
    std_v += v_for_norm[jj]*vg[1,jj]*(vg[0,jj]-mean_v)**2.
    std_a += a_for_norm[jj]*ag[1,jj]*(ag[0,jj]-mean_a)**2.
    std_r += r_for_norm[jj]*rg[1,jj]*(rg[0,jj]-mean_r)**2.

std_v = np.sqrt(std_v)
std_a = np.sqrt(std_a)
std_r = np.sqrt(std_r)

vg[0,:] = (vg[0,:] - mean_v) / std_v
vg[1,:] = vg[1,:] * std_v
ag[0,:] = (ag[0,:] - mean_a) / std_a
ag[1,:] = ag[1,:] * std_a
rg[0,:] = (rg[0,:] - mean_r) / std_r
rg[1,:] = rg[1,:] * std_r



# SAVE FILES

np.savetxt(save_file, np.stack((vg,ag,rg)).T)



# PLOT

if plot:
    op_gen = {'marker':'.','lw':0.4,'ms':5,'label':'GAN'}
    op_real = {'marker':'^','lw':0.4,'ms':7,'label':'DNS'}
    op_leg = {'ncol':1}

    plt.rcParams['font.size'] = 24
    #plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 17
    #plt.rcParams['figure.titlesize'] = 22
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams['figure.figsize'] = (17, 15)
    #plt.rcParams['lines.linewidth'] = 0.5
    #plt.rcParams['lines.markersize'] = 13
    plt.rcParams['legend.markerscale'] = 2
    #plt.rcParams['lines.marker'] = '.'

    fig, ax = plt.subplots(3,2, )
    ax[0,1].set_yscale('log')
    ax[1,1].set_yscale('log')
    ax[2,1].set_yscale('log')
    ax[0,0].set_xlabel('$v_x / \\sigma_{v_x}$')
    ax[0,1].set_xlabel('$v_x / \\sigma_{v_x}$')
    ax[1,0].set_xlabel('$a_x / \\sigma_{a_x}$')
    ax[1,1].set_xlabel('$a_x / \\sigma_{a_x}$')
    ax[2,0].set_xlabel('$r_x / \\sigma_{r_x}$')
    ax[2,1].set_xlabel('$r_x / \\sigma_{r_x}$')
    ax[0,0].set_ylabel('PDF$(v_x)\\cdot\\sigma_{v_x}$')
    ax[1,0].set_ylabel('PDF$(a_x)\\cdot\\sigma_{a_x}$')
    ax[2,0].set_ylabel('PDF$(r_x)\\cdot\\sigma_{r_x}$')
    ax[0,0].set_xlim([-4,4])
    ax[1,0].set_xlim([-4,4])
    ax[2,0].set_xlim([-4,4])

    # plot
    ax[0,0].plot(*(vr), **op_real)
    ax[0,1].plot(*(vr), **op_real)
    ax[0,0].plot(*(vg), **op_gen)
    ax[0,1].plot(*(vg), **op_gen)

    ax[1,0].plot(*(ar), **op_real)
    ax[1,1].plot(*(ar), **op_real)
    ax[1,0].plot(*(ag), **op_gen)
    ax[1,1].plot(*(ag), **op_gen)

    ax[2,0].plot(*(rr), **op_real)
    ax[2,1].plot(*(rr), **op_real)
    ax[2,0].plot(*(rg), **op_gen)
    ax[2,1].plot(*(rg), **op_gen)


    # legend
    ax[0,0].legend(**op_leg)
    ax[0,1].legend(**op_leg)
    ax[1,0].legend(**op_leg)
    ax[1,1].legend(**op_leg)
    ax[2,0].legend(**op_leg)
    ax[2,1].legend(**op_leg)

    # save figure
    fig.tight_layout()
    fig.savefig(write_path, fmt='png', dpi=60)


