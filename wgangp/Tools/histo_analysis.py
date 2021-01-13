#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import os
import os.path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
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

save_path = f'runs/{run}/HighStat/TOT_{number}'
if not osp.exists(save_path):
    os.makedirs(save_path)

save_file = osp.join(save_path, 'pdf.dat')
save_figure = osp.join(save_path, 'pdf')



# FIND SF.DAT FILES

read_file_gene = f'runs/{run}/HighStat/pdf_{number}_me_*.dat'
read_file_real = ['../data/real/pdf0.dat', '../data/real/pdf1.dat']
read_file_gene = glob(read_file_gene)
print(f'found {len(read_file_gene)} sf.dat files:')
for file in read_file_gene:
    print('\t', file)



# READ ALREADY COMPUTED HIST
vr, vm, vstd = stats.make_hist(read_file_real[0], std=True, out=True)
ar, am, astd = stats.make_hist(read_file_real[1], std=True, out=True)

for ii, fl in enumerate(read_file_gene):

    temp = np.loadtxt(fl)

    if ii == 0:
        vg = temp[:,0:2]
        ag = temp[:,2:4]
    else:
        assert all(temp[:,0] == vg[:,0]), 'different bin for velo'
        assert all(temp[:,2] == ag[:,0]), 'different bin for acce'
        vg[:,1] = vg[:,1] + vg[:,1]
        ag[:,1] = ag[:,1] + ag[:,1]


# SUM HISTOS AND COMPUTE STANDARDIZED PDF

db_ve = vg[1,0] - vg[0,0]
db_ac = ag[1,0] - ag[0,0]

v_norm = np.sum(vg[:,1]) * db_ve
a_norm = np.sum(ag[:,1]) * db_ac

# normalize from histogram to density
vg[:,1] = vg[:,1] / v_norm
ag[:,1] = ag[:,1] / a_norm


# standardize
mean_v = 0.
std_v = 0.
mean_a = 0.
std_a = 0.
for jj in range(vg.shape[0]):
    mean_v += db_ve * vg[jj,1]*(vg[jj,0] + db_ve/2)
    mean_a += db_ac * ag[jj,1]*(ag[jj,0] + db_ac/2)
for jj in range(vg.shape[0]):
    std_v += db_ve * vg[jj,1]*((vg[jj,0] + db_ve/2) - mean_v)**2.
    std_a += db_ac * ag[jj,1]*((ag[jj,0] + db_ac/2) - mean_a)**2.

std_v = np.sqrt(std_v)
std_a = np.sqrt(std_a)

vg_std = vg
ag_std = ag

vg_std[:,0] = (vg_std[:,0] - mean_v) / std_v
vg_std[:,1] =  vg_std[:,1] * std_v
ag_std[:,0] = (ag_std[:,0] - mean_a) / std_a
ag_std[:,1] =  ag_std[:,1] * std_a



# SAVE FILES

np.savetxt(save_file, np.stack((vg[:,0],vg[:,1],
                                ag[:,0],ag[:,1],
                                vg_std[:,0],vg_std[:,1],
                                ag_std[:,0],ag_std[:,1])).T)



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
    plt.rcParams['figure.figsize'] = (17, 10)
    #plt.rcParams['lines.linewidth'] = 0.5
    #plt.rcParams['lines.markersize'] = 13
    plt.rcParams['legend.markerscale'] = 2
    #plt.rcParams['lines.marker'] = '.'

    fig, ax = plt.subplots(2,2, )
    ax[0,1].set_yscale('log')
    ax[1,1].set_yscale('log')
    ax[0,0].set_xlabel('$v_x / \\sigma_{v_x}$')
    ax[0,1].set_xlabel('$v_x / \\sigma_{v_x}$')
    ax[1,0].set_xlabel('$a_x / \\sigma_{a_x}$')
    ax[1,1].set_xlabel('$a_x / \\sigma_{a_x}$')
    ax[0,0].set_ylabel('PDF$(v_x)\\cdot\\sigma_{v_x}$')
    ax[1,0].set_ylabel('PDF$(a_x)\\cdot\\sigma_{a_x}$')
    ax[0,0].set_xlim([-4,4])
    ax[1,0].set_xlim([-4,4])

    # plot
    ax[0,0].plot(*(vr), **op_real)
    ax[0,1].plot(*(vr), **op_real)
    ax[0,0].plot(*(vg), **op_gen)
    ax[0,1].plot(*(vg), **op_gen)

    ax[1,0].plot(*(ar), **op_real)
    ax[1,1].plot(*(ar), **op_real)
    ax[1,0].plot(*(ag), **op_gen)
    ax[1,1].plot(*(ag), **op_gen)

    # legend
    ax[0,0].legend(**op_leg)
    ax[0,1].legend(**op_leg)
    ax[1,0].legend(**op_leg)
    ax[1,1].legend(**op_leg)

    # save figure
    fig.tight_layout()
    fig.savefig(save_figure, fmt='png', dpi=72)
