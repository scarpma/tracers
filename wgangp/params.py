#!/usr/bin/env python
# coding: utf-8

REAL_DB_PATH = ('/m100/home/userexternal/fbonacco/Michele/'
                'velocities.npy')
COMPONENTS = slice(0,1)
DB_NAME = "tracers"
WGAN_TYPE = "wgangp"

SIG_LEN = 2000
CHANNELS = 1
NOISE_DIM = 100

DB_MAX = 10.273698864467972
DB_MIN = -9.970374739869616

# Activate to smoothen training dataset
SMOOTH_REAL_DB = False
if SMOOTH_REAL_DB:
    sigma_smooth_real=2
    trunc_smooth_real=5
