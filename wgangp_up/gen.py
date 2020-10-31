#!/usr/bin/env python
# coding: utf-8

from db_utils import *

def build_generator(fs, fm, init_sigma, init_mean, alpha, noise_dim):
    """
    fs = (20,1) dimensione filtro
    fm = 4 numero di filtri
    init_sigma = 0.2 varianza distribuzione normale per l'inizializzazione
        dei pesi del  modello
    init_mean = 0.03 media distribuzione normale per l'inizializzazione
        dei pesi del  modello
    alpha = 0.3 pendenza parte negativa del leaky relu
    """

    reg = l2(l=0.001)
    generator = Sequential()
    # Starting size
    generator.add(Dense(25*fm, kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma),
        input_dim=noise_dim))
    #generator.add(ELU())
    generator.add(ReLU())
    #20x1
    generator.add(Reshape((25, fm)))
    #5x4
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Conv1D(fm//2, fs, padding='same',
        kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma)))
    #generator.add(ELU())
    generator.add(ReLU())
    generator.add(UpSampling1D(size=5))
    #50x4
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Conv1D(fm//4, fs, padding='same',
        kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma)))
    #generator.add(ELU())
    generator.add(ReLU())
    generator.add(UpSampling1D(size=2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Conv1D(fm//8, fs, padding='same',
        kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma)))
    #generator.add(ELU())
    generator.add(ReLU())
    generator.add(UpSampling1D(size=2))
    #50x4
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Conv1D(fm//16, fs, padding='same',
        kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma)))
    #generator.add(ELU())
    generator.add(ReLU())
    generator.add(UpSampling1D(size=2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Conv1D(CHANNELS, fs, padding='same',
        kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(UpSampling1D(size=2))
    generator.add(Conv1D(CHANNELS, fs, padding='same',
        kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(Activation("tanh"))

    # convolutional layer with a single filter that could perform the
    # denoise of the signal.
    # generator.add(Conv2DTranspose(1, (10,1), padding='same',
    #     kernel_initializer=RandomNormal(init_mean, init_sigma)))

    generator.summary()


    return generator



if __name__ == '__main__':
    fs = (100,1)
    fm = 128
    init_sigma = 0.02
    init_mean = 0.01
    alpha = 0.3
    noise_dim = 100
    gen = build_generator(fs, fm, init_sigma, init_mean, alpha, noise_dim)
