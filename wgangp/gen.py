#!/usr/bin/env python
# coding: utf-8

from db_utils import *

def build_generator(fs, fm, init_sigma, init_mean, alpha, noise_dim):
    """
    fs = (100,1) dimensione filtro
    fm = 128 numero di filtri
    init_sigma = 0.2 varianza distribuzione normale per l'inizializzazione
        dei pesi del  modello
    init_mean = 0.03 media distribuzione normale per l'inizializzazione
        dei pesi del  modello
    alpha = 0.3 pendenza parte negativa del leaky relu
    """
    ####kernel_size=21
    ####def gaussiana(x, m=0., std=1.):
    ####    return (1./np.sqrt(2.*np.pi*std**2.))*np.exp(-(x-m)**2./(2.*std**2.))
    ####
    ####def prepare_gaussian_w(kernel_size, input_channels):
    ####    x = np.zeros(shape=(kernel_size, 1, input_channels, 1))
    ####    for i in range(input_channels):
    ####        x[:,0,i,0] = np.arange(-(kernel_size-1)//2, (kernel_size+1)//2, 1)
    ####        #print(np.arange(-(kernel_size-1)//2, (kernel_size+1)//2, 1))
    ####    return gaussiana(x, m=0, std=3)
    ####
    ####def gaussian_layer(kernel_size):
    ####    return DepthwiseConv2D((kernel_size,1), use_bias=False, padding='same')
    ####g_layer_16ch = gaussian_layer(kernel_size)
    ####g_layer_8ch  = gaussian_layer(kernel_size)

    reg = l2(l=0.001)
    #reg = l1(l=0.001)
    generator = Sequential()
    generator.add(Dense(25*fm, kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma),
        input_dim=noise_dim))
    generator.add(ReLU())
    generator.add(Reshape((25, 1, fm)))
    #25x128
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Conv2DTranspose(fm//2, fs, strides=(5,1), padding='same',
        kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(ReLU())
    #125x64
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Conv2DTranspose(fm//4, fs, strides=(2,1), padding='same',
        kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(ReLU())
    generator.add(BatchNormalization(momentum=0.8))
    #250x32
    generator.add(Conv2DTranspose(fm//8, fs, strides=(2,1), padding='same',
        kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(ReLU())
    #500x16
    ####generator.add(g_layer_16ch) #Smoothing kernel not trainable
    #500x16
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Conv2DTranspose(fm//16, fs, strides=(2,1), padding='same',
        kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(ReLU())
    generator.add(BatchNormalization(momentum=0.8))
    #1000x8
    ####generator.add(g_layer_8ch) #Smoothing kernel not trainable
    #1000x8
    generator.add(Conv2DTranspose(CHANNELS, fs, strides=(2,1), padding='same',
        kernel_regularizer=reg, bias_regularizer=reg,
        kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(Activation("tanh"))

    # convolutional layer with a single filter that could perform the
    # denoise of the signal.
    # generator.add(Conv2DTranspose(1, (10,1), padding='same',
    #     kernel_initializer=RandomNormal(init_mean, init_sigma)))

    #2000xCHANNELS
    generator.add(Reshape((2000, CHANNELS)))


    ####kw = prepare_gaussian_w(kernel_size, 16)
    #####for i in range(8):
    #####    print(kw[:,0,i,0])
    ####g_layer_16ch.set_weights([kw])
    ####g_layer_16ch.trainable = False
    ####
    ####kw = prepare_gaussian_w(kernel_size, 8)
    #####for i in range(8):
    #####    print(kw[:,0,i,0])
    ####g_layer_8ch.set_weights([kw])
    ####g_layer_8ch.trainable = False

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
