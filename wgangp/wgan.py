#!/usr/bin/env python
# coding: utf-8


# IMPORT MODULES

from db_utils import *


class RandomWeightedAverage(tensorflow.keras.layers.Layer):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size


    def call(self, inputs, **kwargs):
        alpha = tensorflow.random_uniform((self.batch_size, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


    def compute_output_shape(self, input_shape):
        return input_shape[0]


class WGANGP():

    def __init__(self,gen,critic,noise_dim,n_critic,batch_size,gen_lr,critic_lr,text,set_run):
        self.sig_len = SIG_LEN
        self.channels = CHANNELS
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.text = text

        # Creo cartella della run attuale:
        if set_run == False:
            self.current_run = self.get_run()
            self.dir_path = f"runs/{self.current_run}/"
            print(f"\nCreating new directories for run: {self.current_run}\n")
            os.mkdir(self.dir_path)
        elif type(set_run) == type(2):
            self.current_run = set_run
            self.dir_path = f"runs/{self.current_run}/"
            print(f"\nCreating new directories for run: {self.current_run}\n")
            try: os.mkdir(self.dir_path)
            except FileExistsError:
                print('Selected run already exists, Exiting.')
                exit()


        # Following parameter and optimizer set as recommended in paper
        self.n_critic = n_critic
        #self.gen_lr = self.critic_lr = 0.00001
        self.gen_lr = gen_lr
        self.critic_lr = critic_lr
        self.gen_b1 = self.critic_b1 = 0.0 # wgangp article: 0.0
        self.gen_b2 = self.critic_b2 = 0.9 # wgangp article: 0.9
        gen_optimizer = Adam(learning_rate=self.gen_lr,
                             beta_1=self.gen_b1,beta_2=self.gen_b2)
        critic_optimizer = Adam(learning_rate=self.critic_lr,
                                beta_1=self.critic_b1,beta_2=self.critic_b2)

        self.critic = critic
        self.gen = gen

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.gen.trainable = False

        # Image input (real sample)
        real_img = Input(shape=(self.sig_len,self.channels))

        # Noise input
        z_disc = Input(shape=(self.noise_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.gen(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage(
                self.batch_size)([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                                  outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=critic_optimizer,
                                  loss_weights=[1, 1, 10])
        print(self.critic_model.summary())


        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.gen.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.noise_dim,))
        # Generate images based of noise
        img = self.gen(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.gen_model = Model(z_gen, valid)
        self.gen_model.compile(loss=self.wasserstein_loss, optimizer=gen_optimizer)
        print(self.gen_model.summary())


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on
        prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def gen_noise(self, n_samples):
        yy = np.random.standard_t(4, size=(n_samples, self.noise_dim))
        # yy = np.random.standard_t(3.5, size=(n_samples, self.noise_dim))
        # yy = np.random.normal(0,1, size=(n_samples, self.noise_dim))
        return yy


    def get_run(self):
        runs = glob.glob('runs/*/')
        longest_string = max(runs, key=len)

        print(longest_string)
        if len(longest_string) > 8 :
            #TODO
            print("Attention: more than 99 runs are not supported. Exiting.")
            exit()

        runs = sorted(runs, key=lambda x: int(x[5:-1]))
        runs = [int(line[5:-1]) for line in runs]

        if len(runs)>0: return runs[-1] + 1
        else: return 1


    def plot_trajs(self, gen_trajs, epoch):
        plt.figure(figsize=(13, 2*len(gen_trajs)), dpi=60)
        for i, traj in enumerate(gen_trajs):
            plt.subplot(len(gen_trajs), 1, i+1)
            plt.plot(traj)
        plt.tight_layout()
        plt.savefig(self.dir_path+f'{epoch}_gen_traj.png', fmt='png', dpi=60)
        plt.close()


    def compute_pdf_real(self, db_real):
        #Remove Borders (not for real data)
        velo = db_real[:,:,COMPONENTS] # WE WITHOUT EXTREMES

        #Renorm dataset
        semidisp = (DB_MAX-DB_MIN)/2.
        media = (DB_MAX+DB_MIN)/2.
        velo = velo * semidisp + media

        #Compute first der
        acce = np.gradient(velo,axis=1)

        #Compute PDF
        nbins=600
        pdf_ve_real, bin_ve_real = np.histogram(velo.flatten(), nbins, (-12,12), density=True)
        pdf_ac_real, bin_ac_real = np.histogram(acce.flatten(), nbins, (-6,6),   density=True)

        return pdf_ve_real, pdf_ac_real


    def compute_local_slopes(self, sfg, sfr):
        # COMPUTE LOG DERIVATIVES

        dlg       = np.zeros(shape=sfg.shape)
        dlg[:,0]  = sfg[:,0]
        dlg[:,1:] =  np.gradient( np.log(sfg[:,1:]), np.log(sfg[:,0]), axis=0 )

        dlr       = np.zeros(shape=sfr.shape)
        dlr[:,0]  = sfr[:,0]
        dlr[:,1:] = np.gradient( np.log(sfr[:,1:]), np.log(sfr[:,0]), axis=0 )
    
        # COMPUTE LOG DERIVATIVES ESS

        dlg_ess       = np.zeros(shape=(dlg.shape[0], dlg.shape[1]-1))
        dlg_ess[:,0]  = dlg[:,0]
        for ii in range(1, sfg.shape[1]-1):
            dlg_ess[:,ii] = dlg[:,ii+1] / dlg[:,1]
            
        dlr_ess       = np.zeros(shape=(dlr.shape[0], dlr.shape[1]-1))
        dlr_ess[:,0]  = dlr[:,0]
        for ii in range(1, sfr.shape[1]-1):
            dlr_ess[:,ii] = dlr[:,ii+1] / dlr[:,1]

        return dlg, dlr, dlg_ess, dlr_ess


    def plot_statistics(self, mini_db, epoch, pdf_ve_real, pdf_ac_real):
        #Remove Borders
        velo = mini_db[:,100:1900,:] # WE WITHOUT EXTREMES
        #Compute first der
        acce = np.gradient(velo,axis=1)

        #Compute PDF
        nbins=600
        pdf_ve_gen, bins_ve = np.histogram(velo.flatten(), nbins, (-12,12), density=True)
        pdf_ac_gen, bins_ac = np.histogram(acce.flatten(), nbins, (-6,6),   density=True)

        #Compute SF
        sf_gen = stats.compute_sf(velo)

        #Read SF real
        read_file_real = '../data/real/sf.dat'
        sf_real = np.loadtxt(read_file_real)

        #Compute LS
        ls_gen, ls_real, ess_gen, ess_real = self.compute_local_slopes(sf_gen, sf_real)

        #Plot PDF
        op_gen = {'marker':'.','lw':0.4,'ms':5,'label':'GAN'}
        op_real = {'marker':'^','lw':0.4,'ms':7,'label':'DNS'}
        op_leg = {'ncol':1}

        plt.rcParams['font.size'] = 24
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['legend.fontsize'] = 17
        plt.rcParams['figure.dpi'] = 60
        plt.rcParams['figure.figsize'] = (15, 6)
        plt.rcParams['legend.markerscale'] = 2

        fig, ax = plt.subplots(1,2, )
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[0].set_xlabel('$v_x$')
        ax[1].set_xlabel('$a_x$')
        ax[0].set_ylabel('PDF$(v_x)$')
        ax[1].set_ylabel('PDF$(a_x)$')
        #ax[0,0].set_xlim([-4,4])
        #ax[0,1].set_xlim([-4,4])

        ax[0].plot(bins_ve[:-1], pdf_ve_gen, **op_gen)
        ax[1].plot(bins_ac[:-1], pdf_ac_gen, **op_gen)
        ax[0].plot(bins_ve[:-1], pdf_ve_real, **op_real)
        ax[1].plot(bins_ac[:-1], pdf_ac_real, **op_real)

        ax[0].legend(**op_leg)
        ax[1].legend(**op_leg)

        fig.tight_layout()
        fig.savefig(self.dir_path+f'{epoch}_pdfs.png', fmt='png', dpi=60)

        #Define Figure for SF
        op_gen = {'marker':'.','lw':0.4,'ms':25,'markeredgewidth':1, 'markeredgecolor':"black"}
        op_real = {'marker':'^','lw':0.4,'ms':14}
        op_leg = {'markerscale':1, 'ncol':2}
        op_leg_1 = {'markerscale':1, 'ncol':3}

        plt.rcParams['figure.figsize'] = (17, 12)

        fig, ax = plt.subplots(2,2)

        ax[0,0].set_yscale('log')
        ax[0,0].set_xscale('log')
        ax[0,0].set_xlabel("$\\tau$")
        ax[0,0].set_ylabel('$S_n(\\tau)$')
        ax[0,1].set_xscale('log')
        ax[0,1].set_xlabel("$\\tau$")
        ax[0,1].set_ylabel("$F_n(\\tau)$")
        ax[0,1].set_yscale('log')
        ax[0,1].grid(which='minor', alpha=0.2)
        ax[0,1].grid(which='major', alpha=0.9)
        ax[1,0].set_xscale('log')
        ax[1,0].set_xlabel("$\\tau$")
        ax[1,0].set_ylabel("$\\xi_n(\\tau)$")
        ax[1,1].set_ylim([0,7.5])
        ax[1,1].set_xscale('log')
        ax[1,1].set_xlabel("$\\tau$")
        ax[1,1].set_ylabel("$\\xi_n(\\tau)/ \\xi_2(\\tau)$")

        # PLOT SF
        for ii in range(1,7):
            ax[0,0].plot(sf_real[:,0],sf_real[:,ii],label="DNS n="+str((ii)*2),
                         color='C'+str(ii-1),**op_real)
        for ii in range(1,7):
            ax[0,0].plot(sf_gen[:,0],sf_gen[:,ii],#label="GAN n="+str((ii)*2),
                         color='C'+str(ii-1),**op_gen)


        # PLOT FLATNESS
        ax[0,1].plot(sf_real[:,0], sf_real[:,2]/(sf_real[:,1]**2), label="DNS $F_4$",
                     color='C1',**op_real)
        ax[0,1].plot(sf_real[:,0], sf_real[:,3]/(sf_real[:,1]**3), label="DNS $F_6$",
                     color='C2',**op_real)

        ax[0,1].plot(sf_gen[:,0], sf_gen[:,2]/(sf_gen[:,1]**2),# label="GAN $F_4$",
                     color='C1',**op_gen)
        ax[0,1].plot(sf_gen[:,0], sf_gen[:,3]/(sf_gen[:,1]**3),# label="GAN $F_6$",
                     color='C2',**op_gen)


        # PLOT LOGARITMIC DERIVATIVES
        for ii in range(1,4):
            ax[1,0].plot(*(ls_real[:,[0,ii]].T), label="DNS n="+str((ii)*2),
                         color='C'+str(ii-1),**op_real)
        for ii in range(1,4):
            ax[1,0].plot(*(ls_gen[:,[0,ii]].T),# label="GAN n= "+str((ii)*2),
                         color='C'+str(ii-1),**op_gen)


        # PLOT CONSTANT VALUES FOR ESS REFERENCE
        ax[1,1].plot(ls_real[:,0], (lambda x: [4/2]*len(x))(ls_real[:,0]),ls='--',
                     color="C1")
        ax[1,1].plot(ls_real[:,0], (lambda x: [6/2]*len(x))(ls_real[:,0]),ls='--',
                     color="C2")

        # PLOT LOGARITMIC DERIVATIVES IN ESS
        for ii in range(2):
            ax[1,1].plot(ess_real[:,0], ess_real[:,ii+1],label="DNS n="+str((ii+2)*2),
                         color='C'+str(ii+1),**op_real)
        for ii in range(2):
            ax[1,1].plot(ess_gen[:,0], ess_gen[:,ii+1],#label="GAN n="+str((ii+2)*2),
                         color='C'+str(ii+1),**op_gen)


        ax[0,0].legend(**op_leg)
        ax[0,1].legend(**op_leg_1, loc='upper right')
        ax[1,0].legend(**op_leg)
        ax[1,1].legend(**op_leg)

        fig.tight_layout()
        fig.savefig(self.dir_path+f'{epoch}_sf.png', fmt='png', dpi=60)

        

    def train(self, gen_iters, db_train, db_test):


        # Compute PDF real dataset #
        pdf_ve_real, pdf_ac_real = self.compute_pdf_real(db_train)

        # salvo info log #
        with open(self.dir_path+'logs.txt','a+') as f:
            f.write((f"TRAINING\nbatch={self.batch_size}\nncritic={self.n_critic}\n"
                    f"gen_iterations={gen_iters}\ngen_lr={self.gen_lr} gen_b1={self.gen_b1}"
                    f" gen_b2={self.gen_b2}\ncritic_lr={self.critic_lr}"
                    f" critic_b1={self.critic_b1} critic_b2={self.critic_b2}\n"))
            f.write(self.text)
        log_fl = open(self.dir_path+'logs.txt','a+')
        fl = open(self.dir_path+'training.dat','a+')
        fl.write((f"# gen_iteration, critic_iteration, d_loss_tot, d_loss_true,"
                 f" d_loss_fake, d_loss_gp, d_loss_tot_test, d_loss_true_test,"
                 f" d_loss_fake_test, d_loss_gp_test, g_loss\n"))
        fl.close()

        # ############## #

        d_loss_test = [0., 0., 0., 0.]
        g_loss = 0.
        semidisp = (DB_MAX-DB_MIN)/2.
        media = (DB_MAX+DB_MIN)/2.

        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy gt for gradient penalty

        for gen_iter in range(gen_iters + 1):

            fl = open(self.dir_path+'training.dat','a+')
            for jj in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, db_train.shape[0], self.batch_size)
                # select a precise batch of images
                # batch_start = (gen_iter * self.n_critic + jj) * self.batch_size
                # batch_end = (gen_iter * self.n_critic + jj + 1) * self.batch_size
                # idx = np.arange(batch_start, batch_end)
                imgs = db_train[idx]
                # Sample generator input
                noise = self.gen_noise(self.batch_size)
                # Train the critic
                # print("init disc train") 
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                           [valid, fake, dummy])
                # print("disc trained")
                if (jj!=self.n_critic-1):
                        fl.write(("%7d %7d %11.4e %11.4e %11.4e %11.4e %11.4e"
                                  " %11.4e %11.4e %11.4e %11.4e\n")%(gen_iter,
                                  gen_iter*self.n_critic+jj,d_loss[0],d_loss[1],
                                  d_loss[2],d_loss[3],d_loss_test[0],
                                  d_loss_test[1],d_loss_test[2],d_loss_test[3],
                                  g_loss))

            # ---------------------
            #  Train Generator
            # ---------------------

            #idx = np.random.randint(0, db_test.shape[0], self.batch_size)
            #imgs = db_test[idx]
            #d_loss_test = self.critic_model.test_on_batch([imgs, noise],
            #                                              [valid, fake, dummy])
            noise = self.gen_noise(self.batch_size)
            g_loss = self.gen_model.train_on_batch(noise, valid)

            print((f"Gen_Iter: {gen_iter:6d} [D loss: {d_loss[0]:9.2e}] "
                  f"[d_loss_test: {d_loss_test[0]:9.2e}] "
                  f"[G loss: {g_loss:9.2e}]"))
            fl.write(("%7d %7d %11.4e %11.4e %11.4e %11.4e %11.4e "
                     "%11.4e %11.4e %11.4e %11.4e\n")%(
                    gen_iter,gen_iter*self.n_critic+jj,d_loss[0],d_loss[1],
                    d_loss[2],d_loss[3],d_loss_test[0],d_loss_test[1],
                    d_loss_test[2],d_loss_test[3],g_loss))
            fl.close()

            # SAVE IMGS, MODELS OR MINIDB AND COMPUTE STAT. QUANTITIES

            if gen_iter % 100 == 0:
                self.plot_trajs(self.gen.predict(self.gen_noise(4)),
                                gen_iter)

            if gen_iter % 250 == 0:
                self.critic.save(self.dir_path+f'{gen_iter}_critic.h5')
                self.gen.save(self.dir_path+f'{gen_iter}_gen.h5')

            if gen_iter % 1000 == 0:
                mini_db = self.gen.predict(self.gen_noise(50000))
                mini_db = mini_db * semidisp + media
                #np.save(self.dir_path+f'gen_trajs_{gen_iter}', mini_db)
                #del mini_db
                self.plot_statistics(mini_db, gen_iter, pdf_ve_real, pdf_ac_real)
                #command_line = ["python", "graphics/plot_pdfs.py",
                #        str(self.current_run), str(gen_iter), "--scratch"]
                #subprocess.Popen(command_line, stdout=log_fl, stderr=log_fl)
                #command_line = ["python", "graphics/plot_sf.py",
                #        str(self.current_run), str(gen_iter), "--scratch"]
                #subprocess.Popen(command_line, stdout=log_fl, stderr=log_fl)

        log_fl.close()


if __name__ == '__main__' :

    # ARGUMENTS AND OPTIONS PARSING

    import sys

    option_gen_iters = False
    gen_iters = 20000
    option_ncritic = False
    ncritic = 5
    option_load = False
    load = [0, 0]
    option_batch_size = False
    batch_size = 500
    option_gen_lr = False
    gen_lr = 0.00005
    option_critic_lr = False
    critic_lr = 0.0001
    set_run = False

    while len(sys.argv) > 1:
        if sys.argv[1] == "-h":
            print((f"usage: wgan.py [--gen_iters <{gen_iters}>]"
                   f" [--ncritic <{ncritic}>] [--load <null> <null>]"
                   f" [--batch_size <{batch_size}>] [--gen_lr <{gen_lr}>]"
                   f" [--critic_lr <{critic_lr}>] [--set_run <run>]"))
            exit()
        elif sys.argv[1] == "--gen_iters":
            option_gen_iters = True
            sys.argv.pop(1)
            gen_iters = int(sys.argv.pop(1))
        elif sys.argv[1] == "--ncritic":
            option_ncritic = True
            sys.argv.pop(1)
            ncritic = int(sys.argv.pop(1))
        elif sys.argv[1] == "--set_run":
            _ = sys.argv.pop(1)
            set_run = int(sys.argv.pop(1))
        elif sys.argv[1] == "--load":
            option_load = True
            sys.argv.pop(1)
            load[0] = int(sys.argv.pop(1))
            load[1] = int(sys.argv.pop(1))
        elif sys.argv[1] == "--batch_size":
            option_batch_size = True
            sys.argv.pop(1)
            batch_size = int(sys.argv.pop(1))
        elif sys.argv[1] == "--gen_lr":
            option_gen_lr = True
            sys.argv.pop(1)
            gen_lr = float(sys.argv.pop(1))
        elif sys.argv[1] == "--critic_lr":
            option_critic_lr = True
            sys.argv.pop(1)
            critic_lr = float(sys.argv.pop(1))
        else:
            print("arg not recognized, exiting ...")
            exit()


    # LOADING OR CREATING MODELS

    if option_load:

        run = load[0]
        number = load[1]
        path_gen = f'runs/{run}/{number}_gen.h5'
        path_critic = f'runs/{run}/{number}_critic.h5'
        gen = load_model(path_gen)
        critic = load_model(path_critic)
        critic.trainable = True
        # scrivo stringa info log gen #
        text = f"\nModels loaded. Continuing run {run} from number {number}\n"

    else:

        from gen import *
        from critic import *

        fs=(100,1)
        fm=128
        init_sigma = 0.003
        init_mean = 0.0
        alpha = 0.2
        # scrivo stringa info log gen #
        text = f"GEN\n{fs,fm,init_sigma,init_mean,NOISE_DIM,alpha}\n"
        gen = build_generator(fs,fm,init_sigma,init_mean,alpha,NOISE_DIM)
        fs = 100
        fm = 128
        init_sigma = 0.02
        init_mean = 0.0
        alpha = 0.2
        critic = build_critic(fs,fm,init_sigma,init_mean,alpha)
        # scrivo stringa info log critic #
        text += f"CRITIC\n{fs,fm,init_sigma,init_mean,alpha}\n"


    # WGANGP INIT

    wgan = WGANGP(gen, critic, NOISE_DIM, ncritic, batch_size,
                  gen_lr, critic_lr, text, set_run)
    print((f'\nTrain for {gen_iters} generator iterations.\n'
           f"Parameters:\n    noise_dim: {NOISE_DIM}\n    "
           f"ncritic: {ncritic}\n    batch_size: {batch_size}\n    "
           f"gen_lr: {gen_lr}\n    critic_lr: {critic_lr}\n"))
    print(text)


    # DB IMPORT

    print("\nImporting Databases ... ")
    db_train, db_test = load_data(1.0)


    # TRAINING

    print("\nTraining ... \n",end="")
    wgan.train(gen_iters, db_train, db_test)
