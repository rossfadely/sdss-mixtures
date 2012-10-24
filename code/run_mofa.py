
import numpy as np
import pyfits as pf
import time
import cPickle

from patch import *
from sdss_mixtures_utils import *

import mofa

def make_data_arr(N):

    base = '/home/rfadely/sdss-mixtures/sdss_data/'
    fs = np.loadtxt(base+'fieldinfo.dat')

    count = 0
    i = 0
    while count < N:

        if i==fs.shape[0]:
            assert False, 'Not enough fields processed - '+\
            'need %d have %d' % (N,data.shape[0])

        filename = base+'rband_flipped_'+str(int(fs[i,0]))+\
            '_'+str(int(fs[i,2]))+'_'+str(int(fs[i,1]))+'.fits'

        f = pf.open(filename)
        d = f[0].data
        f.close()

        if count==0:
            data = d
        else:
            data = np.concatenate((data,d))
        
        count += d.shape[0]
        i += 1

    return data[:N,:]



m = 4

# number of K
for k in 2**(np.arange(1)+4):

    # size of data
    for n in 2**(np.arange(1)+18):
        N = 240000
        d = make_data_arr(N)
        
        print k,n,d.shape


        t0 = time.time()
        mix = mofa.Mofa(d,k,m,PPCA=False,init_ppca=True)
        imeans = mix.means.copy()
        mix.run_em()
        t = time.time() - t0
        mix.rs = 0
        mix.latents = 0
        mix.latent_covs = 0
        
        f = open('mofa_var(0.2-0.3)_2.18.16.4.pkl','wb')
        cPickle.dump(mix,f)
        f.close()
