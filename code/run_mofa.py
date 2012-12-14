
import numpy as np
import pyfits as pf
import time
import cPickle

from patch import *
from sdss_mixtures_utils import *

import mofa

def make_data_arr(N):

    base = '/home/rfadely/sdss-mixtures/sdss_data/'
    fs = np.loadtxt(base+'screwed_m1.64_fieldinfo_0.2-0.3.dat')

    count = 0
    i = 0
    while count < N:

        if i==fs.shape[0]:
            assert False, 'Not enough fields processed - '+\
            'need %d have %d' % (N,data.shape[0])

        filename = base+'screwed_m1.64_rband_flipped_'+str(int(fs[i,0]))+\
            '_'+str(int(fs[i,2]))+'_'+str(int(fs[i,1]))+'.fits'

        f = pf.open(filename)
        d = f[0].data
        f.close()

        if count==0:
            data = d
        else:
            data = np.concatenate((data,d))

        if i==0:
            out = np.atleast_2d(fs[i,:]).T
        else:
            out = np.atleast_2d(np.loadtxt(base+'screwed_m1.64trainfieldinfo_0.2-0.3.dat'))
            out = np.concatenate((out,np.atleast_2d(fs[i,:])))


        np.savetxt(base+'screwed_m1.64trainfieldinfo_0.2-0.3.dat',out)

        count += d.shape[0]
        i += 1

    return data[:N,:]



m = 4

# number of K
for k in 2**(np.arange(4)+2):

    # size of data
    for en in np.arange(1)+18:
        
        N = 2**en
        d = make_data_arr(N)
        
        print k,N,d.shape

        t0 = time.time()
        mix = mofa.Mofa(d,k,m,PPCA=True,init_ppca=True)
        imeans = mix.means.copy()
        mix.run_em()
        t = time.time() - t0
        mix.rs = 0
        mix.latents = 0
        mix.latent_covs = 0
        
        base = '/home/rfadely/sdss-mixtures/sdss_data/'
        f = open(base+'screwed_m1.64_mofa_var(0.2-0.3)_2.'+str(en)+'.'+str(k)+'.4.pkl','wb')
        cPickle.dump(mix,f)
        f.close()
