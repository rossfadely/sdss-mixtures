import numpy as np
import pyfits as pf
import time
import mog
import cPickle

from patch import *
from sdss_mixtures_utils import *

def make_data_arr(N):

    NperR = 4*6

    run = [94,4469,5087,8055]
    field = [130,131,132,130,
             195,196,197,
             64,65,66,67,
             97,98,99,100]

    count = 0
    for r in run:
        for f in field:
            for c in np.arange(6)+1:
                filename = 'r_flipped_'+str(r)+'_'+str(f)+'_'+str(c)+'.fits'
                if count>=N:
                    pass
                else:
                    if count == 0:
                        try:
                            f = pf.open(filename)
                            d = f[0].data
                            f.close()
                            count += len(d[:,0])
                        except:
                            pass
                    else:
                        try:
                            f = pf.open(filename)
                            nd = f[0].data
                            d = np.concatenate((d,nd))
                            f.close()
                            count += len(nd[:,0])
                        except:
                            pass
    return d[:N,:]


for k in 2**(np.arange(4)+4):

    for n in 2**(np.arange(2)+16):

        if n>132000:
            d = make_data_arr(n)
            f = open('mix_'+str(k)+'_'+str(n)+'_imeans.pkl','rb')
            im = cPickle.load(f)
            f.close()
            f = open('mix_'+str(k)+'_'+str(n)+'_fmeans.pkl','rb')
            fm = cPickle.load(f)
            f.close()
            f = open('mix_'+str(k)+'_'+str(n)+'_imeans.pkl','rb')
            cov = cPickle.load(f)
            f.close()
            mix = mog.MixtureModel(k,d)
            mix.means = fm
            mix.cov = cov
            print mix.cov.shape
        else:
            f = open('mix_'+str(k)+'_'+str(n)+'.pkl','rb')
            mix = cPickle.load(f)
            f.close()
            d = mix._data
            print 'a',np.shape(mix.cov)
        fig_eigen(mix,mix.means,'mix_'+str(k)+'_'+str(n)+'_eigen')
        print '1'
        fig_patches(mix,8,'mix_'+str(k)+'_'+str(n)+'_draws')
        print '1'
        fig_patches(mix,d[:64,:],'mix_'+str(k)+'_'+str(n)+'_data')
        print '1'
        
