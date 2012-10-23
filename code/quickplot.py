
import numpy as np
import matplotlib.pylab as pl
from mofa import *
from sdss_mixtures_utils import *
import cPickle
import pyfits as pf

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


f = open('mofamix_8_262144_imeans.pkl','rb')
im = cPickle.load(f)
f.close()

f = open('mofamix_8_262144_fmeans.pkl','rb')
fm = cPickle.load(f)
f.close()

f = open('mofamix_8_262144_cov.pkl','rb')
cov = cPickle.load(f)
f.close()

d = make_data_arr(2**18)

mix = Mofa(d,8,4)
mix.means = im
mix.cov = cov

fig_eigen(mix,im,'ppca_8_2.18_eigen.pdf')
fig_patches(mix,8,'ppca_8_2.18_draws.pdf')
