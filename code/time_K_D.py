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


# number of K
for k in 2**(np.arange(8)):

    # size of data
    for n in 2**(np.arange(4)+16):

        d = make_data_arr(n)
        
        print k,n,d.shape


        t0 = time.time()
        mix = mog.MixtureModel(k,d)
        mix.run_kmeans()
        imeans = mix.means.copy()
        mix.run_em()
        t = time.time() - t0

        out = open('mix_'+str(k)+'_'+str(n)+'_imeans.pkl','wb')
        cPickle.dump(imeans,out)
        out = open('mix_'+str(k)+'_'+str(n)+'_fmeans.pkl','wb')
        cPickle.dump(mix.means,out)
        out = open('mix_'+str(k)+'_'+str(n)+'_cov.pkl','wb')
        cPickle.dump(mix.cov,out)

        print 'took t = ',t
        try:
            res = np.empty((1,3))
            res[0,:] = np.array([k,n,t])
            timing = np.atleast_2d(np.loadtxt('results.dat'))
            timing = np.concatenate((timing,res),axis=0)
            np.savetxt('results.dat',timing)
        except:
            res = np.empty((1,3))
            res[0,:] = np.array([k,n,t])
            np.savetxt('results.dat',res)
