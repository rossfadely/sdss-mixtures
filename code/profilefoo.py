
"""
python -m cProfile -o fooprofile profilefoo.py
python
import pstats
p = pstats.Stats('fooprofile')
p.sort_stats('time').print_stats(10)

"""

import numpy as np
import pyfits as pf
import time
import cPickle
import cProfile

from patch import *
from sdss_mixtures_utils import *
from mofa import *



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

m = 4

# number of K
for k in 2**(np.arange(1)+1):

    # size of data
    for n in 2**(np.arange(1)+16):

        d = make_data_arr(n)
        
        print k,n,d.shape

        for seed in range(100):
            np.random.seed(1)
            print seed
            t0 = time.time()
            mix = Mofa(d,k,m,True)
        #mix = mog(k,d)
            imeans = mix.means.copy()
        
            for zz in range(3):

                mix.take_EM_step()



        t = time.time() - t0

        out = open('mofamix_'+str(k)+'_'+str(n)+'_imeans.pkl','wb')
        cPickle.dump(imeans,out)
        out = open('mofamix_'+str(k)+'_'+str(n)+'_fmeans.pkl','wb')
        cPickle.dump(mix.means,out)
        out = open('mofamix_'+str(k)+'_'+str(n)+'_cov.pkl','wb')
        cPickle.dump(mix.covs,out)

        print 'took t = ',t
        try:
            res = np.empty((1,3))
            res[0,:] = np.array([k,n,t])
            timing = np.atleast_2d(np.loadtxt('results.dat'))
            timing = np.concatenate((timing,res),axis=0)
            np.savetxt('mofaresults.dat',timing)
        except:
            res = np.empty((1,3))
            res[0,:] = np.array([k,n,t])
            np.savetxt('mofaresults.dat',res)


