import numpy as np
import pyfits as pf
import time
import cPickle

from patch import *
from sdss_mixtures_utils import *

import mofa

def make_data_arr():

    run = [8055]
    field = [97,98,99,100]
    camcol = [1,2,3,4,5,6]

    count = 0
    for r in run:
        for f in field:
            for c in np.arange(6)+1:
                filename = 'r_flipped_'+str(r)+'_'+str(f)+'_'+str(c)+'.fits'
                if count>=np.Inf:
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
    return d

m = 4

# number of K
for k in 2**(np.arange(1)+3):

    # size of data
    for n in np.arange(1):

        d = make_data_arr()
        
        print k,n,d.shape


        t0 = time.time()
        mix = mofa.Mofa(d,k,m,PPCA=False,init_ppca=True)
        imeans = mix.means.copy()
        mix.run_em()
        t = time.time() - t0
        
        fig_mofa(mix,imeans,'foo.pdf')
        
