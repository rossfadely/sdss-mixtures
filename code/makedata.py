import numpy as np
import pyfits as pf
import scipy as sp

import sdss #Tractor import

from sdss_mixtures_utils import *
from patch import *

np.random.seed(100)

def get_sdss_data(run,camcol,field):
    """Call Tractor functions to get data, invvar images
    of a given SDSS field"""

    d = sdss.get_tractor_image_dr9(run,camcol,field,'r',psf='dg')
    d = d[0]

    return d.data,d.invvar


def make_patches(r,c,f):
    base = '/home/rfadely/sdss-mixtures/sdss_data/'
    tail = str(r)+'_'+str(f)+'_'+str(c)+'.fits'
    
    yflip = 'rband_flipped_'
    nflip = 'rband_unflipped_'

    try:
        f = pf.open(base+yflip+tail)
        N = f[0].data.shape[0]
        f.close()
        return N

    except:
        d , i = get_sdss_data(r,c,f)
        obj = Patches(d,i,var_lim=(0.2,0.3), # magic numbers 2,3
                      flip=True,
                      save_unflipped=True) 
        # write flipped patches
        hdu = pf.PrimaryHDU(obj.data)
        hdu.writeto(base+yflip+tail)
        # write unflipped patches
        hdu = pf.PrimaryHDU(obj.unflipped)
        hdu.writeto(base+nflip+tail)
        
        return obj.data.shape[0]




# get field list
base = '/home/rfadely/sdss-mixtures/sdss_data/'
f = pf.open(base+'dr9fields.fits')
fields = f[1].data
f.close()

# cut on good scores
ind = fields.field('score') > 0.80
fields = fields[ind]

# truffle shuffle
ind = np.random.permutation(len(fields.field(0)))
fields = fields[ind]


Ndata = 2**18

N = 0
i = 0
while N < Ndata:
    r,c,f = fields[i].field('run'), \
        fields[i].field('camcol'), \
        fields[i].field('field')

    if i==0:
        out = np.array([[r,c,f]])
    else:
        out = np.concatenate((out,np.array([[r,c,f]])),axis=0)
        np.savetxt(base+'fieldinfo.dat',out)

    N += make_patches(r,c,f)
    i += 1
    print '\nsuccess %d patches in iter %d\n' % (N,i)

