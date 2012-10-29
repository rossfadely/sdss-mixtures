import numpy as np
import pyfits as pf
import scipy as sp

from sdss_img_utils import getimage
from patch import *

np.random.seed(100)

def get_sdss_data(run,camcol,field):
    """Call modified Tractor functions to get data, invvar images
    of a given SDSS field"""


    d,info = getimage.get_image_dr9(run,camcol,field,'r',psf='dg')
    if d==None:
        return None,None
    else:
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
        if d==None:
            return None
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
# this file lives in the sdss_data dir in git repo
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


Ndata = 2 * 2**18

N = 0
i = 0
while N < Ndata:
    r,c,f = fields[i].field('run'), \
        fields[i].field('camcol'), \
        fields[i].field('field')

    tN = make_patches(r,c,f)

    if tN!=None:
        if i==0:
            out = np.array([[r,c,f]])
        else:
            out = np.concatenate((out,np.array([[r,c,f]])),axis=0)
            np.savetxt(base+'fieldinfo_0.2-0.3.dat',out)
        N += tN
        print '\nsuccess %d patches in iter %d\n' % (tN,i)
        print 'now have %d patches in iter %d\n\n\n' % (N,i)

    i += 1
    

