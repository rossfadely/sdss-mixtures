import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

r,f,c = 5087,96,5

base = '/home/rfadely/sdss-mixtures/sdss_data/'
tail = str(r)+'_'+str(f)+'_'+str(c)+'.fits'    
yflip = 'rband_flipped_'
nflip = 'rband_unflipped_'


f = pf.open(base+yflip+tail)
yflip = f[0].data
f.close()

f = pf.open(base+nflip+tail)
nflip = f[0].data
f.close()



ind = np.random.permutation(yflip.shape[0])
for ii in range(2):

    patch = yflip[ind[ii],:]
    pl.figure
    pl.gray()
    pl.imshow(patch.reshape((8,8)),
              interpolation='nearest',origin='lower',
              vmin = np.min(patch)*1.001,
              vmax = np.max(patch)*0.999)
    pl.colorbar()
    pl.show()

    patch = nflip[ind[ii],:]
    pl.figure
    pl.gray()
    pl.imshow(patch.reshape((8,8)),
              interpolation='nearest',origin='lower',
              vmin = np.min(patch)*1.001,
              vmax = np.max(patch)*0.999)
    pl.colorbar()
    pl.show()
