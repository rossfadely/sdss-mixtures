import pyfits as pf
import scipy as sp

from sdss_mixtures_utils import *
from patch import *

run = [8055]
field = [97,98,99,100]
camcol = [1,2,3,4,5,6]


for r in run:
    for f in field:
        for c in camcol:
            filename = 'r_flipped_'+str(r)+'_'+str(f)+'_'+str(c)+'.fits'

            try:
                f = pf.open(filename)
                f.close()

            except:
                
                try:
                    d , i = get_sdss_data(r,c,f)
                    obj = Patches(d,i,var_lim=(0.2,0.3),flip=True) # magic numbers 2,3
                    
                    print 'successfully read r,c,f',r,c,f
                    hdu = pf.PrimaryHDU(obj.data)
                    hdu.writeto(filename)
                    assert False
                except:
                    pass
