import pyfits as pf
import scipy as sp

from sdss_mixtures_utils import *
from patch import *

run = [8055] # this should read from the sdss run list
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
                    td = Patches(d)
                    ti = Patches(i,flip=False)

                    ind = ti.data == 0
                    ind = np.any(ind,axis=1)
                    ind = ind==0

                    td = td.data[ind]
                    tv = np.var(td,axis=1)
        
                    ind = np.argsort(tv)
                    tv  = tv[ind]
                    td  = td[ind]

                    Nhi = -(len(td))*0.01
                    Nlo = -0.05 * Nhi 

                    ind = np.random.permutation(len(td[:Nhi,0]))
                    ind = ind[:Nlo]

                    print 'var',tv[Nhi]
                    assert False

                    out = np.concatenate((td[ind],td[Nhi:]),axis=0)
                    out = out[np.random.permutation(len(out[:,0])),:]
                
                    hdu = pf.PrimaryHDU(out)
                    hdu.writeto(filename)

                except:
                    pass
