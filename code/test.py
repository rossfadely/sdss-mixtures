from sdss_mog import *


t0 = time.time()            

run,camcol,field = 5087,3,65
filename = 'hivar_patches.fits'


try:
    f = pf.open(filename)
    dpatch = f[0].data
    f.close()

except:


    t = time.time() - t0
    print 'Patch creation time: %0f' % t
    t0 = time.time()            

    var = dpatch.std(axis=1)

    t = time.time() - t0
    print 'Variance compute time: %0f' % t
    t0 = time.time()            

    lov = ipatch.min(axis=1)

    t = time.time() - t0
    print 'Minimum compute time: %0f' % t
    t0 = time.time()            

    ind = (var>0.04) & (lov>0) 
    dpatch = dpatch[ind]

    t = time.time() - t0
    print 'Reduction compute time: %0f' % t
    t0 = time.time()            

    hdu = pf.PrimaryHDU(dpatch)
    hdu.writeto(filename)

mixture = mog.MixtureModel(3, dpatch)
mixture.run_kmeans()
kmeans = mixture.means
mixture.run_em()


fig_eigen(mixture,kmeans,'foooo')
fig_patches(mixture,dpatch,'datafoo')
fig_patches(mixture,8,'drawfoo')

