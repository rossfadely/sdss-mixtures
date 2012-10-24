import numpy as np
import pyfits as pf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

import cPickle

from mofa_plotting import *

f = open('mofa_var(0.2-0.3)_2.18.8.4.pkl','rb')
mix = cPickle.load(f)
f.close()

#fig_mofa(mix,mix.means,'foo.pdf')

print 'done reading pickle'

L = 16
ind = np.random.permutation(mix.data.shape[0])
ind = ind[:L**2]

np.savetxt('patchinds.dat',ind)

fig_patches(L,8,'patches_0293.png',data=mix.data[ind])
fig_patches(L,8,'patches_9835.png',mix=mix)
