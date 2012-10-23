import numpy as np
import matplotlib.pyplot as pl

d = np.loadtxt('results.dat')


c = ['k','b','r','g']
fig=pl.figure()
for ii,dv in enumerate(2**(np.arange(4)+16)):
    ind = d[:,1]==dv
    pl.plot(d[ind,0],d[ind,2],'o',color=c[ii],alpha=0.5,label='%1.3e'%dv)

pl.legend()
pl.xlabel('Number of Components')
pl.ylabel('Wall clock time (s)')
fig.savefig('timing.pdf',format='pdf')
