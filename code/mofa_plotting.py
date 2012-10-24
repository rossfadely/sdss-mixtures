import numpy as np
import matplotlib.pyplot as pl
import pyfits as pf

from numpy.linalg import svd
from matplotlib.backends.backend_pdf import PdfPages
from patch import *

from matplotlib import rc
rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':10})
rc('text', usetex=True)



def init_fig(fig_side=8):
    """
    Initialize figure
    """
    L = fig_side
    factor = 2.0          # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.5 * factor  # size of top/right margin
    whspace = 0.05        # w/hspace size
    plotdim = factor * L + factor * (L - 1.) * whspace
    dim = lbdim + plotdim + trdim
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig = pl.figure(figsize=(dim, dim + L))
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace,hspace=0.0001)
    return fig

def fig_mofa_head(fig,ini_means,mix,k,L,pshape):
    """
    Make block of info, means, psis, lambdas that goes at top of 
    each figure.
    """
    # write some info
    ax = fig.add_subplot(L+2,L,3)
    ax.set_axis_off()
    ax.text(0.0,0.5,'Component = %d' % k,
            transform=ax.transAxes,ha='left',va='center',
            fontsize=35)
    ax = fig.add_subplot(L+2,L,6)
    ax.set_axis_off()
    ax.text(0.0,0.5,'Amp = %0.3f' % mix.amps[k],
            transform=ax.transAxes,ha='left',va='center',
            fontsize=35)

    # write mean patches
    fmpatch = mix.means[k].reshape(pshape)
    impatch = ini_means[k].reshape(pshape)
    ax = fig.add_subplot(L+1,L,1)
    ax.imshow(impatch,origin='lower',interpolation='nearest',
              vmin=np.min(impatch)*1.0001,
              vmax=np.max(impatch)*0.9999)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.text(0.5,1.1,'Initial Mean',
            transform=ax.transAxes,ha='center',va='center')
    ax = fig.add_subplot(L+1,L,L)
    ax.imshow(fmpatch,origin='lower',interpolation='nearest',
              vmin=np.min(fmpatch)*1.0001,
              vmax=np.max(fmpatch)*0.9999)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.text(0.5,1.1,'Final Mean',
            transform=ax.transAxes,ha='center',va='center')

    # psis
    ax = fig.add_subplot(L+2,L,L+1)
    ax.imshow(mix.psis[k].reshape(pshape),
              origin='lower',interpolation='nearest',
              vmin=np.min(mix.psis[k])*1.0001,
              vmax=np.max(mix.psis[k])*0.9999)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.text(0.5,1.05,'Psi',
            transform=ax.transAxes,ha='center',va='center')
            
    for ii in range(mix.M):
        ax = fig.add_subplot(L+2,L,L+2+ii)
        ax.imshow(mix.lambdas[k,:,ii].reshape(pshape),
                  origin='lower',interpolation='nearest',
                  vmin=np.min(mix.lambdas[k,:,ii])*1.0001,
                  vmax=np.max(mix.lambdas[k,:,ii])*0.9999)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5,1.05,'Lambda',
                transform=ax.transAxes,ha='center',va='center')
    return fig


def fig_mofa(mix,kmeans,filename):
    """
    Plot mean,lambdas,psis, high resp. patches, and draws 
    for each component of MOFA, out to single PDF
    """
    if filename[-4:]!='.pdf' : filename += '.pdf'
    pp = PdfPages(filename)

    pshape = (np.sqrt(len(mix.means[0].ravel())),
              np.sqrt(len(mix.means[0].ravel())))
    L = pshape[0]
    pl.gray()

    ampind = np.argsort(mix.amps)
    for i in range(mix.K):
        k   = ampind[-i-1]
        print k,i,ampind
        # write eigenvector patches
        fig = init_fig(L)
        fig = fig_mofa_head(fig,kmeans,mix,k,L,pshape)

        u,s,v = svd(mix.covs[k])
        for ii in range(int(pshape[0]**2)):
            ax = fig.add_subplot(L+2,L,2*L+ii+1)
            ax.imshow(u[ii,:].reshape(pshape),
                      origin='lower',interpolation='nearest',
                      vmin=np.min(u[ii,:])*1.0001,
                      vmax=np.max(u[ii,:])*0.9999)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(0.5,1.1,'Eigval = %1.3e' % s[ii],
                    transform=ax.transAxes,ha='center',va='center')
        pp.savefig()


        # write high rs patches
        fig = init_fig(L)
        fig = fig_mofa_head(fig,kmeans,mix,k,L,pshape)

        rs = mix.rs[k]
        ind = np.argsort(rs)
        for ii in range(int(L)**2):
            ax = fig.add_subplot(L+2,L,2*L+ii+1)
            ax.imshow(mix.data[ind[-ii-1]].reshape(pshape),
                      origin='lower',interpolation='nearest',
                      vmin=np.min(mix.means[k]),vmax=np.max(mix.means[k]))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(0.5,1.1,'rs = %0.3f' % rs[ind[-ii-1]],
                    transform=ax.transAxes,ha='center',va='center')

        pp.savefig()

        # write draws from components
        fig = init_fig(L)
        fig = fig_mofa_head(fig,kmeans,mix,k,L,pshape)

        for ii in range(int(L)**2):
            patch = np.random.multivariate_normal(mix.means[k],mix.covs[k])
            patch = np.array(patch) # is this necessary?
 
            ax = fig.add_subplot(L+2,L,2*L+ii+1)
            ax.imshow(patch.reshape(pshape),
                      origin='lower',interpolation='nearest',
                      vmin=np.min(mix.means[k]),vmax=np.max(mix.means[k]))
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        pp.savefig()

    pp.close()


def fig_patches(fig_side,patch_side,filename,
                data=None,mix=None,rand_flip=True):
    """
    Make a figure with patches that are either data or 
    draws
    """
    L   = fig_side
    fig = init_fig(L)
    pl.gray()

    if mix!=None:
        ks   = np.arange(mix.K,dtype=int)
        ind  = np.argsort(mix.amps)
        ks   = ks[ind]
        ref = np.arange(mix.K)
        amps = mix.amps[ind]
        cum  = np.zeros(mix.K)
        for ii in range(mix.K):
            cum[ii] = np.sum(amps[:ii+1])

    for ii in range(L**2):
        if data!=None:
            patch = data[ii,:]
        else:
            rnd = np.random.rand()
            ind = ref[cum>rnd]
            k = ks[ind[0]]
            patch = np.random.multivariate_normal(mix.means[k],mix.covs[k])
            patch = np.array(patch) # is this necessary?
 
        if rand_flip:
            p = Patches(np.atleast_2d(patch),None,
                        patched=True,flip=False,
                        rand_flip=rand_flip)
            patch = p.data 

        ax = fig.add_subplot(L,L,ii+1)
        ax.imshow(patch.reshape((patch_side,patch_side)),
                  origin='lower',interpolation='nearest',
                  vmin=np.min(patch)*1.001,vmax=np.max(patch)*0.999)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.savefig(filename)
