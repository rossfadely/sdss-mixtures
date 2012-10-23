import sdss
import mog
import time

import numpy as np
import matplotlib.pyplot as pl
import pyfits as pf

from numpy.linalg import svd
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from numpy.lib.stride_tricks import as_strided as ast

from matplotlib import rc
rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':10})
rc('text', usetex=True)


def fig_eigen(mix,kmeans,filename):
    """
    Plot mean and eigenvector patches for each
    component of MOG, out to single PDF
    """
    if filename[-4:]!='.pdf' : filename += '.pdf'
    pp = PdfPages(filename)

    pshape = (np.sqrt(len(mix.means[0].ravel())),
              np.sqrt(len(mix.means[0].ravel())))
    L = pshape[0]
    factor = 2.0          # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.5 * factor  # size of top/right margin
    whspace = 0.05        # w/hspace size
    plotdim = factor * L + factor * (L - 1.) * whspace
    dim = lbdim + plotdim + trdim
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    pl.gray()

    for k in range(mix.means.shape[0]):
        fig = pl.figure(figsize=(dim, dim + L))
        fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                            wspace=whspace, hspace=whspace)

        # write mean patches
        mpatch  = mix.means[k].reshape(pshape)
        kmpatch = kmeans[k,:].reshape(pshape)
        ax = fig.add_subplot(L+1,L,L)
        ax.imshow(kmpatch,origin='lower',interpolation='nearest')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5,1.1,'Initial Mean',
                transform=ax.transAxes,ha='center',va='center')
        ax = fig.add_subplot(L+1,L,1)
        ax.imshow(mpatch,origin='lower',interpolation='nearest')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5,1.1,'Final Mean',
                transform=ax.transAxes,ha='center',va='center')


        # write some info
        ax = fig.add_subplot(L+1,L,3)
        ax.set_axis_off()
        ax.text(0.0,0.5,'Component = %d' % k,
                transform=ax.transAxes,ha='left',va='center',
                fontsize=20)
        ax = fig.add_subplot(L+1,L,6)
        ax.set_axis_off()
        ax.text(0.0,0.5,'Amp = %1.2e' % mix.amps[k],
                transform=ax.transAxes,ha='left',va='center',
                fontsize=20)
        
        u,s,v = svd(mix.cov[k])

        # write eigenvector patches
        for ii in range(pshape[0]**2):
            ax = fig.add_subplot(L+1,L,L+ii+1)
            ax.imshow(u[ii,:].reshape(pshape),origin='lower',interpolation='nearest')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(0.5,1.1,'Eigval = %1.3e' % s[ii],
                    transform=ax.transAxes,ha='center',va='center')
        pp.savefig()

    pp.close()


def fig_patches(mix,patches,filename):
    """
    Plot patches.  For each patch, report likelihood,
    posterior under best component, and total likelihood
    under model.
    """
    if type(patches)!=int:
        Npatches = np.sqrt(len(patches[0,:]))
        inds = np.random.randint(0,len(patches[:,0]),64)
    else:
        Npatches = patches

    pshape = (np.sqrt(len(mix.means[0].ravel())),
              np.sqrt(len(mix.means[0].ravel())))
    factor = 2.0          # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.5 * factor  # size of top/right margin
    wspace = 0.05         # wspace size
    hspace = 0.1          # hspace size
    wdim = factor * Npatches + factor * (Npatches - 1.) * wspace + lbdim + trdim
    hdim = factor * Npatches + factor * (Npatches - 1.) * hspace + lbdim + trdim
    fig = pl.figure(figsize=(wdim, hdim))
    fig.subplots_adjust(left=lbdim/wdim, bottom=lbdim/hdim,
                        right=(wdim-trdim)/wdim,
                        top=(hdim-trdim)/hdim,
                        wspace=wspace, hspace=hspace)
    pl.gray()

    mix.means = mix.means.T
    Nk = len(mix.means[0,:])

    for ii in range(Npatches**2):
        if type(patches)==int:
            t  = [np.random.multivariate_normal(mix.means[:,k].ravel(),mix.cov[k]) \
                  * mix.amps[k] for k in range(Nk)]
            t  = np.array(t)
            patch = t.sum(axis=0)

        else:
            patch = patches[inds[ii],:]

        logL, rs = mix._calc_prob(np.array([patch]))
        loglikes = [mix._log_multi_gauss(k,np.array([patch])) for k in range(Nk)]
        loglikes = np.array(loglikes).flatten()
        bestlike = np.argsort(loglikes)
 
        ax = fig.add_subplot(Npatches,Npatches,ii+1)
        ax.imshow(patch.reshape(pshape),origin='lower',interpolation='nearest')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.0,1.05,'$\ln(p(D))$ = %1.1e' % (logL),
                transform=ax.transAxes,ha='left',va='center')
        ax.text(0.0,1.15,'$\ln(p(D|k=%d))$ = %1.1e' % (bestlike[-1],loglikes[bestlike[-1]]),
                transform=ax.transAxes,ha='left',va='center')

    if filename[-4:]!='.pdf' : filename += '.pdf'
    fig.savefig(filename,format='pdf')


def make_patch_examples(run,camcol,field,outname):

    # get data using tractor call
    data,invvar = get_sdss_data(run,camcol,field)

    # create array of patches
    dpatch   = patchify(data,step=(2,2))
    ipatch   = patchify(invvar,step=(2,2))

    # calc variance in data and min in invvar
    var = dpatch.std(axis=1)
    loi = ipatch.min(axis=1)

    # throw out invvar = 0
    ind = loi > 0
    dpatch = dpatch[ind]
    var    = var[ind]

    # draw 1% from a uniform dist over variance
    # fix this slow bit!!
    val = np.random.rand(0.01 * len(dpatch[:,0])) * \
          (np.max(var)-np.min(var)) + np.min(var)
    ind = np.array([],dtype='int')
    for v in val:
        ind = np.append(ind,(np.abs(var-v).argmin()))

    # write it to file
    hdu = pf.PrimaryHDU(dpatch[ind])
    hdu.writeto(outname)
    

def patchify(A, step=(1,1), block= (8, 8)):
    """Make a Ndata by (flattened) patch, 2D array"""
    shape   = ((A.shape[0] - block[0])/step[0] + 1,
               (A.shape[1] - block[1])/step[1] + 1) + block
    strides = (A.strides[0]*step[0],A.strides[1]*step[1]) + A.strides
    blocks = ast(A, shape= shape, strides= strides)
    blocks = blocks.flatten()
    shape = (shape[0]*shape[1],block[0]*block[1])
    strides = (blocks.itemsize*block[0]*block[1],blocks.itemsize)
    return ast(blocks, shape= shape, strides= strides)


def get_sdss_data(run,camcol,field):
    """Call Tractor functions to get data, invvar images
    of a given SDSS field"""
    d = sdss.get_tractor_image_dr9(run,camcol,field,'r',psf='dg')
    d = d[0]
    return d.data,d.invvar




