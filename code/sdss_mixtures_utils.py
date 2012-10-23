import numpy as np
import matplotlib.pyplot as pl
import pyfits as pf

from numpy.linalg import svd
from matplotlib.backends.backend_pdf import PdfPages

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
        for ii in range(int(pshape[0]**2)):
            ax = fig.add_subplot(L+1,L,L+ii+1)
            ax.imshow(u[ii,:].reshape(pshape),origin='lower',interpolation='nearest')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(0.5,1.1,'Eigval = %1.3e' % s[ii],
                    transform=ax.transAxes,ha='center',va='center')
        pp.savefig()

    pp.close()


def fig_patch(mix,patches,filename):
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

    for ii in range(int(Npatches**2)):
        if type(patches)==int:
            t  = [np.random.multivariate_normal(mix.means[:,k].ravel(),mix.cov[k]) \
                  * mix.amps[k] for k in range(Nk)]
            t  = np.array(t)
            patch = t.sum(axis=0)

        else:
            patch = patches[inds[ii],:]

        #logL, rs = mix._calc_prob(np.array([patch]))
        #loglikes = [mix._log_multi_gauss(k,np.array([patch])) for k in range(Nk)]
        #loglikes = np.array(loglikes).flatten()
        #bestlike = np.argsort(loglikes)
 
        ax = fig.add_subplot(Npatches,Npatches,ii+1)
        ax.imshow(patch.reshape(pshape),origin='lower',interpolation='nearest')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.text(0.0,1.05,'$\ln(p(D))$ = %1.1e' % (logL),
        #        transform=ax.transAxes,ha='left',va='center')
        #ax.text(0.0,1.15,'$\ln(p(D|k=%d))$ = %1.1e' % (bestlike[-1],loglikes[bestlike[-1]]),
        #        transform=ax.transAxes,ha='left',va='center')

    mix.means = mix.means.T

    if filename[-4:]!='.pdf' : filename += '.pdf'
    fig.savefig(filename,format='pdf')

    
def get_sdss_data(run,camcol,field):
    """Call Tractor functions to get data, invvar images
    of a given SDSS field"""
    import sdss #Tractor import

    d = sdss.get_tractor_image_dr9(run,camcol,field,'r',psf='dg')
    d = d[0]

    return d.data,d.invvar


 

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
    factor = 2.0          # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.5 * factor  # size of top/right margin
    whspace = 0.05        # w/hspace size
    plotdim = factor * L + factor * (L - 1.) * whspace
    dim = lbdim + plotdim + trdim
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    pl.gray()

    for k in range(mix.K):
        # first eigenpatches
        fig = pl.figure(figsize=(dim, dim + L))
        fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                            wspace=whspace, hspace=whspace)

        # write mean patches
        mpatch  = mix.means[k].reshape(pshape)
        kmpatch = kmeans[k,:].reshape(pshape)
        ax = fig.add_subplot(L+1,L,L)
        ax.imshow(kmpatch,origin='lower',interpolation='nearest',
                  vmin=np.min(kmpatch)*1.0001,
                  vmax=np.max(kmpatch)*0.9999)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5,1.1,'Initial Mean',
                transform=ax.transAxes,ha='center',va='center')
        ax = fig.add_subplot(L+1,L,1)
        ax.imshow(mpatch,origin='lower',interpolation='nearest',
                  vmin=np.min(mpatch)*1.0001,
                  vmax=np.max(mpatch)*0.9999)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5,1.1,'Final Mean',
                transform=ax.transAxes,ha='center',va='center')


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


        u,s,v = svd(mix.covs[k])

        # write eigenvector patches
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

        # now high rs patches
        fig = pl.figure(figsize=(dim, dim + L))
        fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                            wspace=whspace, hspace=whspace)

        # write mean patches
        mpatch  = mix.means[k].reshape(pshape)
        kmpatch = kmeans[k,:].reshape(pshape)
        ax = fig.add_subplot(L+1,L,L)
        ax.imshow(kmpatch,origin='lower',interpolation='nearest',
                  vmin=np.min(kmpatch)*1.0001,
                  vmax=np.max(kmpatch)*0.9999)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5,1.1,'Initial Mean',
                transform=ax.transAxes,ha='center',va='center')
        ax = fig.add_subplot(L+1,L,1)
        ax.imshow(mpatch,origin='lower',interpolation='nearest',
                  vmin=np.min(mpatch)*1.0001,
                  vmax=np.max(mpatch)*0.9999)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5,1.1,'Final Mean',
                transform=ax.transAxes,ha='center',va='center')


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

        rs = mix.rs[k]
        ind = np.argsort(rs)
        for ii in range(int(L)**2):
            ax = fig.add_subplot(L+2,L,2*L+ii+1)
            ax.imshow(mix.data[ind[-ii-1]].reshape(pshape),
                      origin='lower',interpolation='nearest',
                      vmin=np.min(mpatch),vmax=np.max(mpatch))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(0.5,1.1,'rs = %0.3f' % rs[ind[-ii-1]],
                    transform=ax.transAxes,ha='center',va='center')

        pp.savefig()



        # finally draws
        fig = pl.figure(figsize=(dim, dim + L))
        fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                            wspace=whspace, hspace=whspace)

        # write mean patches
        mpatch  = mix.means[k].reshape(pshape)
        kmpatch = kmeans[k,:].reshape(pshape)
        ax = fig.add_subplot(L+1,L,L)
        ax.imshow(kmpatch,origin='lower',interpolation='nearest',
                  vmin=np.min(kmpatch)*1.0001,
                  vmax=np.max(kmpatch)*0.9999)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5,1.1,'Initial Mean',
                transform=ax.transAxes,ha='center',va='center')
        ax = fig.add_subplot(L+1,L,1)
        ax.imshow(mpatch,origin='lower',interpolation='nearest',
                  vmin=np.min(mpatch)*1.0001,
                  vmax=np.max(mpatch)*0.9999)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5,1.1,'Final Mean',
                transform=ax.transAxes,ha='center',va='center')


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

        for ii in range(int(L)**2):
            t  = np.random.multivariate_normal(mix.means[k],mix.covs[k])
            t  = np.array(t)
            patch = t
 
            ax = fig.add_subplot(L+2,L,2*L+ii+1)
            ax.imshow(patch.reshape(pshape),
                      origin='lower',interpolation='nearest',
                      vmin=np.min(mpatch),vmax=np.max(mpatch))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
 

        pp.savefig()

    pp.close()
