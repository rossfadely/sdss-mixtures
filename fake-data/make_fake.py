import numpy as np
import matplotlib.pyplot as pl
import pyfits as pf


class Fake(object):
    """
    Make some fake data, corrupt `Pcorrupt`
    percentage by first multiplying then adding, spit 
    out to fits if desired.
    """
    def __init__(self,filename=None,pside=8,K=1,N=2**18,Pcorrupt=0.005,add=10.0,
                 allflux=10,hwhm=1.5,mult=2.0,gain=100.,rdns=0.025,
                 column=2,randcenters=True):
        self.D = pside**2
        self.N = N
        self.K = K
        self.add = add
        self.mult = mult
        self.hwhm = hwhm
        self.rdns = rdns
        self.gain = gain
        self.pside = pside
        self.Ncorr = np.floor(Pcorrupt * self.N)
        self.column = column
        self.allflux = allflux
        self.randcenters = randcenters

        self.make_clean()
        self.corrupt() #==True

        if filename!=None:
            hdu = pf.PrimaryHDU(self.data)
            hdu.writeto(filename)

    def make_clean(self):
        """
        Make clean patches with noisy gaussians centered in 
        the patches
        """
        self.data = np.zeros((self.N,self.D))
        self.xy_grid_vals()

        # add a centered gaussian
        if self.allflux!=None:
            self.flux = np.ones((self.N,self.D)) * self.allflux
            self.add_gaussian()
            self.data *= self.flux

        # add noise
        var = self.rdns**2 + self.data/self.gain
        self.data += np.random.randn(self.N,self.D) * np.sqrt(var)        

    def corrupt(self):
        """
        Corrupt a column of pixels for Ncorr data
        """
        ind  = np.tile(np.atleast_2d(self.xgrid==self.xgrid[self.column]),
                       (self.N,1))
        
        ind[:(self.N-self.Ncorr)] = False

        self.data[ind] *= self.mult
        self.data[ind] += self.add

    def xy_grid_vals(self):
        """
        Generate values of x,y grid
        """
        x  = np.array([])
        for ii in range(self.pside):
            x = np.append(x,np.arange(1,self.pside+1))
        x -= np.mean(x)

        y  = np.array([])
        for ii in range(self.pside):
            y = np.append(y,np.zeros(self.pside)+ii+1)
        y -= np.mean(y)
        
        self.xgrid = x
        self.ygrid = y

    def add_gaussian(self):
        """
        Add a gaussian to the center of 
        """
        if self.randcenters:
            x0 = np.random.rand(self.N) * self.pside - self.pside/2.
            y0 = np.random.rand(self.N) * self.pside - self.pside/2.
            x0 = np.atleast_2d(x0)
            y0 = np.atleast_2d(y0)
            x0 = np.tile(x0.T,(1,64))
            y0 = np.tile(y0.T,(1,64))
            gauss = np.exp(-0.5 * (((self.xgrid.T[None,:]-x0)**2.+
                                    (self.ygrid.T[None,:]-y0)**2.) 
                                   / self.hwhm**2) / 
                            np.sqrt(2. * np.pi * self.hwhm**2))
            self.data += gauss
        else:
            gauss = np.exp(-0.5 * (self.xgrid**2.+self.ygrid**2.)
                           / self.hwhm**2) / np.sqrt(2. * np.pi * self.hwhm**2)
            self.data += gauss[None,:]



