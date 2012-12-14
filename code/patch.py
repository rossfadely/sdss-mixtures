import numpy as np

from numpy.lib.stride_tricks import as_strided as ast

class Patches(object):
    """

    Class for creating patches from image data, reducing it to
    lower dimensions via rotation/flips, and producing patches
    drawn from a (mixture of) gaussians.

    `data`: 2D image data or 2D patch array, use `patched`
    `pside`: size of one side of a patch
    `step`: steps by which to stride over image to make patches


    """
    def __init__(self, data, invvar,pshape=(8,8), step=(1,1), 
                 patched=False, flip=True, save_unflipped=False,
                 rand_flip=False, get_xy=False, var_lim=(0.,np.Inf)):
        self.pside  = pshape[0]
        self.pshape = pshape
        self.step   = step
        self.nflips = np.empty(7)

        if patched:
            self.data = data
        else:
            self.data = self.patchify(data)

        if get_xy:
            ys, xs    = np.mgrid[0:data.shape[0],0:data.shape[1]]
            self.xs   = self.patchify(xs)
            self.ys   = self.patchify(ys)
        
        self.indices = self.patchify(
            np.arange(data.shape[0]*data.shape[1],
                      dtype='int').reshape(data.shape[0],
                                           data.shape[1]))
        good      = np.all(self.patchify(invvar) > 0,axis=1) 
        self.data = self.data[good]

        if var_lim!=(0.,np.Inf):
            datavar   = np.var(self.data,axis=1)
            ind       = (datavar > var_lim[0]) & (datavar < var_lim[1])
            assert np.sum(ind)>0
            self.data = self.data[ind]
            self.xs   = self.xs[ind]
            self.ys   = self.ys[ind]
            print '\nAfter var trim',self.data.shape

        self.ndata  = self.data.shape[0]

        if save_unflipped:
            self.unflipped = self.data.copy()
            
        # Decorate this dummy
        if flip:
            assert pshape[0]==pshape[1], 'Flipping set for rect. patches'
            self.centroid()
            self.flip_patches()
            self.centroid()

        if rand_flip:
            self.rand_flip()

    def patchify(self, D):
        """
        Make a Ndata by (flattened) pshape, 2D array
        """
        step    = self.step
        pshape  = self.pshape   

        shape   = ((D.shape[0] - pshape[0])/step[0] + 1,
                   (D.shape[1] - pshape[1])/step[1] + 1) + pshape
        strides = (D.strides[0]*step[0],D.strides[1]*step[1]) + D.strides
        blocks  = ast(D, shape= shape, strides= strides)
        blocks  = blocks.ravel()
        shape   = (shape[0]*shape[1],pshape[0]*pshape[1])
        strides = (blocks.itemsize*pshape[0]*pshape[1],blocks.itemsize)

        return ast(blocks, shape= shape, strides= strides)

    def centroid(self):
        """
        Compute centroids of patches
        """
        x, y = self.xy_grid_vals()

        denom = np.sum(self.data, axis=1)
        self.xc = np.sum(x * self.data,axis=1) / denom
        self.yc = np.sum(y * self.data,axis=1) / denom

    def flip_y(self,ind):
        """
        Flip the patches in array around 'y' axis
        """
        pside = self.pside
        for ii in range(pside):
            self.data[ind,ii*pside:pside*(ii+1)] = \
                np.fliplr(self.data[ind,ii*pside:pside*(ii+1)])

    def flip_xandy(self,ind):
        """
        Flip the patches in array around 'x' and 'y' axis
        """
        self.data[ind] = np.fliplr(self.data[ind])

    def flip_yeqx(self,ind):
        """
        Flip the patches in array around 'y=x' axis
        """
        pside = self.pside
        t = self.data[ind]
        t = np.reshape(t,(t.shape[0],pside,pside))
        self.data[ind] = np.reshape(np.transpose(t,(0,2,1)),
                                    (self.data[ind].shape))

    def flip_patches(self):
        """
        Flip the patches so centroids lie in the 45-90 degree (from 'x' axis)
        eighth of the patch.
        """
        
        ind_x = (self.xc >= 0)
        ind_y = (self.yc >= 0)
        ind_s = (self.xc-self.yc >= 0)
        ind_a = (self.yc+self.xc >= 0)
        
        # Sect 1 = y=x axis flip
        ind = (ind_x==1) & (ind_y==1) & (ind_s==1) & (ind_a==1)
        self.flip_yeqx(ind)
        self.nflips[0] = len(self.data[ind])

        # Sect 2 = x, y=x axis flip
        ind = (ind_x==1) & (ind_y==0) & (ind_s==1) & (ind_a==1)
        self.flip_xandy(ind)
        self.flip_y(ind)
        self.flip_yeqx(ind)
        self.nflips[1] = len(self.data[ind])
        
        # Sect 3 = x axis flip
        ind = (ind_x==1) & (ind_y==0) & (ind_s==1) & (ind_a==0)
        self.flip_y(ind)
        self.flip_xandy(ind)
        self.nflips[2] = len(self.data[ind])

        # Sect 4 = x, y axis flip
        ind = (ind_x==0) & (ind_y==0) & (ind_s==1) & (ind_a==0)
        self.flip_xandy(ind)
        self.nflips[3] = len(self.data[ind])

        # Sect 5 = x, y, y=x axis flip
        ind = (ind_x==0) & (ind_y==0) & (ind_s==0) & (ind_a==0)
        self.flip_yeqx(ind)
        self.flip_xandy(ind)
        self.nflips[4] = len(self.data[ind])

        # Sect 6 = y, y=x axis flip
        ind = (ind_x==0) & (ind_y==1) & (ind_s==0) & (ind_a==0)
        self.flip_y(ind)
        self.flip_yeqx(ind)
        self.nflips[5] = len(self.data[ind])

        # Sect 7 = y axis flip
        ind = (ind_x==0) & (ind_y==1) & (ind_s==0) & (ind_a==1)
        self.flip_y(ind)
        self.nflips[6] = len(self.data[ind])



    def get_1D_patches(self,x):
        """
        Return 1D patches.  If `x` == int, grab random
        number, otherwise return indicies specified in `x`
        """
        if isinstance(x,int):
            rand = np.random.rand(self.ndata)
            ind  = np.argsort(rand)
            return self.data[ind[:x],:]
        else:
            return self.data[x,:]

    def get_2D_patches(self,x):
        """
        Return 2D patches.  If `x` == int, grab random
        number, otherwise return indicies specified in `x`
        """
        if isinstance(x,int):
            rand = np.random.rand(self.ndata)
            ind  = np.argsort(rand)
            return self.data[ind[:x],:].reshape(x,self.pside,self.pside)
        else:
            return self.data[x,:].reshape(len(x),self.pside,self.pside)

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
        
        return x,y

    def rand_flip(self):
        """
        Take patches in self.data, do random flips
        """
        # Generate random centroids
        x, y    = self.xy_grid_vals()
        self.xc = x[np.random.randint(0,len(x),self.ndata)] 
        self.yc = y[np.random.randint(0,len(y),self.ndata)] 

        # Now flip the patches
        self.flip_patches()










