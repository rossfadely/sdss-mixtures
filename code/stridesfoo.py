from numpy.lib.stride_tricks import as_strided as ast
import numpy as np
import time

def block_view(A, step=(1,1), size = 8,block= (8,8)):
    """Provide a 2D block view to 2D array, at specified step and
    block size"""
    shape   = ((A.shape[0] - size)/step[0] + 1,
               (A.shape[1] - size)/step[1] + 1) + block
    strides = (A.strides[0]*step[0],A.strides[1]*step[1]) + A.strides
    return ast(A, shape= shape, strides= strides)

def blockify(A, step=(1,1), block= (8, 8)):
    """Provide a 2D block view to 2D array, at specified step and
    block size"""
    shape   = ((A.shape[0] - block[0])/step[0] + 1,
               (A.shape[1] - block[1])/step[1] + 1) + block
    strides = (A.strides[0]*step[0],A.strides[1]*step[1]) + A.strides
    print shape,strides,A.strides
    return ast(A, shape= shape, strides= strides)

def crazy(A, step=(1,1), block= (8, 8)):
    """Provide a 2D block view to 2D array, at specified step and
    block size"""
    shape   = ((A.shape[0] - block[0])/step[0] + 1,
               (A.shape[1] - block[1])/step[1] + 1) + block
    strides = (A.strides[0]*step[0],A.strides[1]*step[1]) + A.strides
    print shape,strides,A.strides
    foo = ast(A, shape= shape, strides= strides)
    bar = foo.flatten()
    print foo.shape
    shape = (shape[0]*shape[1],block[0]*block[1])
    strides = (bar.itemsize*64,bar.itemsize)
    #print bar
    #print shape,strides,len(bar)
    print ast(bar,shape=shape,strides=strides)



if __name__ == '__main__':
    t0 = time.time()
    n=20
    #A = np.empty((n,n))
    #for ii in range(n):
    #    A[ii,:]= np.random.randn(n)*ii
    A = np.arange(n**2).reshape(n,n)
    B= blockify(A,step=(1,1)).copy()
    print B.flatten()[64:128],B[0,1]
    C = B.flatten().copy()
    print C.strides
    print C.itemsize
    #C = blockify(A,(A.shape[0]*A.shape[1],64))
    #print C
    crazy(A,step=(1,1))

