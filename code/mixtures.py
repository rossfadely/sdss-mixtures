import numpy as np

from scipy.linalg import inv
from scipy.linalg import det
from scipy.linalg import solve

from scipy.cluster.vq import kmeans

import mog

class MixtureModel(object):

    def __init__(self,data,K,M=None,mixtype='mofa'):
        """
        Mixture model for `N` number of data points
        in `D` dimensions.  One can create mixtures
        of Gaussians or reduced-dimensionality mixtures
        of latent variables.  The latter can be `ppca`,
        where an isotropic noise model is used or `mofa`
        where the noise is non-isotropic.
        """
        self.K    = K # number of components
        self.M    = M # latent dimensionality
        self.type = mixtype

        self._data = np.atleast_2d(data)
        self.N = self._data.shape[0]
        self.D = self._data.shape[1]

        # initialize some quantities
        self.kmeans_rs = np.zeros(self.N, dtype=int)
        self.LogL = None
        
        # Randomly choose ``K`` components to be the initial means.
        inds = np.random.randint(self.N, size=self.K)
        self.means = data[inds, :]
        #self.means = kmeans(data,self.K)[0]

        # Randomly assign the amplitudes.
        self.amps = np.random.rand(K)
        self.amps /= np.sum(self.amps)

        if self.type == 'mofa':
            # Randomly assign factor loadings
            self.lam = np.random.randn(self.K,self.D,self.M)

            # Initialize (to be filled)
            self.rs = np.zeros((self.K,self.N))
            self.lat = np.zeros((self.K,self.M,self.N))
            self.lat_cov = np.zeros((self.K,self.M,self.M))

            # Set (high rank) variance to variance of all data
            # Do something approx. here for speed?
            self.psi = np.var(self._data) * np.ones(self.D)

            self.cov = np.zeros((self.K,self.D,self.D))
            for k in range(self.K):
                self.cov[k] = np.dot(self.lam[k],self.lam[k].T) + \
                    np.diag(self.psi)

    def run_kmeans(self, maxiter=200, tol=1e-4, verbose=True):
        """
        Run the K-means algorithm using the C extension.
        
        :param maxiter:
            The maximum number of iterations to try.
        
        :param tol:
            The tolerance on the relative change in the loss function that
            controls convergence.

        :param verbose:
            Print all the messages?

        """
        iterations = _algorithms.kmeans(self._data, self.means,
                                        self.kmeans_rs, tol, maxiter)
        
        if verbose:
                if iterations < maxiter:
                    print("K-means converged after {0} iterations."
                          .format(iterations))
                else:
                    print("K-means *didn't* converge after {0} iterations."
                          .format(iterations))

    def run_em(self):

        if self.type=='mofa':
            for i in range(1):
                self._expectation_mofa()
                self.run_maximization_mofa()

    def _expectation_mofa(self):

        L, rsT = self._calc_prob(self._data)
        self.rs = rsT.T

        print L.sum()
        self._calc_lat_expectation()


    def run_maximization_mofa(self):

        for k in range(self.K):

            # mean update
            sumrs = np.sum(self.rs[k])
            step  = np.dot(self.lam[k],self.lat[k]).T
            print step.shape,self._data.shape
            step  = self.rs[k] * (self._data - step)
            print step.shape
            self.means[k,:] = np.dot(self.rs[k],self._data - step) / sumrs

            # Lambda update
            rs_lat_cov = self.lat_cov[k].ravel()
            for i in range(len(rs_lat_cov)):
                rs_lat_cov[i] = np.sum(self.rs[k] * rs_lat_cov[i])
            rs_lat_cov = rs_lat_cov.reshape(self.lat_cov[k].shape)
            zeroed_data = (self._data-self.means[k]).T
            step  = np.dot(zeroed_data,self.lat[k].T)
            shape = step.shape
            step  = step.ravel() 
            for i in range(len(step)):
                step[i] = np.sum(self.rs[k] * step[i])
            step = step.reshape(shape)
            self.lam[k] = np.dot(step,inv(rs_lat_cov))

            # psi update
            zeroed_data = (self._data-self.means[k]).T
            step = np.dot(self.lam[k],np.dot(self.lat[k],zeroed_data.T))
            step = np.dot(zeroed_data,zeroed_data.T) - step
            shape = step.shape
            step = step.ravel()
            for i in range(len(step)):
                step[i] = np.sum(self.rs[k] * step[i])
            step = step.reshape(shape)
            self.psi += np.diag(step) / self.N

            # amps update
            self.amps[k] = np.sum(self.rs[k]) / self.N

        # after everything is updated, recompute cov
        for k in range(self.K):
            self.cov[k] = np.dot(self.lam[k],self.lam[k].T) + \
                np.diag(self.psi)

    def _loglike_total_mofa(self):

        for k in range(1):
            pt2 = np.dot(self.lam[k],self.lat_cov[k])
            pt2 = np.dot(np.diag(self.psi),step)
            pt2 = np.trace(np.dot(self.lam[k].T,step))

            #pt1 = 
            
    def _invert_cov(self,k):
        """
        Calculate inverse covariance of mofa or ppca model,
        using inversion lemma
        """
        psiI = inv(np.diag(self.psi))
        lam  = self.lam[k]
        lamT = lam.T
        step = inv(np.eye(self.M) + np.dot(lamT,np.dot(psiI,lam)))
        step = np.dot(step,np.dot(lamT,psiI))
        step = np.dot(psiI,np.dot(lam,step))

        return psiI - step


    def _log_multi_gauss(self, k, X):
        # X.shape == (P,D)
        # self.means.shape == (D,K)
        # self.cov[k].shape == (D,D)
        sgn, logdet = np.linalg.slogdet(self.cov[k])
        if sgn <= 0:
            return -np.inf * np.ones(X.shape[0])

        # X1.shape == (P,D)
        X1 = X - self.means[k,:]

        # X2.shape == (P,D)
        X2 = np.linalg.solve(self.cov[k], X1.T).T

        p = -0.5 * np.sum(X1 * X2, axis=1)

        return -0.5 * np.log((2 * np.pi) ** (X.shape[1])) - 0.5 * logdet + p


    def _calc_prob(self, x):
        x = np.atleast_2d(x)

        logrs = []
        for k in range(self.K):
            logrs += [np.log(self.amps[k]) + self._log_multi_gauss(k, x)]
        logrs = np.concatenate(logrs).reshape((-1, self.K), order='F')

        # here lies some ghetto log-sum-exp...
        # nothing like a little bit of overflow to make your day better!
        a = np.max(logrs, axis=1)
        L = a + np.log(np.sum(np.exp(logrs - a[:, None]), axis=1))
        logrs -= L[:, None]
        return L, np.exp(logrs)

    def _calc_lat_expectation(self):

        for k in range(self.K):
            
            zeroed_data = (self._data-self.means[k,:]).T
            
            # invert covariance
            icov = self._invert_cov(k)
            beta = np.dot(self.lam[k].T,icov)

            # Expectation of latents
            self.lat[k,:] = np.dot(beta,zeroed_data)

            # Expectation of latent cov
            step = np.dot(zeroed_data,zeroed_data.T)
            step = np.dot(step,beta.T)
            self.lat_cov[k,:] = np.eye(self.M) - np.dot(beta,self.lam[k]) + \
                np.dot(beta,step)

"""



"""
