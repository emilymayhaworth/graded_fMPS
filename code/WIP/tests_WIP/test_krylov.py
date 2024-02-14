import unittest
import numpy as np
from scipy.linalg import expm
import fermitensor as ftn
import WIP as WIPftn
#from WIPftn.krylov import *

class TestKrylov(unittest.TestCase):

    def test_lanczos_iteration(self):

        rng = np.random.default_rng()

        n = 256
        numiter = 24

        # random Hermitian matrix
        A = ftn.crandn((n, n), rng) / np.sqrt(n)
        A = 0.5 * (A + A.conj().T)

        # random complex starting vector
        vstart = ftn.crandn(n, rng) / np.sqrt(n)

        # simply use A as linear transformation
        alpha, beta, V = WIPftn.lanczos_iteration(lambda x: A @ x, vstart, numiter)

        # check orthogonality of Lanczos vectors
        self.assertTrue(np.allclose(V.T.conj() @ V, np.identity(numiter), rtol=1e-12),
                        msg='matrix of Lanczos vectors must be orthonormalized')

        # Lanczos vectors must tridiagonalize A
        T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
        self.assertTrue(np.allclose(V.conj().T @ A @ V, T, rtol=1e-12),
                        msg='Lanczos vectors must tridiagonalize A')


    def test_eigh_krylov(self):

        rng = np.random.default_rng()

        n = 196
        numiter = 30
        numeig  = 2

        # random Hermitian matrix
        A = ftn.crandn((n, n), rng) / np.sqrt(n)
        A = 0.5 * (A + A.conj().T)

        # random complex starting vector
        vstart = ftn.crandn(n, rng) / np.sqrt(n)

        # simply use A as linear transformation;
        w, u_ritz = WIPftn.eigh_krylov(lambda x: A @ x, vstart, numiter, numeig)

        # check orthogonality of Ritz matrix
        self.assertTrue(np.allclose(u_ritz.conj().T @ u_ritz, np.identity(numeig), rtol=1e-12),
                        msg='matrix of Ritz eigenvectors must be orthonormalized')

        # check U^H A U = diag(w)
        self.assertTrue(np.allclose(u_ritz.conj().T @ A @ u_ritz, np.diag(w), rtol=1e-12),
                        msg='Ritz eigenvectors must diagonalize A within Krylov subspace')

        # reference eigenvalues
        w_ref = np.linalg.eigvalsh(A)

        # compare lowest eigenvalues
        self.assertAlmostEqual(w[0], w_ref[0], delta=0.001,
                               msg='lowest Lanczos eigenvalue should approximate exact eigenvalue')

        self.assertAlmostEqual(w[1], w_ref[1], delta=0.02,
                               msg='second-lowest Lanczos eigenvalue should approximate exact eigenvalue')



if __name__ == '__main__':
    unittest.main()