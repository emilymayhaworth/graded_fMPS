import numpy as np
from scipy import sparse


def crandn(size=None, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None: rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


def generate_fermi_operators(L: int):
    """
    Generate sparse matrix representations of the fermionic creation and
    annihilation operators for `L` sites (or modes).
    """
    I = sparse.identity(2)
    Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
    U = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
    clist = []
    for i in range(L):
        c = sparse.identity(1)
        for j in range(L):
            if j < i:
                c = sparse.kron(c, I)
            elif j == i:
                c = sparse.kron(c, U)
            else:
                c = sparse.kron(c, Z)
        clist.append(c)
    # corresponding annihilation operators
    alist = [c.conj().T for c in clist]
    return (clist, alist)


def truncated_split(a, svd_distr: str, tol=0):
    """
    Split a matrix by SVD, and distribute singular values to one (or both)
    of the isometries; returns a tuple `(u, v)` such that ``a = u @ v``,
    possibly approximately equal due to singular value truncation.
    """
    u, s, v = np.linalg.svd(a, full_matrices=False)
    # truncate small singular values by absolute tolerance
    idx = np.where(s > tol)[0]
    u = u[:, idx]
    v = v[idx, :]
    s = s[idx]
    # use broadcasting to distribute singular values
    if svd_distr == "left":
        u = u * s
    elif svd_distr == "right":
        v = v * s[:, None]
    elif svd_distr == "sqrt":
        sq = np.sqrt(s)
        u = u * sq
        v = v * sq[:, None]
    else:
        raise ValueError('svd_distr parameter must be "left", "right" or "sqrt"')
    return u, v