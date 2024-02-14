import unittest
import numpy as np
from scipy import sparse
import fermitensor as ftn


class TestfHamiltonian(unittest.TestCase):

    def test_spinless_fermi_hubbard_mpo(self):

        # number of lattice sites
        L = 5

        # Hamiltonian parameters
        t = 0.7
        u = 2.5

        mpoH = ftn.spinless_fermi_hubbard_mpo(L, t, u, pbc=False)

        # reference construction
        clist, alist = ftn.generate_fermi_operators(L)
        # number operators
        nlist = [c @ a for c, a in zip(clist, alist)]
        Href = sparse.csr_matrix((2**L, 2**L), dtype=float)
        for i in range(L - 1):
            # kinetic and interaction terms
            Href += (-t) * (clist[i] @ alist[i+1] + clist[i+1] @ alist[i]) + u * (nlist[i] @ nlist[i+1])
        Href = Href.todense()
        self.assertTrue(np.allclose(Href.conj().T, Href))
        # partition into even and odd parity sectors
        idx = (  [i for i in range(2**L) if i.bit_count() % 2 == 0]
               + [i for i in range(2**L) if i.bit_count() % 2 == 1])
        Href = Href[idx, :]
        Href = Href[:, idx]
        # compare
        self.assertTrue(np.allclose(mpoH.as_matrix(), Href))


if __name__ == "__main__":
    unittest.main()
