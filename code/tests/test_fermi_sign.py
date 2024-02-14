import unittest
import numpy as np
import fermitensor as ftn


class TestFermiSign(unittest.TestCase):

    def test_create_sign(self):
        """
        Test sign structure when applying a fermionic creation operator.
        """
        rng = np.random.default_rng()

        # construct an even-parity fMPS with random tensor entries
        # physical and virtual bond dimensions
        d = 2
        De = [ 4,  7,  5, 13,  6,  4]
        Do = [ 5, 11,  4,  9,  8,  5]
        # number of fermionic modes (or lattice sites)
        L = len(De) - 1
        A = [ftn.fMPSTensor(0.5*ftn.crandn((d//2, De[i], De[i+1]), rng),
                            0.5*ftn.crandn((d//2, De[i], Do[i+1]), rng),
                            0.5*ftn.crandn((d//2, Do[i], De[i+1]), rng),
                            0.5*ftn.crandn((d//2, Do[i], Do[i+1]), rng)) for i in range(L)]
        psi = ftn.fMPS(A)

        idx_e = [i for i in range(2**L) if i.bit_count() % 2 == 0]
        idx_o = [i for i in range(2**L) if i.bit_count() % 2 == 1]

        # assemble overall statevector
        psi_vec_e = psi.as_vector("even")
        psi_vec = np.zeros(2**L, dtype=complex)
        for i, j in enumerate(idx_e):
            psi_vec[j] = psi_vec_e[i]

        # fermionic creation and annihilation operators
        clist, _ = ftn.generate_fermi_operators(L)

        for l in range(L):
            # apply creation operator
            c_psi_vec_ref = clist[l] @ psi_vec

            cA = psi.nsites * [None]
            for j in range(l):
                cA[j] = psi.A[j]
            # after applying creation operator at site l:
            # Aee ->  Aeo (physical state |0> -> |1>)
            # Aeo ->  0   (physical state |1> disappears)
            # Aoe ->  0   (physical state |1> disappears)
            # Aoo -> -Aoe (physical state |0> -> |1>, and sign flip due to odd subsequent parity)
            Aee =  np.zeros_like(psi.A[l].block(1, 0, 1))
            Aeo =  psi.A[l].block(0, 0, 0)
            Aoe = -psi.A[l].block(0, 1, 1)
            Aoo =  np.zeros_like(psi.A[l].block(1, 1, 0))
            cA[l] = ftn.fMPSTensor(Aee, Aeo, Aoe, Aoo)
            for j in range(l + 1, L):
                # flip even <-> odd parity of both virtual bonds
                cA[j] = ftn.fMPSTensor(psi.A[j].block(0, 1, 1),  # Aee
                                       psi.A[j].block(1, 1, 0),  # Aeo
                                       psi.A[j].block(1, 0, 1),  # Aoe
                                       psi.A[j].block(0, 0, 0))  # Aoo
            c_psi = ftn.fMPS(cA)
            # assemble overall statevector
            c_psi_vec_o = c_psi.as_vector("odd", sign_eo=1)
            c_psi_vec = np.zeros(2**L, dtype=complex)
            for i, j in enumerate(idx_o):
                c_psi_vec[j] = c_psi_vec_o[i]

            self.assertTrue(np.allclose(c_psi_vec, c_psi_vec_ref))


if __name__ == "__main__":
    unittest.main()
