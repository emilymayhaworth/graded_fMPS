import unittest
import numpy as np
import fermitensor as ftn
from graded_tensor import GradedTensor


class TestfMPS(unittest.TestCase):

    def test_logical_tensor(self):

        rng = np.random.default_rng()
        # physical and virtual bond dimensions
        d = 6
        DL = [ 7, 8]
        DR = [11, 5]
        Alogic = GradedTensor(0.5*ftn.crandn((DL[0] + DL[1], d, DR[0] + DR[1]), rng),
                              (DL[0], d//2, DR[0])).enforce_parity("even")
        A = ftn.fMPSTensor.from_logical_tensor(Alogic.data, DL, DR)
        # compare
        self.assertTrue(np.array_equal(A.to_logical_tensor(), Alogic.data))


    def test_merge_tensor_pair(self):

        rng = np.random.default_rng()

        # physical and virtual bond dimensions
        d  = [ 6,  8]
        De = [ 3,  7,  5]
        Do = [ 2, 11,  4]
        A = [ftn.fMPSTensor(0.5*ftn.crandn((d[i]//2, De[i], De[i+1]), rng),
                            0.5*ftn.crandn((d[i]//2, De[i], Do[i+1]), rng),
                            0.5*ftn.crandn((d[i]//2, Do[i], De[i+1]), rng),
                            0.5*ftn.crandn((d[i]//2, Do[i], Do[i+1]), rng)) for i in range(len(De)-1)]
        Am = ftn.merge_fmps_tensor_pair(A[0], A[1])
        print("shapes", Am.d, Am.DL, Am.DR, A[0].d, A[0].DL, A[0].DR, A[1].d, A[1].DL, A[1].DR)
        # reference calculation
        Am_ref = GradedTensor(np.tensordot(A[0].to_logical_tensor(),
                                           A[1].to_logical_tensor(), 1),
                              (De[0], d[0]//2, d[1]//2, De[2])).flatten_axes(1, 2)
        # compare
        self.assertTrue(np.allclose(Am.to_logical_tensor(), Am_ref.data))


    def test_split_tensor(self):

        rng = np.random.default_rng()

        # physical dimensions
        d0, d1 = 6, 10
        # outer virtual bond dimensions
        De = [13,  7]
        Do = [ 4, 11]
        Apair = ftn.fMPSTensor(0.5*ftn.crandn((d0*d1//2, De[0], De[1]), rng),
                               0.5*ftn.crandn((d0*d1//2, De[0], Do[1]), rng),
                               0.5*ftn.crandn((d0*d1//2, Do[0], De[1]), rng),
                               0.5*ftn.crandn((d0*d1//2, Do[0], Do[1]), rng))
        for svd_distr in ["left", "right", "sqrt"]:
            A0, A1 = ftn.split_fmps_tensor(Apair, d0, d1, svd_distr=svd_distr, tol=0)
            # merged tensor must agree with the original tensor
            Amrg = ftn.merge_fmps_tensor_pair(A0, A1)
            self.assertTrue(np.allclose(Amrg.to_logical_tensor(), Apair.to_logical_tensor()),
                            msg="splitting and subsequent merging must give the same tensor")


    def test_left_orthonormalize(self):

        rng = np.random.default_rng()

        # physical and virtual bond dimensions
        d  = [ 8,  6]
        De = [ 3,  7,  5]
        Do = [ 2, 11,  4]
        A = [ftn.fMPSTensor(0.5*ftn.crandn((d[i]//2, De[i], De[i+1]), rng),
                            0.5*ftn.crandn((d[i]//2, De[i], Do[i+1]), rng),
                            0.5*ftn.crandn((d[i]//2, Do[i], De[i+1]), rng),
                            0.5*ftn.crandn((d[i]//2, Do[i], Do[i+1]), rng)) for i in range(len(De)-1)]
        Am_ref = ftn.merge_fmps_tensor_pair(A[0], A[1])
        # perform orthonormalization
        As, As_next = ftn.left_orthonormalize_fmps_tensor(A[0], A[1])
        # must be an isometry
        Q = As.to_logical_tensor().reshape((-1, As.DR))
        self.assertTrue(np.allclose(Q.conj().T @ Q, np.identity(Q.shape[1])))
        # merged tensor must remain invariant
        Am = ftn.merge_fmps_tensor_pair(As, As_next)
        # compare
        self.assertTrue(np.allclose(Am.to_logical_tensor(), Am_ref.to_logical_tensor()))


    def test_right_orthonormalize(self):

        rng = np.random.default_rng()

        # physical and virtual bond dimensions
        d  = [ 8,  6]
        De = [ 3,  7,  5]
        Do = [ 2, 11,  4]
        A = [ftn.fMPSTensor(0.5*ftn.crandn((d[i]//2, De[i], De[i+1]), rng),
                            0.5*ftn.crandn((d[i]//2, De[i], Do[i+1]), rng),
                            0.5*ftn.crandn((d[i]//2, Do[i], De[i+1]), rng),
                            0.5*ftn.crandn((d[i]//2, Do[i], Do[i+1]), rng)) for i in range(len(De)-1)]
        Am_ref = ftn.merge_fmps_tensor_pair(A[0], A[1])
        # perform orthonormalization
        As, As_prev = ftn.right_orthonormalize_fmps_tensor(A[1], A[0])
        # must be an isometry
        Q = As.to_logical_tensor().reshape((As.DL, -1))
        self.assertTrue(np.allclose(Q @ Q.conj().T, np.identity(Q.shape[0])))
        # merged tensor must remain invariant
        Am = ftn.merge_fmps_tensor_pair(As_prev, As)
        # compare
        self.assertTrue(np.allclose(Am.to_logical_tensor(), Am_ref.to_logical_tensor()))


    def test_as_vector(self):

        rng = np.random.default_rng()

        # physical and virtual bond dimensions
        d = 6
        for parity in ["even", "odd"]:
            if parity == "even":
                De = [13,  7,  5, 13]
                Do = [ 2, 11,  4,  2]
            else:
                De = [13,  7,  5,  2]
                Do = [ 2, 11,  4, 13]
            # number of fermionic modes (or lattice sites)
            L = len(De) - 1
            A = [ftn.fMPSTensor(0.5*ftn.crandn((d//2, De[i], De[i+1]), rng),
                                0.5*ftn.crandn((d//2, De[i], Do[i+1]), rng),
                                0.5*ftn.crandn((d//2, Do[i], De[i+1]), rng),
                                0.5*ftn.crandn((d//2, Do[i], Do[i+1]), rng)) for i in range(L)]
            psi = ftn.fMPS(A)
            self.assertEqual(psi.nsites, L)
            self.assertEqual(psi.bond_dims("even"), De)
            self.assertEqual(psi.bond_dims("odd"),  Do)
            self.assertEqual(psi.bond_dims("both"), [De[i] + Do[i] for i in range(L + 1)])
            self.assertEqual(len(psi.as_vector(parity)), d**L // 2)


    def test_single_particle_state(self):

        rng = np.random.default_rng()

        for L in range(1, 7):
            coeffs = ftn.crandn((L,), rng)
            psi = ftn.fMPS.single_particle_state(coeffs)
            # construct reference state
            clist, _ = ftn.generate_fermi_operators(L)
            psi_ref = np.zeros(2**L, dtype=coeffs.dtype)
            for i in range(L):
                # vacuum state
                vac = np.zeros(2**L)
                vac[0] = 1
                psi_ref += coeffs[i] * (clist[i] @ vac)
            # select entries corresponding to odd particle number
            psi_ref = psi_ref[[i for i in range(2**L) if i.bit_count() % 2 == 1]]
            # compare
            self.assertTrue(np.array_equal(psi.as_vector("odd"), psi_ref))


    def test_two_particle_state(self):

        rng = np.random.default_rng()

        for L in range(2, 7):
            coeffs1 = ftn.crandn((L,), rng)
            coeffs2 = ftn.crandn((L,), rng)
            psi = ftn.fMPS.two_particle_state(coeffs1, coeffs2)
            # construct reference state
            clist, _ = ftn.generate_fermi_operators(L)
            c1 = sum(coeffs1[i] * clist[i] for i in range(L))
            c2 = sum(coeffs2[i] * clist[i] for i in range(L))
            # vacuum state
            vac = np.zeros(2**L)
            vac[0] = 1
            psi_ref = c2 @ (c1 @ vac)
            # select entries corresponding to even particle number
            psi_ref = psi_ref[[i for i in range(2**L) if i.bit_count() % 2 == 0]]
            # compare
            self.assertTrue(np.array_equal(psi.as_vector("even"), psi_ref))


    def test_canonical_basis_state(self):

        rng = np.random.default_rng()

        for L in range(1, 7):
            # number of particles
            nprtc = rng.integers(L + 1)
            modes = np.sort(rng.choice(L, size=nprtc, replace=False))
            psi = ftn.fMPS.canonical_basis_state(L, modes)
            # construct reference state
            clist, _ = ftn.generate_fermi_operators(L)
            # vacuum state
            psi_ref = np.zeros(2**L)
            psi_ref[0] = 1
            for m in modes:
                psi_ref = clist[m] @ psi_ref
            # select entries corresponding to even or odd particle number
            psi_ref = psi_ref[[i for i in range(2**L) if (i.bit_count() - nprtc) % 2 == 0]]
            # compare
            self.assertTrue(np.array_equal(psi.as_vector(nprtc % 2), psi_ref))


if __name__ == "__main__":
    unittest.main()
