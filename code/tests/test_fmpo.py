import unittest
import math
import numpy as np
import fermitensor as ftn
from graded_tensor import GradedTensor


class TestfMPO(unittest.TestCase):

    def test_logical_tensor(self):

        rng = np.random.default_rng()
        # physical and virtual bond dimensions
        d = 6
        DL = [ 7, 8]
        DR = [11, 5]
        Alogic = GradedTensor(0.5*ftn.crandn((DL[0] + DL[1], d, d, DR[0] + DR[1]), rng),
                              (DL[0], d//2, d//2, DR[0])).enforce_parity("even")
        A = ftn.fMPOTensor.from_logical_tensor(Alogic.data, DL, DR)
        # compare
        self.assertTrue(np.array_equal(A.to_logical_tensor(), Alogic.data))


    def test_merge_tensor_pair(self):

        rng = np.random.default_rng()

        # physical and virtual bond dimensions
        d  = [ 6,  8]
        De = [ 3,  7,  5]
        Do = [ 2, 11,  4]
        A = [ftn.fMPOTensor(0.5*ftn.crandn((d[i], d[i]//2, De[i], De[i+1]), rng),
                            0.5*ftn.crandn((d[i], d[i]//2, De[i], Do[i+1]), rng),
                            0.5*ftn.crandn((d[i], d[i]//2, Do[i], De[i+1]), rng),
                            0.5*ftn.crandn((d[i], d[i]//2, Do[i], Do[i+1]), rng)) for i in range(len(De)-1)]
        Am = ftn.merge_fmpo_tensor_pair(A[0], A[1])

        # reference calculation
        Am_ref = GradedTensor(np.einsum(A[0].to_logical_tensor(), (0, 1, 3, 6),
                                        A[1].to_logical_tensor(), (6, 2, 4, 5), (0, 1, 2, 3, 4, 5)),
                              (De[0], d[0]//2, d[1]//2, d[0]//2, d[1]//2, De[2]))
        # flatten physical dimensions
        Am_ref = Am_ref.flatten_axes(3, 4)
        Am_ref = Am_ref.flatten_axes(1, 2)

        # compare
        self.assertTrue(np.allclose(Am.to_logical_tensor(), Am_ref.data))


    def test_as_matrix(self):

        rng = np.random.default_rng()

        # physical and virtual bond dimensions
        d = [6, 4, 8]
        De = [ 3,  4,  5,  3]
        Do = [ 2, 11,  7,  2]
        # number of fermionic modes (or lattice sites)
        L = len(De) - 1
        A = [ftn.fMPOTensor(0.5*ftn.crandn((d[i], d[i]//2, De[i], De[i+1]), rng),
                            0.5*ftn.crandn((d[i], d[i]//2, De[i], Do[i+1]), rng),
                            0.5*ftn.crandn((d[i], d[i]//2, Do[i], De[i+1]), rng),
                            0.5*ftn.crandn((d[i], d[i]//2, Do[i], Do[i+1]), rng)) for i in range(L)]
        op = ftn.fMPO(A)
        self.assertEqual(op.nsites, L)
        self.assertEqual(op.bond_dims("even"), De)
        self.assertEqual(op.bond_dims("odd"),  Do)
        self.assertEqual(op.bond_dims("both"), [De[i] + Do[i] for i in range(L + 1)])
        self.assertEqual(op.as_matrix().shape, 2 * (math.prod(d),))


if __name__ == "__main__":
    unittest.main()
