import unittest
import numpy as np
import fermitensor as ftn
from graded_tensor import GradedTensor


class TestGradedTensor(unittest.TestCase):

    def test_even_parity_tensor(self):
        """
        Verify that interchanging axes with an overall even parity tensor does not incur a sign change.
        """
        rng = np.random.default_rng()
        a = GradedTensor(0.5*ftn.crandn((5, 4, 7), rng), (3, 2, 4)).enforce_parity("even")
        x = ftn.crandn(11, rng)
        # outer product with vector 'x'
        b = GradedTensor(np.kron(a.data.reshape(-1), x).reshape(a.shape + x.shape), a.dim_even + (6,))
        for i in reversed(range(b.ndim - 1)):
            b = b.swap_axes(i, i + 1)
        self.assertTrue(np.allclose(b.data, np.kron(x, a.data.reshape(-1)).reshape(x.shape + a.shape)))


if __name__ == "__main__":
    unittest.main()
