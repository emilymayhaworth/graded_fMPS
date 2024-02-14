import math
import numpy as np


class GradedTensor:
    """
    Graded tensor, such that the first part of each axis has even parity and
    the second part odd parity.
    """
    def __init__(self, data, dim_even: list):
        data = np.asarray(data)
        dim_even = tuple(dim_even)
        if data.ndim != len(dim_even):
            raise ValueError("list of \"even\" dimensions must match number of dimensions in data tensor")
        if any(x > y for x, y in zip(dim_even, data.shape)):
            raise ValueError("entries in 'dim_even' must be less or equal to corresponding array dimensions")
        self.data = data
        self.dim_even = dim_even

    @property
    def shape(self):
        """
        Shape (array dimension) of the tensor.
        """
        return self.data.shape

    @property
    def ndim(self):
        """
        Degree (number of dimensions) of the tensor.
        """
        return self.data.ndim

    def swap_axes(self, i: int, j: int):
        """
        Swap two neighboring axes and return a new graded tensor with the swapped axes.
        """
        if i == j:
            raise ValueError("expecting two different axes")
        if i > j:
            i, j = j, i
        if i + 1 != j:
            raise ValueError(f"expecting neighboring axes, received {i} and {j}")
        perm = list(range(self.data.ndim))
        perm[i] = j
        perm[j] = i
        data = self.data.transpose(perm)
        dim_even = self.dim_even[:i] + (self.dim_even[j], self.dim_even[i]) + self.dim_even[j+1:]
        # sign flip due to interchange
        s = data.shape
        # temporarily flatten preceeding and trailing axes
        data = data.reshape((math.prod(s[:i]), s[i], s[j], math.prod(s[j+1:])))
        data[:, dim_even[i]:, dim_even[j]:, :] = -data[:, dim_even[i]:, dim_even[j]:, :]
        data = data.reshape(s)
        return GradedTensor(data, dim_even)

    def flatten_axes(self, i: int, j: int):
        """
        Flatten two neighboring axes into one axis and return a new graded tensor with the flattened axes.
        """
        if i == j:
            raise ValueError("expecting two different axes")
        if i > j:
            i, j = j, i
        if i + 1 != j:
            raise ValueError(f"expecting neighboring axes, received {i} and {j}")
        s = self.data.shape
        # temporarily flatten preceeding and trailing axes
        s0 = math.prod(s[:i])
        s2 = math.prod(s[j+1:])
        data = self.data.reshape((s0, s[i], s[j], s2))
        # re-order data into even-parity first part and odd-parity second part
        data = np.concatenate((np.reshape(data[:, :self.dim_even[i] , :self.dim_even[j] , :], (s0, -1, s2)),
                               np.reshape(data[:,  self.dim_even[i]:,  self.dim_even[j]:, :], (s0, -1, s2)),
                               np.reshape(data[:, :self.dim_even[i] ,  self.dim_even[j]:, :], (s0, -1, s2)),
                               np.reshape(data[:,  self.dim_even[i]:, :self.dim_even[j] , :], (s0, -1, s2))), axis=1)
        data = data.reshape(s[:i] + (s[i]*s[j],) + s[j+1:])
        # update even-parity dimensions
        dim_even_ij = self.dim_even[i]*self.dim_even[j] + (s[i] - self.dim_even[i])*(s[j] - self.dim_even[j])
        dim_even = self.dim_even[:i] + (dim_even_ij,) + self.dim_even[j+1:]
        return GradedTensor(data, dim_even)

    def has_parity(self, parity) -> bool:
        """
        Whether the graded tensor has the specified overall parity.
        """
        if parity == "even":
            parity = 0
        elif parity == "odd":
            parity = 1
        if not isinstance(parity, int):
            raise ValueError(f"invalid parity argument {parity}")
        # iterate over all data entries
        it = np.nditer(self.data, flags=["multi_index"], op_flags=["readonly"])
        for x in it:
            idx = it.multi_index
            p = 0
            for i in range(len(idx)):
                if idx[i] >= self.dim_even[i]:
                    p = 1 - p
            if p != parity and x[...] != 0:
                return False
        return True

    def enforce_parity(self, parity):
        """
        Enforce the specified overall parity by setting non-conforming entries to zero,
        and return a new graded tensor with the specified parity.
        """
        if parity == "even":
            parity = 0
        elif parity == "odd":
            parity = 1
        if not isinstance(parity, int):
            raise ValueError(f"invalid parity argument {parity}")
        # iterate over all data entries
        data = self.data.copy()
        it = np.nditer(data, flags=["multi_index"], op_flags=["writeonly"])
        for x in it:
            idx = it.multi_index
            p = 0
            for i in range(len(idx)):
                if idx[i] >= self.dim_even[i]:
                    p = 1 - p
            if p != parity:
                x[...] = 0
        return GradedTensor(data, self.dim_even)
