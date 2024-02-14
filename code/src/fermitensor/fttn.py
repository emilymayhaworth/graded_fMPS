import math
import numpy as np
import itertools
from typing import Sequence
from fermitensor.util import truncated_split


class fTTNTensor:
    """
    Fermionic tree tensor network (fTTN) tensor of logical shape
    `(D0, D1, ..., d, ..., Dn)` or `(D0, D1, ..., Dn)` (with or without physical axis),
    where `D0`, ..., `Dn` are the dimensions of the virtual bonds
    connected to the parent and child nodes,
    and `d` the physical dimension (if present).
    Assuming that `d` is even, and that the first half of the physical indices
    has even parity and the second half odd parity.
    The tensor has even overall parity, and entries not compatible with the parity
    have to be zero. (For simplicity, compact data storage is not implemented yet.)
    """
    def __init__(self, De: Sequence[int], Do: Sequence[int], physical: bool, data=None, d=0, physax=-1, dtype=None):
        if len(De) != len(Do):
            raise ValueError("number of even-parity virtual bonds must match number of odd-parity virtual bonds")
        if data is not None:
            data = np.asarray(data)
        self.De = tuple(De)
        self.Do = tuple(Do)
        Dtot = tuple(e + o for e, o in zip(De, Do))
        self.physical = physical    # whether tensor has a physical axis
        if physical:
            if physax < 0 or physax > len(De):
                raise ValueError(f"invalid physax = {physax}")
            self.physax = physax
            if data is not None:
                if data.ndim != len(Dtot) + 1:
                    raise ValueError(f"number of dimensions of `data` does not match expected {len(De) + 1}")
                if data.shape[:physax] != Dtot[:physax] or data.shape[physax+1:] != Dtot[physax:]:
                    raise ValueError("shape of input data does not match virtual bond dimensions")
                self.data = data
            else:
                if d <= 0:
                    raise ValueError("unspecified or invalid physical dimension")
                if d % 2 == 1:
                    raise ValueError("physical dimension must be even")
                self.data = np.zeros(Dtot[:physax] + (d,) + Dtot[physax:], dtype=dtype)
        else:
            if data is not None:
                if data.ndim != len(De):
                    raise ValueError(f"number of dimensions of `data` does not match expected {len(De)}")
                if data.shape != Dtot:
                    raise ValueError("shape of input data does not match virtual bond dimensions")
                self.data = data
            else:
                self.data = np.zeros(Dtot, dtype=dtype)
        if not self.is_parity_consistent():
            raise ValueError("sparsity pattern of tensor entries does not match overall even parity constraint")

    @property
    def d(self) -> int:
        """
        Physical dimension.
        """
        if self.physical:
            return self.data.shape[self.physax]
        else:
            return 0

    @property
    def nvbonds(self) -> int:
        """
        Number of virtual bonds.
        """
        return len(self.De)

    @property
    def D(self) -> tuple:
        """
        Overall virtual bond dimensions.
        """
        return tuple(e + o for e, o in zip(self.De, self.Do))

    def shape(self, parity="both") -> tuple:
        """
        Tensor shape (i.e., its dimensions), including physical dimension (if present).
        """
        if parity == "both":
            return self.data.shape
        elif parity == "even" or parity == 0:
            if self.physical:
                return self.De[:self.physax] + (self.d//2,) + self.De[self.physax:]
            else:
                return self.De
        elif parity == "odd" or parity == 1:
            if self.physical:
                return self.Do[:self.physax] + (self.d//2,) + self.Do[self.physax:]
            else:
                return self.Do
        else:
            raise ValueError(f"invalid argument parity = {parity}")

    @property
    def dtype(self):
        """
        Data type of tensor entries.
        """
        return self.data.dtype

    def orthonormalize(self, method: str, tol=0):
        """
        Orthonormalize the tensor (bring to canonical form) using QR or SVD decomposition,
        returning the matrices (for even and odd parity) to be absorbed into
        the parent tensor along the upstream bond connection.
        """
        # assuming that the leading dimension is the parent virtual bond
        if self.physical and self.physax == 0:
            raise NotImplementedError
        shape_even = self.shape("even")
        # partition indices for all dimensions except first according to parity
        idx_even = []
        idx_odd  = []
        for i, idx in enumerate(itertools.product(*[range(d) for d in self.data.shape[1:]])):
            p = 0
            for j in range(len(idx)):
                if idx[j] >= shape_even[1 + j]:
                    p = 1 - p
            if p == 0:
                idx_even.append(i)
            else:
                idx_odd.append(i)
        if method == "qr":
            s = self.data.shape
            A = self.data.reshape((s[0], -1))
            # separate QR decompositions for even and odd parities
            Qe, Re = np.linalg.qr(A[:self.De[0], idx_even].T, mode="reduced")
            Qo, Ro = np.linalg.qr(A[self.De[0]:, idx_odd ].T, mode="reduced")
            Q = np.zeros((Qe.shape[1] + Qo.shape[1], A.shape[1]), dtype=A.dtype)
            Q[:Qe.shape[1], idx_even] = Qe.T
            Q[Qe.shape[1]:, idx_odd ] = Qo.T
            self.De = (Qe.shape[1],) + self.De[1:]
            self.Do = (Qo.shape[1],) + self.Do[1:]
            self.data = np.reshape(Q, (Q.shape[0],) + s[1:])
            assert self.is_parity_consistent()
            return Re.T, Ro.T
        elif method == "svd":
            s = self.data.shape
            A = self.data.reshape((s[0], -1))
            # separate SVDs for even and odd parities
            Re, Ve = truncated_split(A[:self.De[0], idx_even], "left", tol=tol)
            Ro, Vo = truncated_split(A[self.De[0]:, idx_odd ], "left", tol=tol)
            V = np.zeros((Ve.shape[0] + Vo.shape[0], A.shape[1]), dtype=A.dtype)
            V[:Ve.shape[0], idx_even] = Ve
            V[Ve.shape[0]:, idx_odd ] = Vo
            self.De = (Ve.shape[0],) + self.De[1:]
            self.Do = (Vo.shape[0],) + self.Do[1:]
            self.data = np.reshape(V, (V.shape[0],) + s[1:])
            assert self.is_parity_consistent()
            return Re, Ro
        else:
            raise ValueError(f"invalid method = {method}, must be 'qr' or 'svd'")

    def is_parity_consistent(self) -> bool:
        """
        Test whether the sparsity pattern of the tensor entries matches
        the parity constraints (overall parity must be even).
        """
        shape_even = self.shape("even")
        it = np.nditer(self.data, flags=["multi_index"], op_flags=["readonly"])
        while not it.finished:
            idx = it.multi_index
            p = 0
            for i in range(len(idx)):
                if idx[i] >= shape_even[i]:
                    p = 1 - p
            if p == 1 and self.data[idx] != 0:
                return False
            it.iternext()
        return True


class fTTNNode:
    """
    Fermionic tree tensor network node.

    Member variables:
        tensor: fermionic tree tensor
        tid:    tensor ID
        cids:   IDs of connected tensors (parent and children),
                same order as virtual bonds of tensor

    Assuming that parent bond corresponds to first virtual bond.
    Root node has a dummy parent bond.
    """
    def __init__(self, tensor: fTTNTensor, tid: int, cids: Sequence[int]):
        if len(cids) != tensor.nvbonds:
            raise ValueError("number of virtual bonds must match number of connected nodes")
        self.tensor = tensor
        self.tid = tid
        self.cids = tuple(cids)

    @property
    def physical(self) -> bool:
        """
        Whether the node tensor has a physical axis.
        """
        return self.tensor.physical

    @property
    def nvbonds(self) -> int:
        """
        Number of virtual bonds.
        """
        return self.tensor.nvbonds

    def Dparent(self, parity="both"):
        """
        Virtual parent bond dimension.
        """
        if parity == "both":
            return self.tensor.De[0] + self.tensor.Do[0]
        elif parity == "even" or parity == 0:
            return self.tensor.De[0]
        elif parity == "odd" or parity == 1:
            return self.tensor.Do[0]
        else:
            raise ValueError(f"invalid argument parity = {parity}")


class fTTNS:
    """
    Fermionic tree tensor network state.
    """
    def __init__(self, nodes: Sequence[fTTNNode], rootid: int):
        self.nodes = { node.tid: node for node in nodes }
        self.rootid = rootid

    def add_node(self, node: fTTNNode):
        """
        Add a node.
        """
        if node.tid in self.nodes:
            raise ValueError(f"node with ID {node.tid} already exists")
        self.nodes[node.tid] = node

    def orthonormalize(self, method: str):
        """
        Orthonormalize the tree (bring to canonical form) using QR or SVD decompositions.
        """
        Re, Ro = _orthonormalize_fttn(self.nodes, self.rootid, method)
        assert self.is_consistent()
        return Re, Ro

    def contract_vector(self) -> fTTNTensor:
        """
        Contract all nodes to obtain the vector representation on the full Hilbert space.
        Returns a fTTNTensor with one dummy parent virtual bond and a single physical axis.
        """
        return _contract_fttn(self.nodes, self.rootid)

    def is_consistent(self) -> bool:
        """
        Internal consistency checks.
        """
        return _is_consistent_fttn(self.nodes, self.rootid)


def _orthonormalize_fttn(nodes: dict, tid: int, method: str):
    """
    Orthonormalize the (sub)tree with root `tid`.
    """
    node = nodes[tid]
    # recursively orthonormalize the child subtrees
    A = node.tensor.data
    for i, cid in enumerate(node.cids):
        if i == 0: # parent bond
            continue
        Re, Ro = _orthonormalize_fttn(nodes, cid, method)
        # absorb 'Re' and 'Ro'
        s = A.shape
        ax = i
        if node.physical and i >= node.tensor.physax:
            ax += 1
        A = A.reshape((math.prod(s[:ax]), s[ax], math.prod(s[ax+1:])))
        A = np.concatenate((np.einsum(A[:, :Re.shape[0], :], (0, 3, 2), Re, (3, 1), (0, 1, 2)),
                            np.einsum(A[:, Re.shape[0]:, :], (0, 3, 2), Ro, (3, 1), (0, 1, 2))), axis=1)
        A = A.reshape(s[:ax] + (Re.shape[1] + Ro.shape[1],) + s[ax+1:])
        node.tensor.data = A
        node.tensor.De = node.tensor.De[:i] + (Re.shape[1],) + node.tensor.De[i+1:]
        node.tensor.Do = node.tensor.Do[:i] + (Ro.shape[1],) + node.tensor.Do[i+1:]
    return node.tensor.orthonormalize(method)


def _is_consistent_fttn(nodes: dict, tid: int) -> bool:
    """
    Internal consistency checks for a tree tensor network state.
    """
    if tid not in nodes:
        return False
    node = nodes[tid]
    if len(node.cids) != node.nvbonds:
        return False
    if not node.tensor.is_parity_consistent():
        return False
    for i, cid in enumerate(node.cids):
        if i == 0: # parent bond
            continue
        if not _is_consistent_fttn(nodes, cid):
            return False
        # virtual bond dimensions connected to child node must match
        child = nodes[cid]
        if child.tensor.De[0] != node.tensor.De[i]:
            return False
        if child.tensor.Do[0] != node.tensor.Do[i]:
            return False
    return True


def _contract_fttn(nodes: dict, tid: int) -> fTTNTensor:
    """
    Recursively contract a fermionic tree tensor network,
    returning a fTTNTensor with a single parent virtual bond and a single physical axis.
    """
    node = nodes[tid]
    A = node.tensor.data
    for i, cid in enumerate(node.cids):
        if i == 0: # parent bond
            continue
        Dce = node.tensor.De[i]
        Dco = node.tensor.Do[i]
        C = _contract_fttn(nodes, cid)
        assert C.nvbonds == 1
        assert C.D == (Dce + Dco,)
        # contract along virtual bond to child
        s = A.shape
        ax = i
        if node.physical and i >= node.tensor.physax:
            ax += 1
        A = A.reshape((math.prod(s[:ax]), s[ax], math.prod(s[ax+1:])))
        A = np.einsum(A, (0, 3, 2), C.data, (3, 1), (0, 1, 2))
        A = A.reshape(s[:ax] + (C.d,) + s[ax+1:])
    if node.physical:
        # dimension of native physical axis
        assert A.shape[node.tensor.physax] == node.tensor.d
    else:
        # first virtual bond is parent bond
        assert len(node.cids) > 1, "node without physical axis must have child nodes"
    Dpe = node.Dparent("even")
    Dpo = node.Dparent("odd")
    # iteratively merge physical dimensions from children and native physical axis
    while A.ndim > 2:
        s = A.shape
        A = A.reshape((math.prod(s[:-2]),) + s[-2:])
        t = (A.shape[0], s[-2] * s[-1] // 4)
        # group according to parity
        A = np.concatenate((np.reshape(A[:, :s[-2]//2,  :s[-1]//2 ], t),
                            np.reshape(A[:,  s[-2]//2:,  s[-1]//2:], t),
                            np.reshape(A[:, :s[-2]//2,   s[-1]//2:], t),
                            np.reshape(A[:,  s[-2]//2:, :s[-1]//2 ], t)), axis=1)
        A = A.reshape(s[:-2] + (s[-2]*s[-1],))
    return fTTNTensor((Dpe,), (Dpo,), True, data=A, physax=1)
