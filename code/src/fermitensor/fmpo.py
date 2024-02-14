from typing import Sequence
import numpy as np
from scipy.linalg import block_diag


class fMPOTensor:
    """
    Fermionic matrix product operator (fMPO) tensor of logical shape `(DL, d, d, DR)`
    with `d` the physical dimension and `DL`, `DR` the left and right virtual bond dimensions.
    Assuming that `d` is even, and that the first half of the physical indices
    have even parity and the second half odd parity.
    The tensor has even overall parity, and only stores the non-zero blocks
    compatible with the parity: the input (second) physical dimension of each block
    has dimension d/2 due to the parity constraint.
    The virtual bond dimensions are partitioned as DL = DLe + DLo and DR = DRe + DRo.
    """
    def __init__(self, Aee, Aeo, Aoe, Aoo):
        Aee = np.asarray(Aee)
        Aeo = np.asarray(Aeo)
        Aoe = np.asarray(Aoe)
        Aoo = np.asarray(Aoo)
        if Aee.ndim != 4 or Aeo.ndim != 4 or Aoe.ndim != 4 or Aoo.ndim != 4:
            raise ValueError("expecting tensors of degree 4")
        if len(set((Aee.shape[0], Aeo.shape[0], Aoe.shape[0], Aoo.shape[0]))) != 1:
            raise ValueError("physical output dimensions must match")
        if len(set((Aee.shape[1], Aeo.shape[1], Aoe.shape[1], Aoo.shape[1]))) != 1:
            raise ValueError("physical input dimensions must match")
        # factor 2 since only half of physical input indices are compatible with overall even parity
        if Aee.shape[0] != 2*Aee.shape[1]:
            raise ValueError("incompatible physical input and output dimensions")
        if (Aee.shape[2] != Aeo.shape[2] or Aee.shape[3] != Aoe.shape[3] or
            Aoo.shape[2] != Aoe.shape[2] or Aoo.shape[3] != Aeo.shape[3]):
            raise ValueError("incompatible virtual bond dimensions")
        # store blocks in a single tensor
        self.blocks = np.block([[Aee, Aeo], [Aoe, Aoo]])
        assert self.blocks.ndim == 4
        assert self.blocks.shape[0] == Aee.shape[0]
        assert self.blocks.shape[1] == Aee.shape[1]
        # store virtual bond dimensions with even parity
        self.DLe = Aee.shape[2]
        self.DRe = Aee.shape[3]

    @property
    def d(self) -> int:
        """
        Logical physical dimension.
        """
        return self.blocks.shape[0]

    @property
    def DLo(self) -> int:
        """
        Left virtual bond dimension with odd parity.
        """
        return self.blocks.shape[2] - self.DLe

    @property
    def DRo(self) -> int:
        """
        Right virtual bond dimension with odd parity.
        """
        return self.blocks.shape[3] - self.DRe

    @property
    def DL(self) -> int:
        """
        Overall left virtual bond dimension.
        """
        return self.blocks.shape[2]

    @property
    def DR(self) -> int:
        """
        Overall right virtual bond dimension.
        """
        return self.blocks.shape[3]

    @property
    def dtype(self):
        """
        Data type of tensor entries.
        """
        return self.blocks.dtype

    def block(self, pdo: int, pdi: int, pL: int, pR: int) -> np.ndarray:
        """
        Extract a parity block from the logical fMPO tensor.
        """
        if (pdo + pdi + pL + pR) % 2 != 0:
            raise ValueError("overall parity must be even")
        so = slice(0, self.d//2) if pdo == 0 else slice(self.d//2, self.d)
        sL = slice(0, self.DLe)  if pL  == 0 else slice(self.DLe,  self.DL)
        sR = slice(0, self.DRe)  if pR  == 0 else slice(self.DRe,  self.DR)
        return self.blocks[so, :, sL, sR]

    @classmethod
    def from_logical_tensor(cls, A, DL: Sequence[int], DR: Sequence[int]):
        """
        Construct a fMPOTensor from its logical tensor of shape `(DLe + DLo, d, d, DRe + DRo)`.
        """
        A = np.asarray(A)
        if A.ndim != 4:
            raise ValueError("expecting an input tensor of degree 4")
        # unpack virtual bond dimensions
        DLe, DLo = DL
        DRe, DRo = DR
        if A.shape[1] != A.shape[2]:
            raise ValueError("physical input and output dimensions must agree")
        if A.shape[1] % 2 != 0:
            raise ValueError("physical dimension must be even")
        if A.shape[0] != DLe + DLo:
            raise ValueError("overall left virtual bond dimension does not match array dimension")
        if A.shape[3] != DRe + DRo:
            raise ValueError("overall right virtual bond dimension does not match array dimension")
        dh = A.shape[1] // 2
        Aee = A[:DLe , :, :, :DRe ].transpose((1, 2, 0, 3))
        Aeo = A[:DLe , :, :,  DRe:].transpose((1, 2, 0, 3))
        Aoe = A[ DLe:, :, :, :DRe ].transpose((1, 2, 0, 3))
        Aoo = A[ DLe:, :, :,  DRe:].transpose((1, 2, 0, 3))
        if np.linalg.norm(Aee[:dh, dh:, :, :]) > 0 or np.linalg.norm(Aee[dh:, :dh, :, :]) > 0:
            raise ValueError("Aee block does not have strict even parity")
        if np.linalg.norm(Aeo[:dh, :dh, :, :]) > 0 or np.linalg.norm(Aeo[dh:, dh:, :, :]) > 0:
            raise ValueError("Aeo block does not have strict even parity")
        if np.linalg.norm(Aoe[:dh, :dh, :, :]) > 0 or np.linalg.norm(Aoe[dh:, dh:, :, :]) > 0:
            raise ValueError("Aoe block does not have strict even parity")
        if np.linalg.norm(Aoo[:dh, dh:, :, :]) > 0 or np.linalg.norm(Aoo[dh:, :dh, :, :]) > 0:
            raise ValueError("Aoo block does not have strict even parity")
        Aee = np.concatenate((Aee[:dh, :dh , :, :], Aee[ dh:,  dh:, :, :]), axis=0)
        Aeo = np.concatenate((Aeo[:dh,  dh:, :, :], Aeo[ dh:, :dh , :, :]), axis=0)
        Aoe = np.concatenate((Aoe[:dh,  dh:, :, :], Aoe[ dh:, :dh , :, :]), axis=0)
        Aoo = np.concatenate((Aoo[:dh, :dh , :, :], Aoo[ dh:,  dh:, :, :]), axis=0)
        return cls(Aee, Aeo, Aoe, Aoo)

    def to_logical_tensor(self) -> np.ndarray:
        """
        Construct the logical fMPO tensor of shape `(DL, d, d, DR)`.
        """
        A = np.zeros((self.DL, self.d, self.d, self.DR), dtype=self.blocks.dtype)
        dh = self.blocks.shape[0] // 2
        A[:self.DLe,  :dh,  :dh,  :self.DRe ] = self.block(0, 0, 0, 0).transpose((2, 0, 1, 3))
        A[:self.DLe,   dh:,  dh:, :self.DRe ] = self.block(1, 1, 0, 0).transpose((2, 0, 1, 3))
        A[ self.DLe:, :dh,  :dh,   self.DRe:] = self.block(0, 0, 1, 1).transpose((2, 0, 1, 3))
        A[ self.DLe:,  dh:,  dh:,  self.DRe:] = self.block(1, 1, 1, 1).transpose((2, 0, 1, 3))
        A[:self.DLe,  :dh,   dh:,  self.DRe:] = self.block(0, 1, 0, 1).transpose((2, 0, 1, 3))
        A[:self.DLe,   dh:, :dh,   self.DRe:] = self.block(1, 0, 0, 1).transpose((2, 0, 1, 3))
        A[ self.DLe:, :dh,   dh:, :self.DRe ] = self.block(0, 1, 1, 0).transpose((2, 0, 1, 3))
        A[ self.DLe:,  dh:, :dh,  :self.DRe ] = self.block(1, 0, 1, 0).transpose((2, 0, 1, 3))
        return A


class fMPO:
    """
    Fermionic matrix product operator (fMPO).
    """
    def __init__(self, A: Sequence[fMPOTensor]):
        """
        Create a fermionic matrix product operator.
        """
        for i in range(len(A) - 1):
            if A[i].DRe != A[i+1].DLe or A[i].DRo != A[i+1].DLo:
                raise ValueError("incompatible virtual bond dimensions between neighboring tensors")
        self.A = list(A)

    @property
    def nsites(self) -> int:
        """
        Number of lattice sites.
        """
        return len(self.A)

    def bond_dims(self, parity="both") -> list:
        """
        Virtual bond dimensions.
        """
        if self.nsites == 0:
            return []
        if parity in ("even", 0):
            D = [self.A[i].DLe for i in range(self.nsites)]
            D.append(self.A[-1].DRe)
            return D
        if parity in ("odd", 1):
            D = [self.A[i].DLo for i in range(self.nsites)]
            D.append(self.A[-1].DRo)
            return D
        if parity == "both":
            D = [self.A[i].DL for i in range(self.nsites)]
            D.append(self.A[-1].DR)
            return D
        raise ValueError(f"unknown argument value parity = {parity}")

    def as_matrix(self) -> np.ndarray:
        """
        Merge all tensors to obtain the matrix representation on the full Hilbert space.
        """
        # right-to-left merging for compliance with lexicographical ordering
        op = self.A[-1]
        for i in reversed(range(len(self.A) - 1)):
            op = merge_fmpo_tensor_pair(self.A[i], op)
        # contract leftmost and rightmost virtual bonds
        # (has no effect if these virtual bond dimensions are 1);
        # minus sign from parity matrix, due to swapping right virtual bond
        # to the front for inner product with left virtual bond
        # TODO: also compute even-odd or odd-even physical parity blocks - parity changing, as needed
        return block_diag(np.trace(op.block(0, 0, 0, 0), axis1=2, axis2=3) - np.trace(op.block(0, 0, 1, 1), axis1=2, axis2=3),
                          np.trace(op.block(1, 1, 0, 0), axis1=2, axis2=3) - np.trace(op.block(1, 1, 1, 1), axis1=2, axis2=3))


def merge_fmpo_tensor_pair(A0: fMPOTensor, A1: fMPOTensor) -> fMPOTensor:
    """
    Merge two neighboring fMPO tensors.
    """
    dh = A0.d * A1.d // 4
    A = [[None, None], [None, None]]
    DL = [A0.DLe, A0.DLo]
    DR = [A1.DRe, A1.DRo]
    for pL in [0, 1]:
        for pR in [0, 1]:
            B = [None, None]
            # overall output physical parity
            for pdo in [0, 1]:
                # overall input physical parity
                pdi = (pdo + pL + pR) % 2
                B[pdo] = np.concatenate(
                    (np.concatenate((np.reshape(np.einsum(A0.block(0, 0, pL,   pL), (0, 2, 4, 6), A1.block(  pdo,   pdi,   pL, pR), (1, 3, 6, 5), (0, 1, 2, 3, 4, 5)), (dh, dh, DL[pL], DR[pR])),
                                     np.reshape(np.einsum(A0.block(0, 1, pL, 1-pL), (0, 2, 4, 6), A1.block(  pdo, 1-pdi, 1-pL, pR), (1, 3, 6, 5), (0, 1, 2, 3, 4, 5)), (dh, dh, DL[pL], DR[pR]))), axis=1),
                     np.concatenate((np.reshape(np.einsum(A0.block(1, 0, pL, 1-pL), (0, 2, 4, 6), A1.block(1-pdo,   pdi, 1-pL, pR), (1, 3, 6, 5), (0, 1, 2, 3, 4, 5)), (dh, dh, DL[pL], DR[pR])),
                                     np.reshape(np.einsum(A0.block(1, 1, pL,   pL), (0, 2, 4, 6), A1.block(1-pdo, 1-pdi,   pL, pR), (1, 3, 6, 5), (0, 1, 2, 3, 4, 5)), (dh, dh, DL[pL], DR[pR]))), axis=1)),
                    axis=0)
            A[pL][pR] = np.concatenate(B, axis=0)

    return fMPOTensor(A[0][0], A[0][1], A[1][0], A[1][1])
