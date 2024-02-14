from typing import Sequence
import numpy as np
from fermitensor.util import truncated_split


class fMPSTensor:
    """
    Fermionic matrix product state (fMPS) tensor of logical shape `(DL, d, DR)`
    with `d` the physical dimension and `DL`, `DR` the left and right virtual bond dimensions.
    The tensor has even overall parity, and only stores the non-zero blocks
    compatible with the parity, see Eq. (19) in Bultinck et al.
    Assuming that `d` is even, and that the first half of the physical indices
    have even parity and the second half odd parity.
    The virtual bond dimensions are partitioned as DL = DLe + DLo and DR = DRe + DRo.

    Reference:
        Nick Bultinck, Dominic J. Williamson, Jutho Haegeman, Frank Verstraete
        Fermionic matrix product states and one-dimensional topological phases
        Phys. Rev. B 95, 075108 (2017)
    """
    def __init__(self, Aee, Aeo, Aoe, Aoo):
        Aee = np.array(Aee, copy=False)
        Aeo = np.array(Aeo, copy=False)
        Aoe = np.array(Aoe, copy=False)
        Aoo = np.array(Aoo, copy=False)
        if Aee.ndim != 3 or Aeo.ndim != 3 or Aoe.ndim != 3 or Aoo.ndim != 3:
            raise ValueError("expecting tensors of degree 3")
        if len(set((Aee.shape[0], Aeo.shape[0], Aoe.shape[0], Aoo.shape[0]))) != 1:
            raise ValueError("physical even and odd parity dimensions must match")
        if (Aee.shape[1] != Aeo.shape[1] or Aee.shape[2] != Aoe.shape[2] or
            Aoo.shape[1] != Aoe.shape[1] or Aoo.shape[2] != Aeo.shape[2]):
            raise ValueError("incompatible virtual bond dimensions")
        # store blocks in a single tensor
        self.blocks = np.block([[Aee, Aeo], [Aoe, Aoo]])
        assert self.blocks.ndim == 3
        assert self.blocks.shape[0] == Aee.shape[0]
        # store virtual bond dimensions with even parity
        self.DLe = Aee.shape[1]
        self.DRe = Aee.shape[2]

    @property
    def d(self) -> int:
        """
        Logical physical dimension.
        """
        return 2*self.blocks.shape[0]

    @property
    def DLo(self) -> int:
        """
        Left virtual bond dimension with odd parity.
        """
        return self.blocks.shape[1] - self.DLe

    @property
    def DRo(self) -> int:
        """
        Right virtual bond dimension with odd parity.
        """
        return self.blocks.shape[2] - self.DRe

    @property
    def DL(self) -> int:
        """
        Overall left virtual bond dimension.
        """
        return self.blocks.shape[1]

    @property
    def DR(self) -> int:
        """
        Overall right virtual bond dimension.
        """
        return self.blocks.shape[2]

    @property
    def dtype(self):
        """
        Data type of tensor entries.
        """
        return self.blocks.dtype

    def block(self, pd: int, pL: int, pR: int) -> np.ndarray:
        """
        Extract a parity block from the logical fMPS tensor.
        """
        if (pd + pL + pR) % 2 != 0:
            raise ValueError("overall parity must be even")
        sL = slice(0, self.DLe) if pL == 0 else slice(self.DLe, self.DL)
        sR = slice(0, self.DRe) if pR == 0 else slice(self.DRe, self.DR)
        return self.blocks[:, sL, sR]

    @classmethod
    def from_logical_tensor(cls, A, DL: Sequence[int], DR: Sequence[int]):
        """
        Construct a fMPSTensor from its logical tensor of shape `(DLe + DLo, d, DRe + DRo)`.
        """
        A = np.asarray(A)
        if A.ndim != 3:
            raise ValueError("expecting an input tensor of degree 3")
        # unpack virtual bond dimensions
        DLe, DLo = DL
        DRe, DRo = DR
        if A.shape[1] % 2 != 0:
            raise ValueError("physical dimension must be even")
        if A.shape[0] != DLe + DLo:
            raise ValueError("overall left virtual bond dimension does not match array dimension")
        if A.shape[2] != DRe + DRo:
            raise ValueError("overall right virtual bond dimension does not match array dimension")
        dh = A.shape[1] // 2
        Aee = A[:DLe , :, :DRe ].transpose((1, 0, 2))
        Aeo = A[:DLe , :,  DRe:].transpose((1, 0, 2))
        Aoe = A[ DLe:, :, :DRe ].transpose((1, 0, 2))
        Aoo = A[ DLe:, :,  DRe:].transpose((1, 0, 2))
        # if np.linalg.norm(Aee[ dh:, :, :]) > 0:
        #     raise ValueError("Aee block does not have strict even parity")
        # if np.linalg.norm(Aeo[:dh , :, :]) > 0:
        #     raise ValueError("Aeo block does not have strict even parity")
        # if np.linalg.norm(Aoe[:dh , :, :]) > 0:
        #     raise ValueError("Aoe block does not have strict even parity")
        # if np.linalg.norm(Aoo[ dh:, :, :]) > 0:
        #     raise ValueError("Aoo block does not have strict even parity")
        Aee = Aee[:dh , :, :]
        Aeo = Aeo[ dh:, :, :]
        Aoe = Aoe[ dh:, :, :]
        Aoo = Aoo[:dh , :, :]
        return cls(Aee, Aeo, Aoe, Aoo)

    def to_logical_tensor(self) -> np.ndarray:
        """
        Construct the logical fMPS tensor of shape `(DL, d, DR)`.
        """
        A = np.zeros((self.DL, self.d, self.DR), dtype=self.blocks.dtype)
        dh = self.blocks.shape[0]
        A[:self.DLe , :dh , :self.DRe ] = np.transpose(self.block(0, 0, 0), (1, 0, 2))
        A[ self.DLe:, :dh ,  self.DRe:] = np.transpose(self.block(0, 1, 1), (1, 0, 2))
        A[:self.DLe ,  dh:,  self.DRe:] = np.transpose(self.block(1, 0, 1), (1, 0, 2))
        A[ self.DLe:,  dh:, :self.DRe ] = np.transpose(self.block(1, 1, 0), (1, 0, 2))
        return A


class fMPS:
    """
    Fermionic matrix product state (fMPS).

    Reference:
        Nick Bultinck, Dominic J. Williamson, Jutho Haegeman, Frank Verstraete
        Fermionic matrix product states and one-dimensional topological phases
        Phys. Rev. B 95, 075108 (2017)
    """
    def __init__(self, A: Sequence[fMPSTensor]):
        """
        Create a fermionic matrix product state.
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

    @classmethod
    def single_particle_state(cls, coeffs):
        """
        Construct the fMPS representation of a single-particle fermionic state.
        """
        return cls([fMPSTensor(np.ones((1, 1, 1)),           np.zeros((1, 1, 1)),
                               coeffs[i]*np.ones((1, 1, 1)), np.ones((1, 1, 1))) for i in range(len(coeffs))])

    @classmethod
    def two_particle_state(cls, coeffs1, coeffs2):
        """
        Construct the fMPS representation of a two-particle fermionic state.
        """
        if len(coeffs1) != len(coeffs2):
            raise ValueError("coefficient lists must have same length")
        Alist = []
        # i = 0: wrap-around due to trace is a special case
        Aee = np.array([[[0., 1., 1.], [0., 0., 0.], [0., 0., 0.]]])
        Aeo = np.array([[[coeffs1[0], -coeffs2[0]], [0., 0.], [0., 0.]]])
        Aoe = np.array([[[coeffs2[0], 0., 0.], [coeffs1[0], 0., 0.]]])
        Aoo = np.zeros((1, 2, 2))
        Alist.append(fMPSTensor(Aee, Aeo, Aoe, Aoo))
        for i in range(1, len(coeffs1)):
            Aee = np.identity(3).reshape((1, 3, 3))
            Aeo = np.array([[[0., 0.], [coeffs1[i], 0.], [0., -coeffs2[i]]]])
            Aoe = np.array([[[coeffs2[i], 0., 0.], [coeffs1[i], 0., 0.]]])
            Aoo = np.identity(2).reshape((1, 2, 2))
            Alist.append(fMPSTensor(Aee, Aeo, Aoe, Aoo))
        return cls(Alist)

    @classmethod
    def canonical_basis_state(cls, nsites: int, modes: Sequence[int]):
        """
        Construct a canonical basis state (Slater determinant with respect to standard basis),
        where each mode in `modes` is occupied.
        """
        modes = sorted(modes)
        if len(set(modes)) != len(modes):
            raise ValueError("modes must be pairwise different")
        if min(modes, default=0) < 0 or max(modes, default=0) >= nsites:
            raise ValueError("mode index out of range, must be between 0 and `nsites`")
        Apass = fMPSTensor(np.ones ((1, 1, 1)), np.zeros((1, 1, 1)),
                           np.zeros((1, 1, 1)), np.ones ((1, 1, 1)))
        Afill = fMPSTensor(np.zeros((1, 1, 1)), np.ones ((1, 1, 1)),
                           np.ones ((1, 1, 1)), np.zeros((1, 1, 1)))
        if len(modes) % 2 == 0:
            # "even" channel
            Apass_wrap = fMPSTensor(np.ones ((1, 1, 1)), np.zeros((1, 1, 1)),
                                    np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))
            Afill_wrap = fMPSTensor(np.zeros((1, 1, 1)), np.ones ((1, 1, 1)),
                                    np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))
        else:
            # "odd" channel
            Apass_wrap = fMPSTensor(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)),
                                    np.zeros((1, 1, 1)), np.ones ((1, 1, 1)))
            Afill_wrap = fMPSTensor(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)),
                                    np.ones ((1, 1, 1)), np.zeros((1, 1, 1)))
        return cls([Afill_wrap if 0 in modes else Apass_wrap] + [Afill if i in modes else Apass for i in range(1, nsites)])

    def as_vector(self, parity, sign_oe=1, sign_eo=-1) -> np.ndarray:
        """
        Merge all tensors to obtain the vector representation on the full Hilbert space.
        """
        # right-to-left merging for compliance with lexicographical ordering
        psi = self.A[-1]
        for i in reversed(range(len(self.A) - 1)):
            psi = merge_fmps_tensor_pair(self.A[i], psi)
        #print("as_vector() psi", psi.block(0,0,0), psi.block(1,0,1), psi.block(1, 1, 0), psi.block(0, 1,1))
        if parity in ("even", 0):
            Aee = psi.block(0, 0, 0)
            Aoo = psi.block(0, 1, 1)
            # contract leftmost and rightmost virtual bonds
            # (has no effect if these virtual bond dimensions are 1);
            # minus sign from parity matrix, due to swapping right virtual bond
            # to the front for inner product with left virtual bond
            assert Aee.shape[1] == Aee.shape[2]
            assert Aoo.shape[1] == Aoo.shape[2]
            return np.trace(Aee, axis1=1, axis2=2) - np.trace(Aoo, axis1=1, axis2=2) 
        if parity in ("odd", 1):
            Aeo = psi.block(1, 0, 1)
            Aoe = psi.block(1, 1, 0)
            # contract leftmost and rightmost virtual bonds;
            # require that even and odd virtual bond dimensions on boundary agree;
            # minus sign from convention for inserted Y matrix
            assert Aeo.shape[1] == Aeo.shape[2]
            assert Aoe.shape[1] == Aoe.shape[2]
            #return np.trace(Aoe, axis1=1, axis2=2) - np.trace(Aeo, axis1=1, axis2=2)
            return sign_oe*np.trace(Aoe, axis1=1, axis2=2) + sign_eo*np.trace(Aeo, axis1=1, axis2=2)

        if parity in ("both", [0, 1]):
            Aee = psi.block(0, 0, 0)
            Aoo = psi.block(0, 1, 1)
            Aeo = psi.block(1, 0, 1)
            Aoe = psi.block(1, 1, 0)
            even_trace = np.trace(Aee, axis1=1, axis2=2) - np.trace(Aoo, axis1=1, axis2=2)
            odd_trace = sign_oe*np.trace(Aoe, axis1=1, axis2=2) + sign_eo*np.trace(Aeo, axis1=1, axis2=2)
            print("fmps as vector shapes even odd traces", even_trace.shape, odd_trace.shape)
            return np.concatenate((even_trace, odd_trace), axis=None)
            #np.block([even_trace, odd_trace])
        raise ValueError(f"unknown argument value parity = {parity}")

    
def merge_fmps_tensor_pair(A0: fMPSTensor, A1: fMPSTensor) -> fMPSTensor:
    """
    Merge two neighboring fMPS tensors.
    """
    Aee = np.concatenate((np.reshape(np.einsum(A0.block(0, 0, 0), (0, 2, 3), A1.block(0, 0, 0), (1, 3, 4), (0, 1, 2, 4)), (-1, A0.DLe, A1.DRe)),
                          np.reshape(np.einsum(A0.block(1, 0, 1), (0, 2, 3), A1.block(1, 1, 0), (1, 3, 4), (0, 1, 2, 4)), (-1, A0.DLe, A1.DRe))), axis=0)
    Aeo = np.concatenate((np.reshape(np.einsum(A0.block(0, 0, 0), (0, 2, 3), A1.block(1, 0, 1), (1, 3, 4), (0, 1, 2, 4)), (-1, A0.DLe, A1.DRo)),
                          np.reshape(np.einsum(A0.block(1, 0, 1), (0, 2, 3), A1.block(0, 1, 1), (1, 3, 4), (0, 1, 2, 4)), (-1, A0.DLe, A1.DRo))), axis=0)
    Aoe = np.concatenate((np.reshape(np.einsum(A0.block(0, 1, 1), (0, 2, 3), A1.block(1, 1, 0), (1, 3, 4), (0, 1, 2, 4)), (-1, A0.DLo, A1.DRe)),
                          np.reshape(np.einsum(A0.block(1, 1, 0), (0, 2, 3), A1.block(0, 0, 0), (1, 3, 4), (0, 1, 2, 4)), (-1, A0.DLo, A1.DRe))), axis=0)
    Aoo = np.concatenate((np.reshape(np.einsum(A0.block(0, 1, 1), (0, 2, 3), A1.block(0, 1, 1), (1, 3, 4), (0, 1, 2, 4)), (-1, A0.DLo, A1.DRo)),
                          np.reshape(np.einsum(A0.block(1, 1, 0), (0, 2, 3), A1.block(1, 0, 1), (1, 3, 4), (0, 1, 2, 4)), (-1, A0.DLo, A1.DRo))), axis=0)
    return fMPSTensor(Aee, Aeo, Aoe, Aoo)


def split_fmps_tensor(A: fMPSTensor, d0: int, d1: int, svd_distr: str, tol=0):
    """
    Split a fMPS tensor with logical shape `D0 x d0*d1 x D2` into
    two fMPS tensors with shapes `D0 x d0 x D1` and `D1 x d1 x D2`, respectively.
    """
    assert d0 * d1 == A.d, "physical dimension of fMPS tensor must be equal to d0 * d1"
    d0h = d0 // 2
    d1h = d1 // 2

    # collect entries corresponding to inner bond with even or odd parity, respectively
    Be = np.zeros((d0h, A.DL, d1h, A.DR), A.dtype)
    Bo = np.zeros((d0h, A.DL, d1h, A.DR), A.dtype)
    for pL in [0, 1]:
        for pR in [0, 1]:
            B = A.block((pL + pR) % 2, pL, pR)
            for pd0 in [0, 1]:
                Bh = B[:d0h*d1h, :, :] if pd0 == 0 else B[d0h*d1h:, :, :]
                Bh = Bh.reshape((d0h, d1h, Bh.shape[1], Bh.shape[2])).transpose((0, 2, 1, 3))
                # store block
                s0 = slice(0, A.DLe) if pL == 0 else slice(A.DLe, A.DL)
                s1 = slice(0, A.DRe) if pR == 0 else slice(A.DRe, A.DR)
                if (pd0 + pL) % 2 == 0:
                    Be[:, s0, :, s1] = Bh
                else:
                    Bo[:, s0, :, s1] = Bh

    # perform SVD splitting, inner bond with even parity
    u, v = truncated_split(Be.reshape((d0h * A.DL, d1h * A.DR)), svd_distr, tol)
    u = u.reshape((d0h, A.DL, -1))
    v = v.reshape((-1, d1h, A.DR)).transpose((1, 0, 2))
    A0ee = u[:, :A.DLe , :]
    A0oe = u[:,  A.DLe:, :]
    A1ee = v[:, :, :A.DRe ]
    A1eo = v[:, :,  A.DRe:]

    # perform SVD splitting, inner bond with odd parity
    u, v = truncated_split(Bo.reshape((d0h * A.DL, d1h * A.DR)), svd_distr, tol)
    u = u.reshape((d0h, A.DL, -1))
    v = v.reshape((-1, d1h, A.DR)).transpose((1, 0, 2))
    A0eo = u[:, :A.DLe , :]
    A0oo = u[:,  A.DLe:, :]
    A1oe = v[:, :, :A.DRe ]
    A1oo = v[:, :,  A.DRe:]

    return (fMPSTensor(A0ee, A0eo, A0oe, A0oo),
            fMPSTensor(A1ee, A1eo, A1oe, A1oo))

def left_orthonormalize_fmps_tensor(A: fMPSTensor, Anext: fMPSTensor):
    """
    Left-orthonormalize a local fMPSTensor `A` by a QR decomposition,
    and update the fMPSTensor `Anext` at the next site.
    """
    # blocks corresponding to right virtual bond with even parity
    Aee = A.block(0, 0, 0)
    Aoe = A.block(1, 1, 0)
    # perform QR decomposition
    Q, R = np.linalg.qr(np.concatenate((Aee.reshape((-1, A.DRe)),
                                        Aoe.reshape((-1, A.DRe))), axis=0))
    # update blocks
    m = Aee.shape[0]*Aee.shape[1]
    Aee = Q[:m , :].reshape((Aee.shape[0], Aee.shape[1], -1))
    Aoe = Q[ m:, :].reshape((Aoe.shape[0], Aoe.shape[1], -1))
    # update Anext blocks: multiply with R from left
    Aee_next = np.tensordot(R, Anext.block(0, 0, 0), (1, 1)).transpose((1, 0, 2))
    Aeo_next = np.tensordot(R, Anext.block(1, 0, 1), (1, 1)).transpose((1, 0, 2))

    # blocks corresponding to right virtual bond with odd parity
    Aeo = A.block(1, 0, 1)
    Aoo = A.block(0, 1, 1)
    # perform QR decomposition
    Q, R = np.linalg.qr(np.concatenate((Aeo.reshape((-1, A.DRo)),
                                        Aoo.reshape((-1, A.DRo))), axis=0))
    # update blocks
    m = Aeo.shape[0]*Aeo.shape[1]
    Aeo = Q[:m , :].reshape((Aeo.shape[0], Aeo.shape[1], -1))
    Aoo = Q[ m:, :].reshape((Aoo.shape[0], Aoo.shape[1], -1))
    # update Anext blocks: multiply with R from left
    Aoe_next = np.tensordot(R, Anext.block(1, 1, 0), (1, 1)).transpose((1, 0, 2))
    Aoo_next = np.tensordot(R, Anext.block(0, 1, 1), (1, 1)).transpose((1, 0, 2))

    return fMPSTensor(Aee, Aeo, Aoe, Aoo), fMPSTensor(Aee_next, Aeo_next, Aoe_next, Aoo_next)


def right_orthonormalize_fmps_tensor(A: fMPSTensor, Aprev: fMPSTensor):
    """
    Right-orthonormalize a local fMPSTensor `A` by a QR decomposition,
    and update the fMPSTensor `Aprev` at the previous site.
    """
    # blocks corresponding to left virtual bond with even parity
    Aee = A.block(0, 0, 0)
    Aeo = A.block(1, 0, 1)
    # flip left and right virtual bond dimensions and perform QR decomposition
    Q, R = np.linalg.qr(np.concatenate((Aee.transpose((0, 2, 1)).reshape((-1, A.DLe)),
                                        Aeo.transpose((0, 2, 1)).reshape((-1, A.DLe))), axis=0))
    # update blocks
    m = Aee.shape[0]*Aee.shape[2]
    Aee = Q[:m , :].reshape((Aee.shape[0], Aee.shape[2], -1)).transpose((0, 2, 1))
    Aeo = Q[ m:, :].reshape((Aeo.shape[0], Aeo.shape[2], -1)).transpose((0, 2, 1))
    # update Aprev blocks: multiply with R from right
    Aee_prev = np.tensordot(Aprev.block(0, 0, 0), R, (2, 1))
    Aoe_prev = np.tensordot(Aprev.block(1, 1, 0), R, (2, 1))

    # blocks corresponding to left virtual bond with odd parity
    Aoe = A.block(1, 1, 0)
    Aoo = A.block(0, 1, 1)
    # flip left and right virtual bond dimensions and perform QR decomposition
    Q, R = np.linalg.qr(np.concatenate((Aoe.transpose((0, 2, 1)).reshape((-1, A.DLo)),
                                        Aoo.transpose((0, 2, 1)).reshape((-1, A.DLo))), axis=0))                                  
    # update blocks
    m = Aoe.shape[0]*Aoe.shape[2]
    Aoe = Q[:m , :].reshape((Aoe.shape[0], Aoe.shape[2], -1)).transpose((0, 2, 1))
    Aoo = Q[ m:, :].reshape((Aoo.shape[0], Aoo.shape[2], -1)).transpose((0, 2, 1))
    # update Aprev blocks: multiply with R from right
    Aeo_prev = np.tensordot(Aprev.block(1, 0, 1), R, (2, 1))
    Aoo_prev = np.tensordot(Aprev.block(0, 1, 1), R, (2, 1))

    return fMPSTensor(Aee, Aeo, Aoe, Aoo), fMPSTensor(Aee_prev, Aeo_prev, Aoe_prev, Aoo_prev)