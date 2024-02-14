import unittest
import abc
import numpy as np
import fermitensor as ftn


class TestfTTN(unittest.TestCase):

    def test_orthonormalization(self):
        """
        Test orthonormalization of a tree tensor network state.
        """
        rng = np.random.default_rng()
        for method in ["qr", "svd"]:
            scaffold = TreeScaffoldPhysicalNode(4, (1, 1), 4, 2, [
                TreeScaffoldPhysicalNode (3, (3, 2), 6, 1, []),
                TreeScaffoldBranchingNode(7, (5, 4), [
                    TreeScaffoldPhysicalNode(2, (4, 1), 4, 1, []),
                    TreeScaffoldPhysicalNode(9, (2, 3), 2, 1, [])]),
                TreeScaffoldPhysicalNode (1, (7, 2), 4, 1, [])])
            ttn = construct_random_tree(scaffold, rng)
            self.assertTrue(ttn.is_consistent())
            # contract tree tensor network state
            tvec_ref = ttn.contract_vector()
            # orthonormalize
            Re, Ro = ttn.orthonormalize(method=method)
            self.assertEqual(Re.shape, (1, 1))
            self.assertEqual(Ro.shape, (1, 1))
            # contract again after orthonormalization
            tvec = ttn.contract_vector()
            self.assertAlmostEqual(np.linalg.norm(tvec.data[0, :]), 1)
            self.assertAlmostEqual(np.linalg.norm(tvec.data[1, :]), 1)
            self.assertTrue(np.allclose(Re[0, 0] * tvec.data[0, :], tvec_ref.data[0, :]))
            self.assertTrue(np.allclose(Ro[0, 0] * tvec.data[1, :], tvec_ref.data[1, :]))

    def test_contract_tree(self):
        """
        Test tree contraction, by constructing a tree with MPS layout
        and comparing with a corresponding fMPS.
        """
        rng = np.random.default_rng()

        # reference fMPS
        # physical and virtual bond dimensions
        d  = [6, 4, 8, 6]
        De = [1, 3, 7, 6, 1]
        Do = [1, 4, 5, 2, 1]
        A = [ftn.fMPSTensor(0.5*ftn.crandn((d[i]//2, De[i], De[i+1]), rng),
                            0.5*ftn.crandn((d[i]//2, De[i], Do[i+1]), rng),
                            0.5*ftn.crandn((d[i]//2, Do[i], De[i+1]), rng),
                            0.5*ftn.crandn((d[i]//2, Do[i], Do[i+1]), rng)) for i in range(len(De)-1)]
        # zero entries corresponding to rightmost virtual bond with odd parity,
        # since this bond cannot easily be represented by tree tensor network
        A[-1].block(0, 1, 1)[:, :, :] = 0
        A[-1].block(1, 0, 1)[:, :, :] = 0
        mps = ftn.fMPS(A)

        # construct the tree;
        # root node corresponds to leftmost MPS tensor and has a dummy parent axis
        tt = [ftn.fTTNTensor(De[i:i+2], Do[i:i+2], True, data=A[i].to_logical_tensor(), physax=1) for i in range(len(De)-2)]
        # select entries of rightmost virtual bond of MPS with even parity
        tt.append(ftn.fTTNTensor((De[-2],), (Do[-2],), True, data=A[-1].to_logical_tensor()[:, :, :1].reshape((-1, A[-1].d)), physax=1))
        node7 = ftn.fTTNNode(tt[0], 7, (-1, 4))
        node4 = ftn.fTTNNode(tt[1], 4, ( 7, 5))
        node5 = ftn.fTTNNode(tt[2], 5, ( 4, 9))
        node9 = ftn.fTTNNode(tt[3], 9, ( 5,))
        ttn = ftn.fTTNS([node7, node4, node5, node9], 7)
        self.assertTrue(ttn.is_consistent())
        tvec = ttn.contract_vector()
        self.assertTrue(np.allclose(tvec.data[0, :tvec.d//2 ], mps.as_vector("even")))
        self.assertTrue(np.allclose(tvec.data[1,  tvec.d//2:], mps.as_vector("odd")))


class AbstractTreeScaffoldNode(abc.ABC):
    """
    Scaffold node base class for constructing a fermionic tree tensor network state,
    storing the bond parities of the parent bond connection
    and physical dimension (if present).
    """
    def __init__(self, tid: int, Dparent: tuple[int], children: list):
        self.tid = tid
        self.Dparent = Dparent
        self.children = children

    @property
    def De(self) -> tuple:
        """
        Dimensions of the virtual bonds with even parity.
        """
        return (self.Dparent[0],) + tuple(child.Dparent[0] for child in self.children)

    @property
    def Do(self) -> tuple:
        """
        Dimensions of the virtual bonds with odd parity.
        """
        return (self.Dparent[1],) + tuple(child.Dparent[1] for child in self.children)


class TreeScaffoldBranchingNode(AbstractTreeScaffoldNode):
    """
    Scaffold node without a physical axis for constructing a fermionic tree tensor network state.
    """
    def __init__(self, tid: int, Dparent: tuple[int], children: list):
        super().__init__(tid, Dparent, children)


class TreeScaffoldPhysicalNode(AbstractTreeScaffoldNode):
    """
    Scaffold node containing a physical axis for constructing a fermionic tree tensor network state.
    """
    def __init__(self, tid: int, Dparent: tuple[int], d: int, physax: int, children: list):
        super().__init__(tid, Dparent, children)
        self.d = d
        self.physax = physax


def construct_random_tree(srootnode: AbstractTreeScaffoldNode, rng: np.random.Generator, state: ftn.fTTNS = None):
    """
    Construct a fermionic tree tensor network state with random entries of the nodes.
    """
    if srootnode.tid == -1:
        raise ValueError("tensor ID cannot be -1 (used for dummy parent node ID)")
    if state is None:
        state = ftn.fTTNS({}, srootnode.tid)
    # recursive function call for child nodes
    for child in srootnode.children:
        state = construct_random_tree(child, rng, state)
        # set actual parent node ID
        cnode = state.nodes[child.tid]
        cnode.cids = (srootnode.tid,) + cnode.cids[1:]
    # generate tensor with random entries for current node
    De = srootnode.De
    Do = srootnode.Do
    Dtot = tuple(e + o for e, o in zip(De, Do))
    if isinstance(srootnode, TreeScaffoldBranchingNode):
        shape = Dtot
        dim_even = De
    elif isinstance(srootnode, TreeScaffoldPhysicalNode):
        shape = Dtot[:srootnode.physax] + (srootnode.d,) + Dtot[srootnode.physax:]
        dim_even = De[:srootnode.physax] + (srootnode.d // 2,) + De[srootnode.physax:]
    else:
        assert False
    data = np.zeros(shape, dtype=complex)
    # fill with random entries, respecting parity constraints
    it = np.nditer(data, flags=["multi_index"], op_flags=["readonly"])
    while not it.finished:
        idx = it.multi_index
        p = 0
        for i in range(len(idx)):
            if idx[i] >= dim_even[i]:
                p = 1 - p
        if p == 0:
            data[idx] = 0.5 * ftn.crandn(rng=rng)
        it.iternext()
    if isinstance(srootnode, TreeScaffoldBranchingNode):
        tensor = ftn.fTTNTensor(De, Do, False, data)
    elif isinstance(srootnode, TreeScaffoldPhysicalNode):
        tensor = ftn.fTTNTensor(De, Do, True, data, srootnode.d, srootnode.physax)
    else:
        assert False
    state.add_node(ftn.fTTNNode(tensor, srootnode.tid, (-1,) + tuple(child.tid for child in srootnode.children)))
    return state


if __name__ == "__main__":
    unittest.main()
