import numpy as np
from fermitensor.fmpo import fMPOTensor, fMPO


def spinless_fermi_hubbard_mpo(L: int, t: float, u: float, pbc=False) -> fMPO:
    """
    Construct spinless Fermi-Hubbard Hamiltonian
    with nearest-neighbor hopping and interaction on a 1D lattice as fMPO.

    Args:
        L:  number of lattice sites
        t:  kinetic hopping parameter
        u:  Hubbard interaction strength

    Returns:
        fMPO: spinless Fermi-Hubbard Hamiltonian
    """
    assert L >= 2
    I = np.identity(2)
    O = np.zeros((2, 2))
    # creation and annihilation operators for a single spin and lattice site
    a = np.array([[0., 1.], [0., 0.]])
    c = np.array([[0., 0.], [1., 0.]])
    # number operator
    n = np.array([[0., 0.], [0., 1.]])
    # construct bulk fMPO tensor
    Aee = np.array([
        [ I,   O,   O  ],
        [ O,   I,   u*n],
        [ n,   O,   O  ]])
    Aeo = np.array([
        [ O,   O  ],
        [-t*c,-t*a],
        [ O,   O  ]])
    Aoe = np.array([
        [ a,   O,   O],
        [ c,   O,   O]])
    Aoo = np.array([
        [ O,   O],
        [ O,   O]])
    # reorder dimensions for np.block
    Aee = Aee.transpose((2, 3, 0, 1))
    Aeo = Aeo.transpose((2, 3, 0, 1))
    Aoe = Aoe.transpose((2, 3, 0, 1))
    Aoo = Aoo.transpose((2, 3, 0, 1))
    A = fMPOTensor.from_logical_tensor(np.block([[Aee, Aeo], [Aoe, Aoo]]).transpose((2, 0, 1, 3)), (3, 2), (3, 2))
    if not pbc:
        # leftmost Aee block
        Lee = np.array([
            [ O,   I,   u*n],
            [ O,   O,   O  ],
            [ O,   O,   O  ]])
        Lee = Lee.transpose((2, 3, 0, 1))
        # leftmost Aeo block
        Leo = np.array([
            [-t*c,-t*a],
            [ O,   O  ],
            [ O,   O  ]])
        Leo = Leo.transpose((2, 3, 0, 1))
        A0 = fMPOTensor.from_logical_tensor(np.block([[Lee, Leo], [Aoe, Aoo]]).transpose((2, 0, 1, 3)), (3, 2), (3, 2))
        return fMPO([A0] + (L - 1) * [A])
    else: 
        # leftmost Aee block
        Lee = np.array([
            [ O,   I,   u*n],
            [ O,   O,   O  ],
            [ O,   O,   O  ],

            [ O,   O,   O  ], 
            [ O,   O,   O  ], 
            [ O,   O,   O  ],
            [ O,   O,   O  ] 
            ])
        Lee = Lee.transpose((2, 3, 0, 1))
        # leftmost Aeo block
        Leo = np.array([
            [-t*c,-t*a],
            [ O,   O  ],
            [ O,   O  ], 

            [ O,   O  ],
            [ O,   O  ], 
            [ O,   O  ],
            [ O,   O  ]
            
            ])
        Leo = Leo.transpose((2, 3, 0, 1))
        A0 = fMPOTensor.from_logical_tensor(np.block([[Lee, Leo], [Aoe, Aoo]]).transpose((2, 0, 1, 3)), (7, 2), (3, 2))
        Reo = np.array([
            [-t*c,-t*a,   O,   O,   O,   O],
            [ O,   O,   O,   O,   O,   O],
            [ O,   O,   O,   O,   O,   O ], 
            ])
        Reo = Reo.transpose((2, 3, 0, 1))
        Roo = np.array([
            [ O,   O,   O,   O,   O,   O],
            [ O,   O,   O,   O,   O,   O]])
        Roo = Roo.transpose((2, 3, 0, 1))
        Alast = fMPOTensor.from_logical_tensor(np.block([[Aee, Reo], [Aoe, Roo]]).transpose((2, 0, 1, 3)), (3, 2), (3, 6))
        return fMPO([A0] + L * [A] + [Alast]) #(L - 2)





    """else:
        #raise NotImplementedError
        #just adding here extra dummy tensors either side for bond dim 1 
        # leftmost dummy Aee block

        # A0i = np.zeros((1, 2, 2, 5)) #hard coded - only 5 due to above - to match 
        # A0i[0, :, :, 0] = np.identity(2)
        A0i = [1] + [0] * ((L - 1)//2) + [1] + [0] * ((L - 1)//2)
        A0i = np.reshape(A0i, (1,2,2,5))

        A0 = fMPOTensor.from_logical_tensor(A0i, [1, 0], [3, 2])

        # Alasti = np.zeros((5, 2, 2, 1))
        # Alasti[0, :, :, 0] = np.identity(2)
        Alasti = [1] + [0] * ((L - 1)//2) + [1] + [0] * ((L - 1)//2)
        Alasti = np.reshape(A0i, (5, 2, 2, 1))
        Alast = fMPOTensor.from_logical_tensor(Alasti, [3, 2], [1, 0])

        # leftmost real Aee block
        Lee = np.array([
            [ O,   I,   u*n],
            [ O,   O,   O  ],
            [ O,   O,   O  ]])
        Lee = Lee.transpose((2, 3, 0, 1))
        # leftmost Aeo block
        Leo = np.array([
            [-t*c,-t*a],
            [ O,   O  ],
            [ O,   O  ]])
        Leo = Leo.transpose((2, 3, 0, 1))
        A1 = fMPOTensor.from_logical_tensor(np.block([[Lee, Leo], [Aoe, Aoo]]).transpose((2, 0, 1, 3)), (3, 2), (3, 2))

        print("v bonds", A0.DL, A0.DR, A0.d, A1.DL, A1.DR, A1.d, Alast.DL, Alast.DR, Alast.d)
        return fMPO([A0] + [A1] + (L - 1) * [A] + [Alast])
"""