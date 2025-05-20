"""Molecular constants of benzal chloride molecule"""

import numpy as np
from richmol.asymtop import atom_mass

# Ab initio calculations performed in Orca
#   geometry optimization, dipole moment and polarizability: B3LYP/def2-TZVPP
#   electric field gradient: RKS+DKH2/DKH-def2-TZVP

atom_labels = [
    "Cl",
    "Cl",
    "C",
    "C",
    "C",
    "C",
    "C",
    "C",
    "C",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
]

atom_masses = [atom_mass(atom) for atom in atom_labels]

# Cartesian coordinates of atoms from file:
#   etc/data/benzal_chloride/benzal_chloride_b3lyp_def2tzvpp_gopt.xyz

# Cartesian coordinates in Angstrom
atom_xyz = np.array(
    [
        [2.53472848782682, -0.16102191079696, -1.47533772496125],
        [2.53474299300801, -0.16316662563317, 1.47517546572605],
        [0.24840819569841, 0.20233613374586, 0.00018661211263],
        [-0.64068102335793, 1.27428478025160, 0.00081903411514],
        [-0.24820006145181, -1.10182312870802, -0.00064775915707],
        [1.71950353838917, 0.48474796831750, 0.00039650463350],
        [-2.01409910612749, 1.04896380692966, 0.00061990571895],
        [-1.61628180274550, -1.32486894586471, -0.00084207221412],
        [-2.50346093349892, -0.25026279308798, -0.00021131814678],
        [-0.26185913521701, 2.28891120618995, 0.00146679383123],
        [0.43892671792030, -1.93733470382279, -0.00114439389832],
        [1.92010313574198, 1.54853896438079, 0.00116469460244],
        [-2.69583866771020, 1.88917513731446, 0.00111323139944],
        [-1.99452267977882, -2.33873833253597, -0.00149158926928],
        [-3.57097965869701, -0.42903155668024, -0.0003673844925],
    ]
)

# Dipole moment and polarizability from file:
#   etc/data/benzal_chloride/benzal_chloride_b3lyp_def2tzvpp_pol.out

# molecular-frame dipole moment in au
dip_mol = [-0.872996310, 0.325848173, 0.000236011]

# molecular-frame polarizability moment in au
pol_mol = [
    [132.842028851, -1.854948137, 0.000315944],
    [-1.854948137, 98.533145964, 0.008635983],
    [0.000315944, 0.008635983, 81.749074699],
]

# Electric field gradient from file:
#   etc/data/benzal_chloride/benzal_chloride_rks_def2tzvpp_efg.out

# molecular-frame EFG tensor on Cl1 in au**-3
efg_mol_cl1 = [
    [-0.8231340, -0.9867321, -2.1798057],
    [-0.9867321, -1.2636775, 1.6388787],
    [-2.1798057, 1.6388787, 2.0868116],
]

# molecular-frame EFG tensor on Cl2 in au**-3
efg_mol_cl2 = [
    [-0.8231915, -0.9899681, 2.1783056],
    [-0.9899681, -1.2588640, -1.6436744],
    [2.1783056, -1.6436744, 2.0820556],
]
