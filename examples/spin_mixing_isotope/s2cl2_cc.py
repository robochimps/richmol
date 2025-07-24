"""Molecular constants of S-Cl-Cl-S"""

import numpy as np
from richmol.asymtop import atom_mass
from scipy import constants

# Ab initio calculations performed in CFOUR
#   CCSD(T)/aug-cc-pVTZ

atom_labels = ["S", "S", "Cl", "Cl"]
atom_labels_35_37 = ["S", "S", "Cl35", "Cl37"]

atom_masses = [atom_mass(atom) for atom in atom_labels]
atom_masses_35_37 = [atom_mass(atom) for atom in atom_labels_35_37]

# Cartesian coordinates in Angstrom
atom_xyz = (
    np.array(
        [
            [-1.06826080, 1.53616043, 1.44584157],
            [1.06826080, -1.53616043, 1.44584157],
            [0.38167705, 3.93103765, -1.32193496],
            [-0.38167705, -3.93103765, -1.32193496],
        ]
    )
    * constants.value("Bohr radius")
    * 1e10
)

# molecular-frame dipole moment in au
dip_mol = [0.0, 0.0, 0.3651417613]

# molecular-frame EFG tensor on Cl1 in au**-3
efg_mol_cl1 = [
    [-1.1619194649, 1.1270882794, -1.5436752004],
    [1.1270882794, 0.3458566615, -2.4497318836],
    [-1.5436752004, -2.4497318836, 0.8160628044],
]

# molecular-frame EFG tensor on Cl2 in au**-3
efg_mol_cl2 = [
    [-1.1619194649, 1.1270882794, 1.5436752004],
    [1.1270882794, 0.3458566615, 2.4497318836],
    [1.5436752004, 2.4497318836, 0.8160628044],
]
