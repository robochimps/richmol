# Richmol

**Richmol** is a Python package for simulations of rotational and rovibrational molecular spectra and dynamics in external fields.

The package is developed as part of ongoing research projects.
Features are added based on project requirements and community interest.

## Features
Richmol currently supports:
- Rigid rotor dynamics in external fields, including dipole and polarizability interactions.
- Absorption spectra involving dipole and quadrupole transitions, Raman transitions.
- Nuclear-spin hyperfine effects, with current support for quadrupole interactions.

The following capabilities are straightforward to add, depending on research needs and community contributions:
- Rotational dynamics using Watson-type effective Hamiltonians.
- Vibrational effects using external matrix elements from programs like [TROVE](https://github.com/Trovemaster/TROVE) or [vibrojet](https://github.com/robochimps/vibrojet).
- Higher-order molecule-field interaction tensors, such as hyperpolarizability.
- Spin-rotation and spin-spin hyperfine interactions.

## Installation
Clone the repository and run
```bash
bash build_and_install.sh
```
<!-- ``` -->
<!-- pip install --upgrade git+https://github.com/robochimps/richmol.git -->
<!-- ``` -->

## Examples

Several examples can be found in the [examples](examples) folder.

Documentation is a work in progress ...

## Questions and contact

If you have questions regarding existing functionality or future requests, please open an issue or reach out to the authors directly at andrey.yachmenev@robochimps.com