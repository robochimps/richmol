#!/bin/bash

python -m numpy.f2py -c expokit_src/expokit.pyf expokit_src/expokit.f \
    expokit_src/blas.f expokit_src/lapack.f \
    --f90flags=-O3 --f90flags=-fallow-argument-mismatch 
    # --f90flags="-O0 -g -fcheck=all"

mv expokit*.so richmol/
pip install .