# Deep Learning for High Contrast Imaging

## Setup

The database must be located in `./data/db_new` (for the Inria cluster, replace it with a symlink to `/scratch/vasher/tbodrito/exo/data/db_new`).


## MD-REX

Multi-Distribution Reconstruction of Extended circumstellar environment (MD-REX) is a variationnal approach toolbox that minimize a residual log-likelihood with constraints over disk smoothness and sparsity.

The log-likelihood is comuputed over different distributions of patchs, listed below: 
* * [
* *  patches 8x8   + no symmetry, patches 8x8   + 180deg symmetry, patches   8x8 + 180&90deg symmetry,
* *  patches 16x16 + no symmetry, patches 16x16 + 180deg symmetry, patches 16x16 + 180&90deg symmetry,
* *  patches 32x32 + no symmetry, patches 32x32 + 180deg symmetry, patches 32x32 + 180&90deg symmetry,
* *  patches 64x64 + no symmetry, patches 32x32 + 180deg symmetry, patches 32x32 + 180&90deg symmetry,
* * ]

The MD-REX functionnal is minimized using the VMLMB software available at
<https://github.com/emmt/VMLMB>, which is a limited-memory BFGS algorithm. The package  located in `/reconstruction/mdrex.py`. Some examples of use are given in the folder `/scripts/.
