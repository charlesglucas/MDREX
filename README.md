# Deep Learning for High Contrast Imaging

## Setup

The database must be located in `./data/db_new` (for the Inria cluster, replace it with a symlink to `/scratch/vasher/tbodrito/exo/data/db_new`).

## MD-REX

Direct imaging of circumstellar disks in total intensity is extremely
challenging because disks are faint, extended structures located very close to
a star that is typically $10^{7}$ times brighter. Ground-based observations
obtained with the VLT are degraded by atmospheric turbulence, residual
wavefront errors, coronagraphic diffraction, and detector noise, producing
complex nuisance signals dominated by quasi-static speckles. Differential
imaging techniques (ADI/SDI) mitigate these effects but suffer from
self-subtraction, especially for extended disks.

Multi-Distribution Reconstruction of Extended circumstellar environment (MD-REX) is a variationnal approach toolbox to separate a disk
signal from the speckle-dominated background in high-contrast
imaging. It minimizes a residual log-likelihood while penalizing over disk smoothness and sparsity. This extends the variational REXPACO framework, available at `https://github.com/olivier-flasseur/rexpaco_demo`, by introducing a
multidistribution patch-based model for the nuisance to better capture the
heterogeneous statistics of speckle. Indeed, the log-likelihood is computed over different distributions of patchs, listed below: 
*  patches 8x8   + no symmetry, patches 8x8   + 180deg symmetry, patches   8x8 + 180&90deg symmetry,
*  patches 16x16 + no symmetry, patches 16x16 + 180deg symmetry, patches 16x16 + 180&90deg symmetry,
*  patches 32x32 + no symmetry, patches 32x32 + 180deg symmetry, patches 32x32 + 180&90deg symmetry,
*  patches 64x64 + no symmetry, patches 32x32 + 180deg symmetry, patches 32x32 + 180&90deg symmetry

The MD-REX functionnal is minimized using the VMLMB software available at
<https://github.com/emmt/VMLMB>, which is a limited-memory BFGS algorithm. The package  located in `/reconstruction/mdrex.py`. Some examples of use are given in the folder `/scripts/`.
