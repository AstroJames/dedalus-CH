# Beattie 2026 Dedalus KHI Validation

Minimal Dedalus code used for the Kelvin--Helmholtz validation runs in
Beattie (2026), "Cole--Hopf-Type Transform for the Isothermal Compressible
Navier--Stokes Equations."

The repository contains the direct isothermal compressible Navier--Stokes
solver, the transformed-variable solver, the lockstep comparison driver, and
the plotting scripts used to make the manuscript validation figures.

## Contents

- `direct_ivp_2d.py`: direct 2D isothermal compressible Navier--Stokes IVP.
- `transformed_ivp_2d.py`: transformed-variable RK solver using Dedalus fields
  and periodic Poisson LBVPs.
- `compare_khi_v2.py`: lockstep direct/transformed KHI comparison driver.
- `poisson_projection.py`: zero-mean periodic Poisson solver helper.
- `plot_compare_khi_v2.py`: diagnostic PNG plots from comparison HDF5 output.
- `make_khi_manuscript_figures.py`: manuscript-ready PDF figures from
  comparison HDF5 output.
- `test_poisson_projection.py`: smoke test for the Poisson projection helper.
- `check_rhs_consistency.py`: pointwise RHS consistency diagnostic.

Generated HDF5 outputs and figures are intentionally not tracked.

## Environment

The runs were developed with Dedalus v3 in a conda environment. A minimal
environment can be created with:

```bash
conda env create -f environment.yml
conda activate beattie2026-dedalus
```

On some local machines, single-rank OpenMPI runs are quieter with:

```bash
export OMPI_MCA_btl=self
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp
```

## Smoke Checks

Check the periodic Poisson helper:

```bash
python test_poisson_projection.py --nx 32 --ny 32
```

Check transformed/direct RHS consistency:

```bash
python check_rhs_consistency.py --nx 32 --ny 32
```

Run a short direct/transformed comparison:

```bash
python -u compare_khi_v2.py \
  --nx 64 --ny 64 \
  --t-end 0.02 --dt 0.0005 \
  --diag-dt 0.005 --snapshot-dt 0.005 \
  --output output/khi_v2_compare_64_t002.h5
```

For a quick MPI smoke run without writing snapshots:

```bash
mpirun -n 2 python -u compare_khi_v2.py \
  --nx 32 --ny 32 \
  --t-end 0.004 --dt 0.0005 \
  --diag-dt 0.002 \
  --mpi-smoke
```

## Manuscript Validation Run

The manuscript figures were generated from a longer comparison output. The
run corresponding to the current 128^2 validation case is:

```bash
python -u compare_khi_v2.py \
  --nx 128 --ny 128 \
  --t-end 1.8849555921538759 \
  --dt 0.0005 \
  --diag-dt 0.18849555921538758 \
  --snapshot-dt 0.18849555921538758 \
  --output output/khi_v2_compare_128_t10tsh_dealias.h5
```

Here `t_end = 10 t_sh`, with
`t_sh = khi_width_fraction * ly / khi_u0`.

Create diagnostic PNG plots:

```bash
python plot_compare_khi_v2.py \
  output/khi_v2_compare_128_t10tsh_dealias.h5 \
  --output-dir output/khi_v2_compare_128_t10tsh_plots
```

Create the manuscript PDF figures:

```bash
python make_khi_manuscript_figures.py \
  output/khi_v2_compare_128_t10tsh_dealias.h5 \
  --output-dir figures
```

## Notes

The direct solver uses Dedalus `RK443`; the transformed solver uses an explicit
RK4 driver because the transformed RHS requires nonlinear Poisson projections
at each stage. The comparison therefore tests the full reconstruction pathway
and includes the residual effect of the different timestepper implementations.
