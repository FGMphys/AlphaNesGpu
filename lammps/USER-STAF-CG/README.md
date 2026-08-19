# USER-STAF-CG — LAMMPS package for `pair_style staf/cg`

Links **libstaf_cg** (CUDA AF/force with dual cutoff + color maps + ONNX Runtime MLP).
Does **not** link full-atom `libstaf`. Filenames (`pair_staf_cg.*`) sit alongside
`pair_staf` so both packages can live in the same LAMMPS `src/` tree.

Binary: **`lmp_staf_cg`**. Do not overwrite production `lmp_staf`.

## Units and virial

STAF-CG models are trained in **eV** / **eV/Å**. Use `units metal` (press in bar)
for configurational pressure. The pair style sets `no_virial_fdotr_compute=1` and
tallies the diagonal pair virial from libstaf_cg. At `run 0` with zero velocities,
thermo `press` is the configurational (pair-virial) pressure.

Origami 1-epoch export uses intra `Rc=50` Å, so ghosts need:

```lammps
comm_modify     cutoff 50
```

(or `max(Rc_intra, Rc_inter, Rc_ang)`).

## Input snippet

```lammps
units           metal
atom_style      atomic
pair_style      staf/cg 50 50 float
pair_coeff      * * /path/to/model_onnx_float32
comm_modify     cutoff 50
newton          on
```

Optional 3rd pair_style arg: `float` (default) or `double`. Cutoffs default to
`50 50` if omitted. LAMMPS atom type can be all `1`; bead colors come from
`color_type_map.dat` in the model directory.

## Install (after libstaf_cg is built)

```bash
source /home/francegm/AlphaNesGpu/scripts/staf_gpu_env.sh
bash /home/francegm/AlphaNesGpu/lammps/USER-STAF-CG/Install.sh /path/to/lammps/src
# copies Makefile.staf_cg into src/MAKE/MINE/ when that dir exists
cd /path/to/lammps/src
make staf_cg -j
# → src/lmp_staf_cg
```

If the first `make staf_cg` reuses `Obj_staf` objects, delete `Obj_staf_cg/force.o`
and rebuild so `style_pair.h` picks up `staf/cg`.

## Tests

Parity (E / F / configurational P vs Python STAF-CG):
`test/test-lammps-staf-cg-parity/`
