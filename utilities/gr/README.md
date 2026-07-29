# utilities/gr — fast multi-species g(r)

C cell-list implementation (inspired by `code_jsw_tmmc/utilities/calculate_gr.c`,
which already supports type pairs via `type_map` / `type.dat`) exposed to Python.

## Build

```bash
cd /home/francegm/AlphaNesGpu/utilities/gr
make -j
```

Produces `libstaf_gr.so`.

## Python

```bash
python python/compute_gr_cli.py \
  --traj /path/to/traj.lammpstrj \
  --T 223 --pairs OO,OH,HH \
  --min-step 10001 \
  -o /path/to/outdir
```

`--min-step 10001` keeps only frames after an NVT warm-up of 10000 steps (NPT segment).
```

Or import:

```python
from staf_gr import compute_gr_frames, read_lammpstrj
frames = list(read_lammpstrj("traj.lammpstrj"))
r, g = compute_gr_frames(frames, ta=0, tb=0)  # O–O
```

Types: 0=O, 1=H (LAMMPS dump 1/2 remapped).

## Notes

- Orthorhombic PBC, minimum image.
- Same-type pairs counted once (i<j); unlike pairs all A×B.
- Cell list uniquifies neighbor cells (correct when `ncell==1`).
