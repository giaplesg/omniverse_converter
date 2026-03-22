# Omniverse Converter

PLY / Gaussian Splatting → USDZ converters for NVIDIA Omniverse and Isaac Sim.

## Scripts

### 1. `gs_add_rgb.py` — Add RGB to Gaussian Splatting PLY
Converts a Brush/3DGS PLY (with `f_dc_0/1/2` SH fields) to include standard `red/green/blue` uchar properties.
Opens in CloudCompare, MeshLab, Potree, SuperSplat, etc.

```bash
python gs_add_rgb.py input.ply [output.ply]
```

---

### 2. `gs_clean_to_usdz.py` — Clean GS PLY → USDZ Point Cloud
Full cleaning pipeline + USDZ point cloud export for Omniverse.

Cleaning steps:
- Opacity threshold (removes transparent/ghost splats)
- Scale filter (removes exploded/microscopic splats)
- Statistical Outlier Removal (SOR via scipy KDTree)
- Optional voxel downsample

```bash
python gs_clean_to_usdz.py input.ply [options]

Options:
  --opacity    0.05     min opacity after sigmoid  (default 0.05)
  --max-scale  2.0      max Gaussian radius in scene units (default 2.0)
  --min-scale  0.0001   min Gaussian radius (default 0.0001)
  --sor-k      20       SOR: number of neighbours (default 20)
  --sor-std    2.0      SOR: std-dev multiplier (default 2.0)
  --voxel      0.0      voxel down-sample size, 0 = off (default 0)
  --no-sor              skip SOR step
  --out        <path>   output stem
```

---

### 3. `gs_mesh_usdz.py` — GS PLY → Mesh via Delaunay 2D
Terrain-style meshing using 2D Delaunay triangulation.
Best for flat/aerial scans.

```bash
python gs_mesh_usdz.py input.ply [--voxel 0.25] [--alpha 1.5] [--max-edge 3.0] [--out stem]
```

---

### 4. `gs_mesh_v2.py` — GS PLY → Mesh via VTK Implicit Surface
Implicit surface reconstruction using `pyvista.reconstruct_surface()`.
Better for volumetric GS clouds.

```bash
python gs_mesh_v2.py input.ply [--voxel 0.4] [--sor-std 1.5] [--nbr 30] [--smooth 50] [--out stem]
```

---

### 5. `gs_mesh_v3.py` — GS PLY → Mesh via Screened Poisson (Best Quality)
Screened Poisson Surface Reconstruction via pymeshlab.
Highest quality mesh for complex scenes.

```bash
python gs_mesh_v3.py input.ply [--voxel 0.2] [--sor-std 1.5] [--depth 10] [--out stem]

Options:
  --voxel      0.2    voxel downsample size in metres (default 0.2)
  --sor-std    1.5    SOR std-dev multiplier (default 1.5)
  --depth      10     Poisson octree depth — higher = more detail (default 10)
  --out        stem   output file stem
```

---

## Requirements

```bash
pip install numpy scipy pyvista pymeshlab usd-core
```

> `usd-core` provides the `pxr` Python bindings for USD/USDZ export.
> `pymeshlab` wraps MeshLab algorithms (Screened Poisson, etc.).
> `pyvista` wraps VTK for surface reconstruction and mesh operations.

---

## Workflow

Typical pipeline from Brush / 3DGS export to Omniverse:

```
Brush export.ply
      │
      ▼
gs_add_rgb.py          → export_rgb.ply     (add standard RGB fields)
      │
      ▼
gs_clean_to_usdz.py    → export_clean.usdz  (point cloud for Omniverse)
      │
      ▼
gs_mesh_v3.py          → export_mesh.usdc   (polygon mesh for simulation)
```

## Notes

- Coordinate system: Y-up (set via `upAxis=Y` in USD stage metadata)
- Coordinates are auto-centered to scene centroid on export
- USDZ = ZIP_STORED archive of a `.usdc` (binary USD) file
- Isaac Sim: use `.usdc` directly via **File > Open** (USDZ requires drag-and-drop or the asset browser)
- Vertex colors are exported as `displayColor` primvar with `vertex` interpolation
