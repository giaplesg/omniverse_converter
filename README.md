# omniverse_converter

PLY / Gaussian Splatting → USDZ converters for NVIDIA Omniverse and Isaac Sim.

Two workflows are supported:

1. **Post-process** an existing Brush export with `gs_add_rgb.py` (no rebuild needed)
2. **Patch Brush source** so every future export natively includes `red/green/blue` fields

---

## Quick Start

```bash
pip install numpy scipy pyvista pymeshlab usd-core
```

---

## Workflow Overview

```
Brush  ──export──>  export.ply  (SH coefficients only, no RGB)
                        │
           ┌────────────┴────────────────────────┐
           │ Option A: post-process               │ Option B: patch Brush source
           ▼                                      ▼
   gs_add_rgb.py                          patch export.rs + cargo build
           │                                      │
           └──────────────┬──────────────────────┘
                          ▼
                  export_rgb.ply   (has red/green/blue uchar fields)
                          │
              ┌───────────┼───────────────┐
              ▼           ▼               ▼
   gs_clean_to_usdz  gs_mesh_v3    gs_mesh_usdz
   (point cloud)     (Poisson mesh) (Delaunay mesh)
              │           │               │
              ▼           ▼               ▼
          .usdz        .usdc           .usdz
       (drag into    (File > Open    (drag into
       Omniverse)    in Isaac Sim)   Omniverse)
```

---

## Scripts

### `gs_add_rgb.py` — Add RGB to existing Brush PLY (no rebuild needed)

Reads any Brush / 3DGS PLY with `f_dc_0/1/2` SH fields and adds standard
`red green blue` uchar properties by computing:

```
rgb = clamp(f_dc * 0.28209479177387814 + 0.5, 0, 1) * 255
```

Output opens directly in CloudCompare, MeshLab, Potree, SuperSplat, etc.

```bash
python gs_add_rgb.py input.ply [output.ply]
# default output: input_rgb.ply
```

---

### `gs_clean_to_usdz.py` — Clean GS PLY → USDZ Point Cloud

Full noise-removal pipeline + colored point cloud export for Omniverse.

```bash
python gs_clean_to_usdz.py input.ply [options]

Options:
  --opacity    0.05     min opacity after sigmoid  (default 0.05)
  --max-scale  2.0      max Gaussian radius in scene units (default 2.0)
  --min-scale  0.0001   min Gaussian radius (default 0.0001)
  --sor-k      20       SOR neighbours (default 20)
  --sor-std    2.0      SOR std-dev multiplier (default 2.0)
  --voxel      0.0      voxel downsample size, 0 = off (default 0)
  --no-sor              skip SOR step
  --out        <stem>   output file stem (default: input_clean)
```

---

### `gs_mesh_usdz.py` — Mesh via Delaunay 2D

Terrain/aerial mesh using 2D Delaunay triangulation projected onto the
best-fit plane. Best for flat drone scans.

```bash
python gs_mesh_usdz.py input.ply [--voxel 0.25] [--alpha 1.5] [--max-edge 3.0] [--out stem]
```

---

### `gs_mesh_v2.py` — Mesh via VTK Implicit Surface

Uses `pyvista.reconstruct_surface()` (VTK SurfaceReconstructionFilter).
Fast but coarser than Poisson.

```bash
python gs_mesh_v2.py input.ply [--voxel 0.4] [--sor-std 1.5] [--nbr 30] [--smooth 50] [--out stem]
```

---

### `gs_mesh_v3.py` — Mesh via Screened Poisson (best quality)

Uses `pymeshlab` Screened Poisson Surface Reconstruction.
Best quality mesh for complex outdoor/indoor scenes.

```bash
python gs_mesh_v3.py input.ply [--voxel 0.2] [--sor-std 1.5] [--depth 10] [--out stem]

  --voxel      0.2    voxel downsample in metres (default 0.2)
  --sor-std    1.5    SOR std-dev multiplier (default 1.5)
  --depth      10     Poisson octree depth — higher = more detail (default 10)
```

---

## Patching Brush for Native RGB Export

By default Brush exports PLY files with only SH coefficients (`f_dc_0/1/2`, `f_rest_*`).
This patch adds `red`, `green`, `blue` uchar fields to every export so the file
opens correctly in CloudCompare, MeshLab, and the scripts in this repo without
needing `gs_add_rgb.py` as a post-processing step.

### File to edit

```
<brush-repo>/crates/brush-serde/src/export.rs
```

### Step 1 — Add the `SH_C0` import (top of file)

```diff
 use brush_render::gaussian_splats::Splats;
 use brush_render::sh::sh_coeffs_for_degree;
+use brush_render::shaders::SH_C0;
 use burn::prelude::Backend;
```

### Step 2 — Add fields to `DynamicPlyGaussian` struct

```diff
 struct DynamicPlyGaussian {
     x: f32,
     y: f32,
     z: f32,
+    nx: f32,
+    ny: f32,
+    nz: f32,
+    red: u8,
+    green: u8,
+    blue: u8,
     scale_0: f32,
```

### Step 3 — Update `field_count` and serialize the new fields

Find the `impl Serialize for DynamicPlyGaussian` block and apply:

```diff
-        // Calculate total number of fields: 11 core + 3 DC + rest_coeffs
-        let field_count = 14 + self.rest_coeffs.len();
+        // Calculate total number of fields: 11 core + 3 normals + 3 rgb + 3 DC + rest_coeffs
+        let field_count = 20 + self.rest_coeffs.len();
         let mut state = serializer.serialize_struct("DynamicPlyGaussian", field_count)?;

         state.serialize_field("x", &self.x)?;
         state.serialize_field("y", &self.y)?;
         state.serialize_field("z", &self.z)?;
+        state.serialize_field("nx", &self.nx)?;
+        state.serialize_field("ny", &self.ny)?;
+        state.serialize_field("nz", &self.nz)?;
+        state.serialize_field("red", &self.red)?;
+        state.serialize_field("green", &self.green)?;
+        state.serialize_field("blue", &self.blue)?;
         state.serialize_field("scale_0", &self.scale_0)?;
```

### Step 4 — Populate the new fields when building each splat

Find the `DynamicPlyGaussian { x: means[i * 3], ...` constructor and apply:

```diff
+            // Convert DC SH component to linear RGB: rgb = f_dc * SH_C0 + 0.5
+            let to_u8 = |v: f32| (v.mul_add(SH_C0, 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
+
             DynamicPlyGaussian {
                 x: means[i * 3],
                 y: means[i * 3 + 1],
                 z: means[i * 3 + 2],
+                nx: 0.0,
+                ny: 0.0,
+                nz: 0.0,
+                red:   to_u8(sh_red[0]),
+                green: to_u8(sh_green[0]),
+                blue:  to_u8(sh_blue[0]),
                 scale_0: log_scales[i * 3],
```

### Step 5 — Rebuild Brush

> Close Brush before building — cargo cannot replace a running executable.

```bash
cd <brush-repo>
cargo build --release -p brush-app
```

The rebuilt `brush.exe` (or `brush` on Linux/macOS) will include `red/green/blue`
in every PLY export automatically.

---

## Loading in Isaac Sim

- **Point cloud** (`.usdz`): drag into the Content browser or use **File > Add Reference**
- **Mesh** (`.usdc`): **File > Open** → select the `.usdc` file directly
- Coordinates are Y-up, centered at scene centroid, 1 unit = 1 metre

---

## Requirements

| Package | Purpose |
|---------|---------|
| `numpy` | array operations |
| `scipy` | KDTree, interpolation |
| `pyvista` | VTK mesh operations |
| `pymeshlab` | Screened Poisson reconstruction |
| `usd-core` | pxr Python bindings for USD/USDZ export |

```bash
pip install numpy scipy pyvista pymeshlab usd-core
```
