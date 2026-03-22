"""
Point Cloud -> Mesh -> USDZ for Omniverse
Uses 2D Delaunay triangulation (terrain meshing) — ideal for aerial/drone scans.
Downsamples first, then reconstructs, then exports as colored USD mesh.

Usage:
    python gs_mesh_usdz.py input.ply [--voxel 0.3] [--alpha 2.0] [--out stem]
"""
import numpy as np, re, os, zipfile, argparse, time
import pyvista as pv
from scipy.spatial import KDTree
from pxr import Usd, UsdGeom, Vt, Gf

# ── Load PLY ──────────────────────────────────────────────────────────────────
def load_ply(path):
    with open(path,'rb') as f:
        lines=[]
        while True:
            l=f.readline().decode('ascii','ignore').rstrip()
            lines.append(l)
            if l=='end_header': break
        raw=f.read()
    props=[]; n=0
    for l in lines:
        m=re.match(r'element vertex (\d+)',l)
        if m: n=int(m.group(1))
        m=re.match(r'property (\w+) (\w+)',l)
        if m: props.append((m.group(1),m.group(2)))
    TYPE={'float':('f',4,np.float32),'uchar':('B',1,np.uint8)}
    dtype=np.dtype([(name,TYPE[typ][2]) for typ,name in props])
    data=np.frombuffer(raw[:n*dtype.itemsize],dtype=dtype)
    return data

# ── Voxel downsample ──────────────────────────────────────────────────────────
def voxel_downsample(xyz, colors, voxel_size):
    t0=time.time()
    idx=(xyz/voxel_size).astype(np.int32)
    keys=idx[:,0].astype(np.int64)*1_000_000 + idx[:,1].astype(np.int64)*1000 + idx[:,2].astype(np.int64)
    order=np.argsort(keys)
    keys_sorted=keys[order]
    _,first=np.unique(keys_sorted,return_index=True)
    sel=order[first]
    print(f"  Voxel {voxel_size}m: {len(xyz):,} -> {len(sel):,} pts  [{time.time()-t0:.1f}s]")
    return xyz[sel], colors[sel]

# ── Transfer colors from original cloud to mesh vertices (nearest neighbour) ──
def transfer_colors(mesh_pts, src_pts, src_colors):
    t0=time.time()
    tree=KDTree(src_pts)
    _,idx=tree.query(mesh_pts, k=1, workers=-1)
    print(f"  Color transfer: {len(mesh_pts):,} verts  [{time.time()-t0:.1f}s]")
    return src_colors[idx]

# ── Export USD mesh ───────────────────────────────────────────────────────────
def export_usdz(mesh, vert_colors, out_usdc, out_usdz):
    t0=time.time()
    stage=Usd.Stage.CreateNew(out_usdc)
    stage.SetMetadata('upAxis','Y')
    UsdGeom.SetStageMetersPerUnit(stage,1.0)

    m=UsdGeom.Mesh.Define(stage,'/world/vathylakos_mesh')

    # Vertices
    pts=mesh.points.astype(np.float32)
    m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))

    # Faces (pyvista stores as flat array: [3, i0,i1,i2, 3, i0,i1,i2 ...])
    faces_flat=mesh.faces
    n_faces=mesh.n_cells
    face_idx=faces_flat.reshape(-1,4)[:,1:]   # strip the leading 3
    vc=Vt.IntArray([3]*n_faces)
    fi=Vt.IntArray(face_idx.flatten().tolist())
    m.GetFaceVertexCountsAttr().Set(vc)
    m.GetFaceVertexIndicesAttr().Set(fi)

    # Vertex colors as primvar
    colors_f=vert_colors.astype(np.float32)/255.0
    cpv=m.GetDisplayColorPrimvar()
    cpv.Set(Vt.Vec3fArray.FromNumpy(colors_f))
    cpv.SetInterpolation('vertex')

    # Normals (flat shading fallback)
    m.SetNormalsInterpolation('uniform')

    stage.GetRootLayer().Save()
    del stage
    sz=os.path.getsize(out_usdc)/1e6
    print(f"  USDC: {sz:.1f} MB  [{time.time()-t0:.1f}s]")

    with zipfile.ZipFile(out_usdz,'w',zipfile.ZIP_STORED) as zf:
        zf.write(out_usdc,os.path.basename(out_usdc))
    try: os.remove(out_usdc)
    except: pass
    print(f"  USDZ: {out_usdz}  ({os.path.getsize(out_usdz)/1e6:.1f} MB)")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--voxel', type=float, default=0.25,
                    help='Voxel size for downsample in metres (default 0.25)')
    ap.add_argument('--alpha', type=float, default=1.5,
                    help='Delaunay alpha — max triangle circumradius in metres (default 1.5). '
                         'Smaller = tighter mesh, larger = fills more gaps')
    ap.add_argument('--max-edge', type=float, default=3.0,
                    help='Remove triangles with any edge > this length (default 3.0m)')
    ap.add_argument('--out', type=str, default=None)
    args=ap.parse_args()

    stem=args.out or os.path.splitext(args.input)[0]+'_mesh'
    out_usdc=stem+'.usdc'
    out_usdz=stem+'.usdz'
    out_ply =stem+'.ply'

    print('='*60)
    print(f'  Input : {args.input}')
    print(f'  Voxel : {args.voxel}m  Alpha : {args.alpha}m  MaxEdge : {args.max_edge}m')
    print('='*60)
    T=time.time()

    # 1. Load
    print('\n[1] Loading...')
    data=load_ply(args.input)
    n_orig=len(data)

    # Filter scale outliers if present
    if 'scalar_scale_0' in data.dtype.names:
        s=np.stack([np.exp(data[f].astype(np.float32))
                    for f in ('scalar_scale_0','scalar_scale_1','scalar_scale_2')],axis=1).max(axis=1)
        data=data[s<0.15]
        print(f'  Scale filter: {n_orig:,} -> {len(data):,}')

    # Center
    cx=float((data['x'].min()+data['x'].max())/2)
    cy=float((data['y'].min()+data['y'].max())/2)
    cz=float((data['z'].min()+data['z'].max())/2)
    xyz=np.stack([
        data['x'].astype(np.float32)-cx,
        data['y'].astype(np.float32)-cy,
        data['z'].astype(np.float32)-cz,
    ],axis=1)
    colors=np.stack([data['red'],data['green'],data['blue']],axis=1)

    print(f'  Bounds: X[{xyz[:,0].min():.1f},{xyz[:,0].max():.1f}]  '
          f'Y[{xyz[:,1].min():.1f},{xyz[:,1].max():.1f}]  '
          f'Z[{xyz[:,2].min():.1f},{xyz[:,2].max():.1f}]')

    # 2. Downsample
    print(f'\n[2] Voxel downsample ({args.voxel}m)...')
    xyz_ds, colors_ds=voxel_downsample(xyz, colors, args.voxel)

    # 3. Delaunay 2D (terrain mesh)
    print(f'\n[3] Delaunay 2D triangulation on {len(xyz_ds):,} pts...')
    t1=time.time()
    cloud=pv.PolyData(xyz_ds)
    # delaunay_2d projects onto best-fit plane — use alpha to cut large triangles
    mesh=cloud.delaunay_2d(alpha=args.alpha, progress_bar=True)
    print(f'  Raw mesh: {mesh.n_points:,} verts  {mesh.n_cells:,} faces  [{time.time()-t1:.1f}s]')

    # 4. Remove long-edge triangles (holes/cliffs shouldn't be bridged)
    print(f'\n[4] Filtering long edges (> {args.max_edge}m)...')
    t1=time.time()
    pts_m=mesh.points
    faces_m=mesh.faces.reshape(-1,4)[:,1:]
    v0=pts_m[faces_m[:,0]]; v1=pts_m[faces_m[:,1]]; v2=pts_m[faces_m[:,2]]
    e01=np.linalg.norm(v1-v0,axis=1)
    e12=np.linalg.norm(v2-v1,axis=1)
    e20=np.linalg.norm(v0-v2,axis=1)
    keep=np.where((e01<args.max_edge)&(e12<args.max_edge)&(e20<args.max_edge))[0]
    kept_faces=faces_m[keep]
    flat=np.hstack([np.full((len(kept_faces),1),3),kept_faces]).flatten()
    mesh2=pv.PolyData(pts_m, flat)
    mesh2=mesh2.clean()
    print(f'  Filtered: {mesh2.n_cells:,} faces  [{time.time()-t1:.1f}s]')

    # 5. Transfer colors from original cloud to mesh vertices
    print(f'\n[5] Transferring colors...')
    vert_colors=transfer_colors(mesh2.points, xyz, colors)

    # 6. Save cleaned PLY (optional, for inspection in CloudCompare)
    print(f'\n[6] Saving mesh PLY...')
    mesh2.save(out_ply)
    print(f'  PLY: {out_ply}  ({os.path.getsize(out_ply)/1e6:.1f} MB)')

    # 7. Export USDZ
    print(f'\n[7] Exporting USDZ...')
    export_usdz(mesh2, vert_colors, out_usdc, out_usdz)

    print(f'\n{"="*60}')
    print(f'  Verts  : {mesh2.n_points:,}')
    print(f'  Faces  : {mesh2.n_cells:,}')
    print(f'  PLY    : {out_ply}')
    print(f'  USDZ   : {out_usdz}')
    print(f'  Total  : {time.time()-T:.0f}s')

if __name__=='__main__':
    main()
