"""
GS Point Cloud -> Mesh (v3) - Screened Poisson Surface Reconstruction
Uses pymeshlab (MeshLab algorithms) for high-quality mesh from GS point clouds.
"""
import numpy as np, re, os, zipfile, time, argparse
import pymeshlab
from scipy.spatial import KDTree
from pxr import Usd, UsdGeom, Vt, Gf

SH_C0 = 0.28209479177387814

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
    dtype=np.dtype([(name,TYPE[typ][2]) for typ,name in props if typ in TYPE])
    return np.frombuffer(raw[:n*dtype.itemsize], dtype=dtype)

def voxel_ds(xyz, colors, vsize):
    idx=(xyz/vsize).astype(np.int32)
    keys=idx[:,0].astype(np.int64)*1_000_000+idx[:,1].astype(np.int64)*1000+idx[:,2].astype(np.int64)
    order=np.argsort(keys)
    _,uidx=np.unique(keys[order],return_index=True)
    sel=order[uidx]
    return xyz[sel], colors[sel]

def sor(xyz, k=20, std_mult=2.0):
    tree=KDTree(xyz)
    d,_=tree.query(xyz, k=k+1, workers=-1)
    md=d[:,1:].mean(axis=1)
    return xyz[md <= md.mean()+std_mult*md.std()]

def transfer_colors(mesh_pts, src_pts, src_colors):
    tree=KDTree(src_pts)
    _,idx=tree.query(mesh_pts, k=1, workers=-1)
    return src_colors[idx]

def export_usdz(verts, faces, vert_colors, out_usdc, out_usdz):
    stage=Usd.Stage.CreateNew(out_usdc)
    stage.SetMetadata('upAxis','Y')
    UsdGeom.SetStageMetersPerUnit(stage,1.0)
    m=UsdGeom.Mesh.Define(stage,'/world/vathylakos')
    pts=verts.astype(np.float32)
    m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]*len(faces)))
    m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))
    cpv=m.GetDisplayColorPrimvar()
    cpv.Set(Vt.Vec3fArray.FromNumpy(vert_colors.astype(np.float32)/255.0))
    cpv.SetInterpolation('vertex')
    stage.GetRootLayer().Save()
    del stage
    with zipfile.ZipFile(out_usdz,'w',zipfile.ZIP_STORED) as zf:
        zf.write(out_usdc,os.path.basename(out_usdc))
    try: os.remove(out_usdc)
    except: pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--voxel',   type=float, default=0.2,
                    help='Voxel downsample size in metres (default 0.2)')
    ap.add_argument('--sor-std', type=float, default=1.5, dest='sor_std')
    ap.add_argument('--depth',   type=int,   default=10,
                    help='Poisson octree depth — higher = more detail (default 10)')
    ap.add_argument('--min-samples', type=float, default=1.5, dest='min_samples',
                    help='Poisson min_samples parameter (default 1.5)')
    ap.add_argument('--out', default=None)
    args=ap.parse_args()

    stem=args.out or os.path.splitext(args.input)[0]+'_mesh3'
    T=time.time()

    print('='*60)
    print(f'  Input  : {args.input}')
    print('='*60)

    # 1. Load + pre-filter
    print('\n[1] Loading...')
    data=load_ply(args.input)
    n0=len(data)

    # Filter blob-scale outliers if GS scale fields present
    if 'scalar_scale_0' in data.dtype.names:
        s=np.stack([np.exp(data[f].astype(np.float32))
                    for f in ('scalar_scale_0','scalar_scale_1','scalar_scale_2')],axis=1).max(axis=1)
        data=data[s<0.15]

    cx=float((data['x'].min()+data['x'].max())/2)
    cy=float((data['y'].min()+data['y'].max())/2)
    cz=float((data['z'].min()+data['z'].max())/2)

    xyz_full=np.stack([data['x'].astype(np.float32)-cx,
                       data['y'].astype(np.float32)-cy,
                       data['z'].astype(np.float32)-cz],axis=1)

    if 'red' in data.dtype.names:
        col_full=np.stack([data['red'],data['green'],data['blue']],axis=1)
    else:
        r=(data['f_dc_0'].astype(np.float32)*SH_C0+0.5).clip(0,1)
        g=(data['f_dc_1'].astype(np.float32)*SH_C0+0.5).clip(0,1)
        b=(data['f_dc_2'].astype(np.float32)*SH_C0+0.5).clip(0,1)
        col_full=(np.stack([r,g,b],axis=1)*255).astype(np.uint8)

    print(f'  Loaded: {len(data):,}  center:({cx:.1f},{cy:.1f},{cz:.1f})')
    print(f'  Bounds: X[{xyz_full[:,0].min():.1f},{xyz_full[:,0].max():.1f}]  '
          f'Y[{xyz_full[:,1].min():.1f},{xyz_full[:,1].max():.1f}]  '
          f'Z[{xyz_full[:,2].min():.1f},{xyz_full[:,2].max():.1f}]')

    # 2. Voxel downsample
    print(f'\n[2] Voxel downsample ({args.voxel}m)...')
    t=time.time()
    xyz,colors=voxel_ds(xyz_full, col_full, args.voxel)
    print(f'  {n0:,} -> {len(xyz):,} pts  [{time.time()-t:.1f}s]')

    # 3. SOR
    print(f'\n[3] Statistical Outlier Removal (std={args.sor_std})...')
    t=time.time()
    before=len(xyz)
    xyz_clean=sor(xyz, k=20, std_mult=args.sor_std)
    print(f'  {before:,} -> {len(xyz_clean):,}  (-{before-len(xyz_clean):,})  [{time.time()-t:.1f}s]')

    # 4. Poisson reconstruction via pymeshlab
    print(f'\n[4] Screened Poisson Surface Reconstruction (depth={args.depth})...')
    t=time.time()

    ms=pymeshlab.MeshSet()
    # Create point cloud mesh
    v=pymeshlab.Mesh(vertex_matrix=xyz_clean.astype(np.float64))
    ms.add_mesh(v, 'pointcloud')

    # Estimate normals
    print('  Computing normals...')
    ms.compute_normal_for_point_clouds(k=20, smoothiter=2)

    # Screened Poisson
    print('  Running Poisson reconstruction...')
    ms.generate_surface_reconstruction_screened_poisson(
        depth=args.depth,
        fulldepth=5,
        cgdepth=0,
        scale=1.1,
        samplespernode=args.min_samples,
        pointweight=4.0,
        iters=8,
        confidence=False,
        preclean=True,
    )
    ms.set_current_mesh(1)  # the reconstructed surface
    cur=ms.current_mesh()
    verts=cur.vertex_matrix().astype(np.float32)
    faces=cur.face_matrix()
    print(f'  Raw: {len(verts):,} verts  {len(faces):,} faces  [{time.time()-t:.1f}s]')

    # 5. Remove interior artifacts (keep largest component)
    print('\n[5] Keeping largest component...')
    t=time.time()
    ms.compute_selection_by_small_disconnected_components_per_face(
        nbfaceratio=0.1
    )
    ms.meshing_remove_selected_faces()
    ms.meshing_remove_unreferenced_vertices()
    cur=ms.current_mesh()
    verts=cur.vertex_matrix().astype(np.float32)
    faces=cur.face_matrix()
    print(f'  Clean: {len(verts):,} verts  {len(faces):,} faces  [{time.time()-t:.1f}s]')

    # 6. Transfer colors from full cloud
    print('\n[6] Transferring colors from original cloud...')
    t=time.time()
    vc=transfer_colors(verts, xyz_full, col_full)
    print(f'  Done  [{time.time()-t:.1f}s]')

    # 7. Save
    print('\n[7] Saving...')
    ply_path=stem+'.ply'
    ms.save_current_mesh(ply_path)
    print(f'  PLY : {ply_path}  ({os.path.getsize(ply_path)/1e6:.1f} MB)')

    export_usdz(verts, faces, vc, stem+'.usdc', stem+'.usdz')
    print(f'  USDZ: {stem}.usdz  ({os.path.getsize(stem+".usdz")/1e6:.1f} MB)')

    print(f'\n{"="*60}')
    print(f'  Verts : {len(verts):,}')
    print(f'  Faces : {len(faces):,}')
    print(f'  Total : {time.time()-T:.0f}s')

if __name__=='__main__':
    main()
