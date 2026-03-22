"""
Gaussian Splat PLY → Clean PLY + USDZ for Omniverse
Removes noise via:
  1. Opacity threshold  — drops transparent/ghost splats
  2. Scale filter       — drops exploded / microscopic splats
  3. SOR                — Statistical Outlier Removal (scipy KDTree)

Then exports to USDZ (colored point cloud) ready for Omniverse.

Usage:
    python gs_clean_to_usdz.py input.ply [options]

Options (all optional):
    --opacity    0.05     min opacity after sigmoid  (0–1, default 0.05)
    --max-scale  2.0      max Gaussian radius in scene units (default 2.0)
    --min-scale  0.0001   min Gaussian radius (default 0.0001)
    --sor-k      20       SOR: number of neighbours (default 20)
    --sor-std    2.0      SOR: std-dev multiplier   (default 2.0)
    --voxel      0.0      voxel down-sample size, 0 = off (default 0)
    --no-sor              skip SOR step
    --out        <path>   output stem (default: input_clean)
"""

import sys, os, re, struct, math, argparse, time
import numpy as np
from scipy.spatial import KDTree

# ── SH constant ───────────────────────────────────────────────────────────────
SH_C0 = 0.28209479177387814

# ── PLY helpers ───────────────────────────────────────────────────────────────
TYPE_SIZE = {"float":4,"double":8,"int":4,"uint":4,"short":2,"ushort":2,"char":1,"uchar":1}
TYPE_FMT  = {"float":"f","double":"d","int":"i","uint":"I","short":"h","ushort":"H","char":"b","uchar":"B"}
NP_DTYPE  = {"float":np.float32,"double":np.float64,"int":np.int32,"uint":np.uint32,
             "short":np.int16,"ushort":np.uint16,"char":np.int8,"uchar":np.uint8}

def read_ply(path):
    with open(path,"rb") as f:
        lines=[]
        while True:
            line=f.readline().decode("ascii","ignore").rstrip()
            lines.append(line)
            if line=="end_header": break
        props=[]
        n_verts=0
        for l in lines:
            m=re.match(r"element vertex (\d+)",l)
            if m: n_verts=int(m.group(1))
            m=re.match(r"property (\w+) (\w+)",l)
            if m: props.append((m.group(1),m.group(2)))
        stride=sum(TYPE_SIZE[t] for t,_ in props)
        fmt="<"+"".join(TYPE_FMT[t] for t,_ in props)
        raw=f.read()

    print(f"  Read   : {n_verts:,} vertices  stride={stride}B")
    # Parse into structured numpy array
    dtype=np.dtype([(name, NP_DTYPE[typ]) for typ,name in props])
    data=np.frombuffer(raw[:n_verts*stride], dtype=dtype)
    return data, lines, props

def write_ply(path, data, orig_header_lines, props):
    new_header=[]
    prop_done=False
    for l in orig_header_lines:
        if re.match(r"element vertex",l):
            new_header.append(f"element vertex {len(data)}")
            prop_done=False
            continue
        if re.match(r"property",l):
            if not prop_done:
                # Inject: xyz + rgb + rest of props
                for t,n in props:
                    new_header.append(f"property {t} {n}")
                prop_done=True
            continue
        if l=="end_header":
            new_header.append("end_header")
            break
        new_header.append(l)
    header_bytes=("\n".join(new_header)+"\n").encode("ascii")
    with open(path,"wb") as f:
        f.write(header_bytes)
        f.write(data.tobytes())
    print(f"  Wrote PLY : {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

# ── Cleaning steps ─────────────────────────────────────────────────────────────
def sigmoid(x): return 1.0/(1.0+np.exp(-x.astype(np.float32)))

def filter_opacity(data, min_opacity):
    t0=time.time()
    if "raw_opacities" in data.dtype.names:
        field="raw_opacities"
    elif "opacity" in data.dtype.names:
        field="opacity"
    else:
        print("  [SKIP] No opacity field found")
        return data
    opc=sigmoid(data[field])
    mask=opc>=min_opacity
    before=len(data); data=data[mask]
    print(f"  Opacity >{min_opacity}: kept {len(data):,}/{before:,}  (-{before-len(data):,})  [{time.time()-t0:.1f}s]")
    return data

def filter_scale(data, min_s, max_s):
    t0=time.time()
    fields=[n for n in ("scale_0","scale_1","scale_2") if n in data.dtype.names]
    if not fields:
        print("  [SKIP] No scale fields found")
        return data
    scales=np.stack([np.exp(data[f].astype(np.float32)) for f in fields], axis=1)
    max_per=scales.max(axis=1)
    mask=(max_per>=min_s)&(max_per<=max_s)
    before=len(data); data=data[mask]
    print(f"  Scale [{min_s},{max_s}]: kept {len(data):,}/{before:,}  (-{before-len(data):,})  [{time.time()-t0:.1f}s]")
    return data

def filter_sor(data, k=20, std_mult=2.0):
    t0=time.time()
    pts=np.stack([data["x"],data["y"],data["z"]],axis=1).astype(np.float32)
    print(f"  SOR: building KDTree for {len(pts):,} points...")
    tree=KDTree(pts)
    dists,_=tree.query(pts, k=k+1, workers=-1)
    mean_dists=dists[:,1:].mean(axis=1)  # exclude self
    mu=mean_dists.mean(); sigma=mean_dists.std()
    threshold=mu+std_mult*sigma
    mask=mean_dists<=threshold
    before=len(data); data=data[mask]
    print(f"  SOR k={k} std={std_mult}: kept {len(data):,}/{before:,}  (-{before-len(data):,})  [{time.time()-t0:.1f}s]")
    return data

def filter_voxel(data, voxel_size):
    if voxel_size<=0: return data
    t0=time.time()
    pts=np.stack([data["x"],data["y"],data["z"]],axis=1).astype(np.float32)
    voxel_idx=(pts/voxel_size).astype(np.int32)
    _,uniq=np.unique(voxel_idx,axis=0,return_index=True)
    before=len(data); data=data[uniq]
    print(f"  Voxel {voxel_size}: kept {len(data):,}/{before:,}  (-{before-len(data):,})  [{time.time()-t0:.1f}s]")
    return data

# ── USDZ export ───────────────────────────────────────────────────────────────
def export_usdz(data, out_path):
    try:
        from pxr import Usd, UsdGeom, Vt, Gf, Sdf
    except ImportError:
        print("  ERROR: pxr not found. Run: pip install usd-core")
        return

    t0=time.time()
    usdc_path=out_path.replace(".usdz",".usdc")

    stage=Usd.Stage.CreateNew(usdc_path)
    stage.SetMetadata("upAxis","Y")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    pts_prim=UsdGeom.Points.Define(stage,"/world/gaussian_splat")

    # Positions
    xyz=list(zip(data["x"].astype(float),
                 data["y"].astype(float),
                 data["z"].astype(float)))
    pts_prim.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in xyz]))

    # Colors from f_dc (SH → RGB)
    if "f_dc_0" in data.dtype.names:
        r=(data["f_dc_0"].astype(np.float32)*SH_C0+0.5).clip(0,1)
        g=(data["f_dc_1"].astype(np.float32)*SH_C0+0.5).clip(0,1)
        b=(data["f_dc_2"].astype(np.float32)*SH_C0+0.5).clip(0,1)
    elif "red" in data.dtype.names:
        r=data["red"].astype(np.float32)/255.0
        g=data["green"].astype(np.float32)/255.0
        b=data["blue"].astype(np.float32)/255.0
    else:
        r=g=b=np.ones(len(data),dtype=np.float32)*0.5

    colors=Vt.Vec3fArray([Gf.Vec3f(float(r[i]),float(g[i]),float(b[i])) for i in range(len(data))])
    pts_prim.GetDisplayColorAttr().Set(colors)

    # Width (visual size) from scale
    if "scale_0" in data.dtype.names:
        scales=np.stack([np.exp(data[f].astype(np.float32))
                        for f in ("scale_0","scale_1","scale_2")],axis=1)
        widths=scales.max(axis=1)*2.0
    else:
        widths=np.full(len(data),0.01,dtype=np.float32)
    pts_prim.GetWidthsAttr().Set(Vt.FloatArray(widths.tolist()))

    # Opacity
    if "opacity" in data.dtype.names or "raw_opacities" in data.dtype.names:
        field="raw_opacities" if "raw_opacities" in data.dtype.names else "opacity"
        opc=sigmoid(data[field])
        pts_prim.GetDisplayOpacityAttr().Set(Vt.FloatArray(opc.tolist()))

    stage.GetRootLayer().Save()
    del stage  # release pxr handle before touching the file
    print(f"  USDC : {usdc_path}  ({os.path.getsize(usdc_path)/1e6:.1f} MB)  [{time.time()-t0:.1f}s]")

    # Pack into USDZ
    t1=time.time()
    import zipfile
    with zipfile.ZipFile(out_path,"w",zipfile.ZIP_STORED) as zf:
        zf.write(usdc_path, os.path.basename(usdc_path))
    try:
        os.remove(usdc_path)
    except Exception:
        pass  # Windows may still hold the handle briefly
    print(f"  USDZ : {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)  [{time.time()-t1:.1f}s]")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser(description="Clean GS PLY and export to USDZ")
    ap.add_argument("input")
    ap.add_argument("--opacity",   type=float, default=0.05)
    ap.add_argument("--max-scale", type=float, default=2.0,   dest="max_scale")
    ap.add_argument("--min-scale", type=float, default=0.0001,dest="min_scale")
    ap.add_argument("--sor-k",     type=int,   default=20,    dest="sor_k")
    ap.add_argument("--sor-std",   type=float, default=2.0,   dest="sor_std")
    ap.add_argument("--voxel",     type=float, default=0.0)
    ap.add_argument("--no-sor",    action="store_true",       dest="no_sor")
    ap.add_argument("--out",       type=str,   default=None)
    args=ap.parse_args()

    src=args.input
    stem=args.out or os.path.splitext(src)[0]+"_clean"
    out_ply=stem+".ply"
    out_usdz=stem+".usdz"

    print("="*60)
    print(f"  Input  : {src}")
    print(f"  Output : {out_ply}")
    print(f"           {out_usdz}")
    print("="*60)

    t_start=time.time()

    # 1. Load
    data,header,props=read_ply(src)

    # 2. Clean
    print("\n--- Cleaning ---")
    data=filter_opacity(data, args.opacity)
    data=filter_scale(data, args.min_scale, args.max_scale)
    if args.voxel>0:
        data=filter_voxel(data, args.voxel)
    if not args.no_sor:
        data=filter_sor(data, k=args.sor_k, std_mult=args.sor_std)

    pct=100*len(data)/sum(1 for _ in range(1))  # just len
    print(f"\n  Final  : {len(data):,} splats  (total time so far: {time.time()-t_start:.0f}s)")

    # 3. Save cleaned PLY
    print("\n--- Saving PLY ---")
    write_ply(out_ply, data, header, props)

    # 4. Export USDZ
    print("\n--- Exporting USDZ ---")
    export_usdz(data, out_usdz)

    print(f"\n  Done in {time.time()-t_start:.0f}s")
    print(f"  PLY  -> {out_ply}")
    print(f"  USDZ -> {out_usdz}  (drag into Omniverse)")

if __name__=="__main__":
    main()
