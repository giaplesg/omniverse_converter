"""
Gaussian Splat PLY → PLY with RGB
Converts a GS PLY (with f_dc_0/1/2 SH fields) to include
standard red/green/blue uchar properties so it opens in
CloudCompare, MeshLab, Potree, SuperSplat, etc.

Usage:
    python gs_add_rgb.py input.ply [output.ply]

If output.ply is not specified, saves as input_rgb.ply
"""

import sys, os, struct, re

SH_C0 = 0.28209479177387814  # DC spherical harmonics constant

def parse_ply_header(f):
    lines = []
    while True:
        line = f.readline().decode("ascii", errors="ignore").rstrip()
        lines.append(line)
        if line == "end_header":
            break
    return lines

def parse_properties(header_lines):
    props = []
    for line in header_lines:
        m = re.match(r"property\s+(\w+)\s+(\w+)", line)
        if m:
            props.append((m.group(1), m.group(2)))  # (type, name)
    return props

TYPE_SIZE = {
    "float": 4, "double": 8,
    "int": 4, "uint": 4,
    "short": 2, "ushort": 2,
    "char": 1, "uchar": 1,
}
TYPE_FMT = {
    "float": "f", "double": "d",
    "int": "i", "uint": "I",
    "short": "h", "ushort": "H",
    "char": "b", "uchar": "B",
}

def sh_dc_to_rgb_u8(v):
    """Convert a single SH DC coefficient to 0-255 uint8."""
    rgb = v * SH_C0 + 0.5
    return max(0, min(255, round(rgb * 255)))

def convert(src_path, dst_path):
    with open(src_path, "rb") as f:
        header_lines = parse_ply_header(f)
        props = parse_properties(header_lines)

        prop_names = [p[1] for p in props]

        # Verify required fields exist
        for req in ("f_dc_0", "f_dc_1", "f_dc_2"):
            if req not in prop_names:
                print(f"ERROR: Field '{req}' not found in PLY. Is this a Gaussian Splat file?")
                print("Fields found:", prop_names)
                sys.exit(1)

        if "red" in prop_names:
            print("NOTE: PLY already has 'red' field — will re-compute and overwrite.")

        # Get vertex count
        n_verts = 0
        for line in header_lines:
            m = re.match(r"element vertex (\d+)", line)
            if m:
                n_verts = int(m.group(1))
        print(f"  Vertices : {n_verts:,}")

        # Compute stride
        stride = sum(TYPE_SIZE[t] for t, _ in props)
        fmt_str = "<" + "".join(TYPE_FMT[t] for t, _ in props)
        print(f"  Stride   : {stride} bytes/vertex")
        print(f"  Fields   : {prop_names}")

        # Read all vertex data
        data = f.read(n_verts * stride)
        if len(data) != n_verts * stride:
            print(f"WARNING: Expected {n_verts * stride} bytes, got {len(data)}")

    # Find indices of f_dc fields
    idx = {name: i for i, (_, name) in enumerate(props)}
    i_r = idx["f_dc_0"]
    i_g = idx["f_dc_1"]
    i_b = idx["f_dc_2"]

    # Build new header: add rgb after z (or after existing z/nz/blue)
    new_props = []
    rgb_inserted = False
    has_xyz = all(k in prop_names for k in ("x","y","z"))
    has_normals = all(k in prop_names for k in ("nx","ny","nz"))

    for t, name in props:
        if name in ("red","green","blue"):
            continue  # remove old ones if present
        new_props.append((t, name))
        # Insert rgb after z (no normals) or after nz (with normals)
        if not rgb_inserted:
            if has_normals and name == "nz":
                new_props += [("uchar","red"),("uchar","green"),("uchar","blue")]
                rgb_inserted = True
            elif not has_normals and name == "z":
                new_props += [("uchar","red"),("uchar","green"),("uchar","blue")]
                rgb_inserted = True

    if not rgb_inserted:
        new_props += [("uchar","red"),("uchar","green"),("uchar","blue")]

    print(f"  Output fields: {[n for _,n in new_props]}")

    # Build new header string
    new_header_lines = []
    for line in header_lines:
        if re.match(r"property\s+\w+\s+(red|green|blue)$", line):
            continue  # skip old rgb
        if line == "end_header":
            break
        if re.match(r"property", line):
            continue  # will re-add from new_props
        if re.match(r"element vertex", line):
            new_header_lines.append(f"element vertex {n_verts}")
            for t, n in new_props:
                new_header_lines.append(f"property {t} {n}")
            continue
        new_header_lines.append(line)
    new_header_lines.append("end_header")
    new_header = "\n".join(new_header_lines) + "\n"

    # Process vertices
    new_fmt_str = "<" + "".join(TYPE_FMT[t] for t, _ in new_props)
    records = struct.iter_unpack(fmt_str, data)

    out_rows = []
    for row in records:
        r = sh_dc_to_rgb_u8(row[i_r])
        g = sh_dc_to_rgb_u8(row[i_g])
        b = sh_dc_to_rgb_u8(row[i_b])

        # Build new row from new_props
        new_row = []
        for t, name in new_props:
            if name == "red":   new_row.append(r)
            elif name == "green": new_row.append(g)
            elif name == "blue":  new_row.append(b)
            else:
                orig_idx = idx[name]
                new_row.append(row[orig_idx])
        out_rows.append(struct.pack(new_fmt_str, *new_row))

    with open(dst_path, "wb") as out:
        out.write(new_header.encode("ascii"))
        for row_bytes in out_rows:
            out.write(row_bytes)

    size_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f"\n  Saved: {dst_path}")
    print(f"  Size : {size_mb:.1f} MB")
    print("  Done — open in CloudCompare, MeshLab, or SuperSplat")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    src = sys.argv[1]
    if len(sys.argv) >= 3:
        dst = sys.argv[2]
    else:
        base, ext = os.path.splitext(src)
        dst = base + "_rgb" + ext

    print(f"Input  : {src}")
    print(f"Output : {dst}")
    convert(src, dst)
