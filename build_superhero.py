"""Fortnite-style one-colour superhero (Boundless full-mask look, no eyes).

Run:  blender --background --python build_superhero.py
Needs the MPFB (MakeHuman) Blender extension, which supplies the anatomical base body and the
UE-style game_engine skeleton.

Measurements follow Epic's UEFN docs: "Player characters in Fortnite are 192 cm tall."
Units are metres, Z is up, the model faces -Y, and it stands in the A-pose rest pose.
Outputs: superhero.glb/.fbx/.blend, textures/hex_normal.png, renders/*.png
"""
import importlib
import math
import os

import addon_utils
import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

OUT = os.path.dirname(os.path.abspath(__file__))
HEIGHT = 1.92                              # Fortnite player height (Epic UEFN docs)
SUIT_COLOUR = (0.015, 0.06, 0.70, 1.0)     # the one colour (linear RGB), Hypersonic-style royal blue
HEX_SIZE = 0.012                           # spandex hex cell width in metres

# Stylisation on top of the MakeHuman base: Fortnite's hero build has broad shoulders and lats,
# a narrow waist, chunky gloved hands and big boots. The mask presses the ears flat and closes the eyes.
TARGETS = {
    "head-scale-vert-incr": 0.85, "head-scale-horiz-incr": 0.7, "head-scale-depth-incr": 0.6,
    "head-square": 0.4,
    "measure-neck-height-decr": 0.7,
    "measure-neck-circ-incr": 0.8,
    "torso-vshape-incr": 0.8,
    "torso-muscle-dorsi-incr": 0.8,
    "torso-muscle-pectoral-incr": 0.7,
    "torso-scale-horiz-incr": 0.3,
    "measure-shoulder-dist-incr": 0.9,
    "measure-waist-circ-decr": 0.4,
    "measure-upperarm-circ-incr": 0.8,
    "l-upperarm-muscle-incr": 0.6, "r-upperarm-muscle-incr": 0.6,
    "l-lowerarm-muscle-incr": 0.5, "r-lowerarm-muscle-incr": 0.5,
    "measure-thigh-circ-incr": 0.7,
    "l-upperleg-muscle-incr": 0.5, "r-upperleg-muscle-incr": 0.5,
    "measure-calf-circ-incr": 0.5,
    "l-upperarm-shoulder-muscle-incr": 0.8, "r-upperarm-shoulder-muscle-incr": 0.8,
    "l-hand-scale-incr": 0.2, "r-hand-scale-incr": 0.2,
    "l-hand-fingers-length-decr": 0.3, "r-hand-fingers-length-decr": 0.3,
    "l-hand-fingers-diameter-incr": 0.8, "r-hand-fingers-diameter-incr": 0.8,
    "l-foot-scale-incr": 0.2, "r-foot-scale-incr": 0.2,
    "l-ear-flap-decr": 1.0, "r-ear-flap-decr": 1.0,
    "l-ear-wing-decr": 1.0, "r-ear-wing-decr": 1.0,
    "l-ear-scale-decr": 0.5, "r-ear-scale-decr": 0.5,
    "eye-left-closure": 1.0, "eye-right-closure": 1.0,
}

MOD = next(m.__name__ for m in addon_utils.modules() if m.__name__.split(".")[-1] == "mpfb")
addon_utils.enable(MOD, default_set=True)
HS = importlib.import_module(MOD + ".services.humanservice").HumanService
TS = importlib.import_module(MOD + ".services.targetservice").TargetService

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
scene = bpy.context.scene
FINGERS = [f"{f}_0{i}" for f in ("index", "middle", "ring", "pinky", "thumb") for i in (1, 2, 3)]


# ----------------------------------------------------------------------------- helpers
def get_co(me):
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def set_co(me, co):
    me.vertices.foreach_set("co", co.ravel())
    me.update()


def weights(obj):
    """{group name: per-vertex weight array}"""
    nv = len(obj.data.vertices)
    w = np.zeros((nv, len(obj.vertex_groups)))
    for v in obj.data.vertices:
        for g in v.groups:
            w[v.index, g.group] = g.weight
    return {g.name: w[:, g.index] for g in obj.vertex_groups}


def wsum(W, names):
    return sum((W[n] for n in names if n in W), np.zeros(len(next(iter(W.values())))))


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def activate(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def apply_modifiers(obj):
    activate(obj)
    for m in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def subset_bmesh(obj, vmask):
    """bmesh of the faces whose vertices are all inside vmask."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    dead = [f for f in bm.faces if not all(vmask[v.index] for v in f.verts)]
    bmesh.ops.delete(bm, geom=dead, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    return bm


# ----------------------------------------------------------------------------- 1. base body
macro = TS.get_default_macro_info_dict()
macro.update(gender=1.0, age=0.5, muscle=1.0, weight=0.5, proportions=1.0, height=0.8)
body = HS.create_human(macro_detail_dict=macro)
body.name = "Superhero"
for name, value in TARGETS.items():
    path = TS.target_full_path(name)
    assert path, f"missing MakeHuman target {name}"
    TS.load_target(body, path, weight=value)
TS.bake_targets(body)

# scale so the body (helpers excluded) is exactly HEIGHT, feet on z=0
co = get_co(body.data)
W0 = weights(body)
in_body = W0["body"] > 0.5
zmin, zmax = co[in_body, 2].min(), co[in_body, 2].max()
co[:, 2] -= zmin
co *= HEIGHT / (zmax - zmin)
set_co(body.data, co)

eye_c = [Vector(co[W0[g] > 0.5].mean(axis=0)) for g in ("helper-l-eye", "helper-r-eye")]
teeth = co[(W0["helper-upper-teeth"] > 0.5) | (W0["helper-lower-teeth"] > 0.5)]
mouth_c = Vector((0.0, teeth[:, 1].min(), teeth[:, 2].mean()))


rig = HS.add_builtin_rig(body, "game_engine")
rig.name = rig.data.name = "Skeleton"
B = {b.name: b for b in rig.data.bones}
for m in list(body.modifiers):
    if m.type == "ARMATURE":
        body.modifiers.remove(m)
apply_modifiers(body)            # the MASK modifier: removes eyeball/teeth/tongue/hair helpers

# ----------------------------------------------------------------------------- 2. full-face mask + boot toe box
# The mask fabric stretches across the eye sockets and mouth like a membrane (Laplace fill with the
# surrounding face held fixed), the rest of the head is softened, and the ears flatten.
# Boots: toes are pressed into one rounded toe box (projection onto an ellipsoid).
me = body.data
co = get_co(me)
W = weights(body)
head_w = np.clip(W["head"], 0, 1)
edges = np.empty(len(me.edges) * 2, dtype=np.int64)
me.edges.foreach_get("vertices", edges)
e = edges.reshape(-1, 2)
deg = np.bincount(e.ravel(), minlength=len(co)).astype(float)


def laplacian(c):
    acc = np.empty_like(c)
    for k in range(3):
        acc[:, k] = (np.bincount(e[:, 0], weights=c[e[:, 1], k], minlength=len(c))
                     + np.bincount(e[:, 1], weights=c[e[:, 0], k], minlength=len(c)))
    return acc / deg[:, None] - c


def taubin(c, w, iters):
    for _ in range(iters):
        c += laplacian(c) * (0.5 * w)[:, None]
        c += laplacian(c) * (-0.53 * w)[:, None]


def membrane(c, free, iters=2500):
    for _ in range(iters):
        c[free] += laplacian(c)[free]


eye_d = np.minimum(*[np.linalg.norm(co - np.array(c), axis=1) for c in eye_c])
mouth_d = np.hypot(co[:, 0] / 0.03, (co[:, 2] - mouth_c.z) / 0.016)
mouth_front = co[:, 1] < mouth_c.y + 0.012
# Cavities (mouth bag, deep lip folds) = face vertices more than 5 mm inside a coarse voxel copy of
# the body; at 1.5 cm voxels the mouth opening is sealed shut, so the copy has no cavity.
proxy = body.copy()
proxy.data = body.data.copy()
scene.collection.objects.link(proxy)
prem = proxy.modifiers.new("remesh", "REMESH")
prem.mode = "VOXEL"
prem.voxel_size = 0.02
apply_modifiers(proxy)
proxy_bvh = BVHTree.FromObject(proxy, bpy.context.evaluated_depsgraph_get())
bpy.data.objects.remove(proxy)
depth = np.zeros(len(co))
near_face = (head_w > 0.3) & (np.linalg.norm(co - np.array(mouth_c), axis=1) < 0.12)
for i in np.nonzero(near_face)[0]:
    loc, n, _, dist = proxy_bvh.find_nearest(Vector(co[i]))
    if (Vector(co[i]) - loc).dot(n) < 0:
        depth[i] = dist
mouth_cavity = near_face & ((depth > 0.003) | ((mouth_d < 0.7) & mouth_front))
membrane(co, (eye_d < 0.03) | mouth_cavity)
print(f"[superhero] mouth membrane verts={int(mouth_cavity.sum())}")

w = 0.3 * head_w
w = np.maximum(w, 1.0 - smoothstep(0.03, 0.05, eye_d))
w = np.maximum(w, 0.8 * (1.0 - smoothstep(1.0, 1.8, mouth_d)) * mouth_front)
face = (head_w > 0.9) & (co[:, 2] < eye_c[0].z) & (co[:, 2] > eye_c[0].z - 0.07)
nose_tip = co[face][np.argmin(co[face][:, 1])]
w = np.maximum(w, 0.45 * (1.0 - smoothstep(0.012, 0.03, np.linalg.norm(co - nose_tip, axis=1))))
head_x = np.abs(co[head_w > 0.9, 0]).max()
w = np.maximum(w, 0.7 * head_w * smoothstep(0.75 * head_x, 0.88 * head_x, np.abs(co[:, 0])))
w = np.maximum(w, 0.3 * np.clip(W["foot_l"] + W["foot_r"], 0, 1))
taubin(co, w, 60)
set_co(me, co)
for p in me.polygons:
    p.use_smooth = True

# ----------------------------------------------------------------------------- 3. materials
def hex_normal_image():
    """Tileable pointy-top hex spandex normal map (OpenGL convention)."""
    n_cols, n_rows2 = 16, 16
    s = 64.0                                    # px between hex centres horizontally
    r = s / math.sqrt(3)
    wpx, hpx = int(n_cols * s), int(round(n_rows2 * 3 * r))
    y, x = np.mgrid[0:hpx, 0:wpx].astype(float)
    q = (math.sqrt(3) / 3 * x - y / 3) / r
    rr = (2 / 3 * y) / r
    cx, cz = q, rr
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    fix_x = (dx > dy) & (dx > dz)
    fix_z = ~fix_x & (dz >= dy)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(fix_z, -rx - ry, rz)
    hx = r * math.sqrt(3) * (rx + rz / 2)
    hy = r * 1.5 * rz
    ox, oy = x - hx, y - hy
    d = np.maximum.reduce([np.abs(ox),
                           np.abs(0.5 * ox + math.sqrt(3) / 2 * oy),
                           np.abs(0.5 * ox - math.sqrt(3) / 2 * oy)]) / (s / 2)
    h = (1.0 - smoothstep(0.78, 0.98, d)) + 0.12 * (1.0 - d * d)
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    k = 3.0
    n = np.stack([-k * gx, -k * gy, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    rgba = np.concatenate([n * 0.5 + 0.5, np.ones((hpx, wpx, 1))], axis=-1)
    img = bpy.data.images.new("hex_normal", wpx, hpx, alpha=False)
    img.pixels.foreach_set(rgba.astype(np.float32).ravel())
    os.makedirs(os.path.join(OUT, "textures"), exist_ok=True)
    img.filepath_raw = os.path.join(OUT, "textures", "hex_normal.png")
    img.file_format = "PNG"
    img.save()
    img.colorspace_settings.name = "Non-Color"
    return img, wpx * HEX_SIZE / s, hpx * HEX_SIZE / s   # tile size in metres


def metres_per_uv(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv = bm.loops.layers.uv.active
    ratios = []
    for f in list(bm.faces)[::7]:
        pts = [l[uv].uv for l in f.loops]
        a_uv = 0.5 * abs(sum(pts[i].x * pts[i - 1].y - pts[i - 1].x * pts[i].y for i in range(len(pts))))
        if a_uv > 1e-12:
            ratios.append(math.sqrt(f.calc_area() / a_uv))
    bm.free()
    return float(np.median(ratios))


def make_material(name, roughness, coat, hex_img=None, uv_scale=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = SUIT_COLOUR
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = 0.2
    mat.diffuse_color = SUIT_COLOUR
    if hex_img is not None:
        uvn = nt.nodes.new("ShaderNodeTexCoord")
        mapn = nt.nodes.new("ShaderNodeMapping")
        mapn.inputs["Scale"].default_value = (*uv_scale, 1.0)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = hex_img
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.inputs["Strength"].default_value = 0.35
        nt.links.new(uvn.outputs["UV"], mapn.inputs["Vector"])
        nt.links.new(mapn.outputs["Vector"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


hex_img, tile_w, tile_h = hex_normal_image()
mpu = metres_per_uv(me)
SUIT = make_material("Suit", 0.42, 0.15, hex_img, (mpu / tile_w, mpu / tile_h))
TRIM = make_material("Trim", 0.3, 0.6)
me.materials.append(SUIT)
me.materials.append(TRIM)

# ----------------------------------------------------------------------------- 4. suit layers
co = get_co(me)
W = weights(body)
Z = Vector((0, 0, 1))
bone_names = [b.name for b in rig.data.bones]
ARM = {s: [f"upperarm_{s}", f"lowerarm_{s}", f"hand_{s}"] + [f"{f}_{s}" for f in FINGERS] for s in "lr"}
arm_w = wsum(W, ARM["l"] + ARM["r"])
shoulder_z = B["upperarm_l"].head_local.z
neck = B["neck_01"].head_local
knee_z = B["calf_l"].head_local.z
belt_z = B["pelvis"].head_local.z + 0.075
boot_top = knee_z - 0.12
parts = []


def shell(name, vmask, thickness, mat_index, keep=(), src=None):
    """Raised suit layer: faces of src (default: body) in vmask, trimmed to clean edges by planes,
    then thickened. keep: (point, normal) planes; geometry on the side the normal points to is cut."""
    src = src or body
    bm = subset_bmesh(src, vmask if vmask is not None else np.ones(len(src.data.vertices), bool))
    for pco, pno in keep:
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-5,
                               plane_co=pco, plane_no=pno, clear_outer=True)
    m = bpy.data.meshes.new(name)
    bm.to_mesh(m)
    bm.free()
    m.materials.append(SUIT)
    m.materials.append(TRIM)
    for p in m.polygons:
        p.material_index = mat_index
        p.use_smooth = True
    ob = bpy.data.objects.new(name, m)
    scene.collection.objects.link(ob)
    if src is body:
        for g in body.vertex_groups:
            ob.vertex_groups.new(name=g.name)
    sol = ob.modifiers.new("solidify", "SOLIDIFY")
    sol.thickness = thickness
    sol.offset = 1.0
    sol.use_quality_normals = True
    bev = ob.modifiers.new("bevel", "BEVEL")
    bev.width = min(0.0015, thickness * 0.35)
    bev.segments = 2
    bev.limit_method = "ANGLE"
    bev.angle_limit = math.radians(40)
    apply_modifiers(ob)
    parts.append(ob)


for s in "lr":
    wrist = B[f"hand_{s}"].head_local
    fdir = (wrist - B[f"lowerarm_{s}"].head_local).normalized()
    t = (co - np.array(wrist)) @ np.array(fdir)
    side_arm = wsum(W, ARM[s][1:])
    near_wrist = (t > -0.12) & (side_arm > 0.5)
    shell(f"glove_{s}", near_wrist, 0.003, 1, keep=[(wrist - fdir * 0.065, -fdir)])
    shell(f"glove_cuff_{s}", near_wrist, 0.008, 1,
          keep=[(wrist - fdir * 0.072, -fdir), (wrist - fdir * 0.045, fdir)])


# Boot "last": closed lower-leg volume, voxel-remeshed so the toes fuse into one toe box.
bm = subset_bmesh(body, (co[:, 2] < boot_top + 0.1) & (arm_w < 0.1))
bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-5,
                       plane_co=(0, 0, boot_top + 0.08), plane_no=Z, clear_outer=True)
bmesh.ops.holes_fill(bm, edges=[ed for ed in bm.edges if ed.is_boundary])
last_me = bpy.data.meshes.new("boot_last")
bm.to_mesh(last_me)
bm.free()
last = bpy.data.objects.new("boot_last", last_me)
scene.collection.objects.link(last)
rem = last.modifiers.new("remesh", "REMESH")
rem.mode = "VOXEL"
rem.voxel_size = 0.008
smo = last.modifiers.new("smooth", "SMOOTH")
smo.factor = 1.0
smo.iterations = 16
apply_modifiers(last)
shell("boots", None, 0.006, 1, keep=[(Vector((0, 0, boot_top)), Z)], src=last)
shell("boot_cuffs", None, 0.011, 1,
      keep=[(Vector((0, 0, boot_top)), Z), (Vector((0, 0, boot_top - 0.035)), -Z)], src=last)
shell("soles", None, 0.011, 1, keep=[(Vector((0, 0, 0.022)), Z)], src=last)
bpy.data.objects.remove(last)


shell("belt", (np.abs(co[:, 2] - belt_z) < 0.06) & (arm_w < 0.1), 0.009, 1,
      keep=[(Vector((0, 0, belt_z + 0.022)), Z), (Vector((0, 0, belt_z - 0.022)), -Z)])


def add_object_from_bmesh(name, bm, mat_index):
    m = bpy.data.meshes.new(name)
    bm.to_mesh(m)
    bm.free()
    m.materials.append(SUIT)
    m.materials.append(TRIM)
    for p in m.polygons:
        p.material_index = mat_index
        p.use_smooth = True
    ob = bpy.data.objects.new(name, m)
    scene.collection.objects.link(ob)
    parts.append(ob)
    return ob


# belt buckle: rounded plate, recessed frame, raised centre (Boundless buckle silhouette)
full_bvh = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
hit = full_bvh.ray_cast(Vector((0, -1, belt_z)), Vector((0, 1, 0)))
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=1.0)
bmesh.ops.scale(bm, vec=(0.1, 0.018, 0.062), verts=bm.verts)
front = next(f for f in bm.faces if f.normal.y < -0.9)
bmesh.ops.inset_individual(bm, faces=[front], thickness=0.008, depth=-0.004)
bmesh.ops.inset_individual(bm, faces=[front], thickness=0.013, depth=0.0065)
bmesh.ops.translate(bm, vec=hit[0] + Vector((0, -0.006, 0)), verts=bm.verts)
buckle = add_object_from_bmesh("buckle", bm, 1)
bev = buckle.modifiers.new("bevel", "BEVEL")
bev.width = 0.0018
bev.segments = 3
apply_modifiers(buckle)


# raised seam piping (Hypersonic/Mighty Volt panel lines), ray-cast onto the body
def bvh_for(vmask):
    bm = subset_bmesh(body, vmask)
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    return tree


def trace(tree, rays, facing=True):
    pts = []
    for o, d in rays:
        loc, n, _, _ = tree.ray_cast(Vector(o), Vector(d))
        if loc is not None and (not facing or n.dot(Vector(d)) < 0):
            pts.append(loc)
    if len(pts) < 5:
        return []
    arr = np.array(pts)
    for _ in range(3):                                   # remove ray jitter
        arr[1:-1] = (arr[:-2] + arr[1:-1] * 2 + arr[2:]) / 4
    out = []
    for p in arr:
        loc, n, _, _ = tree.find_nearest(Vector(p))
        out.append(loc + n * 0.0004)
    return out


def pipe(name, pts, radius=0.0022, cyclic=False):
    if len(pts) < 2:
        print("[superhero] WARNING no hits for", name)
        return
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 1
    cu.use_fill_caps = not cyclic
    sp = cu.splines.new("POLY")
    sp.use_cyclic_u = cyclic
    sp.points.add(len(pts) - 1)
    for p, P in zip(sp.points, pts):
        p.co = (*P, 1.0)
    cob = bpy.data.objects.new(name, cu)
    scene.collection.objects.link(cob)
    bpy.context.view_layer.update()
    m = bpy.data.meshes.new_from_object(cob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(cob)
    m.materials.clear()
    m.materials.append(SUIT)
    m.materials.append(TRIM)
    for p in m.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, m)
    scene.collection.objects.link(ob)
    parts.append(ob)


torso_tree = bvh_for(arm_w < 0.5)
# mask-to-suit seam ring around the neck, dipping lower at the throat
neck_tree = bvh_for((wsum(W, ["neck_01", "head"]) > 0.3) & (np.hypot(co[:, 0], co[:, 1] - neck.y) < 0.11))
rays = []
for a in np.linspace(0, 2 * math.pi, 160, endpoint=False):
    d2 = Vector((math.cos(a), math.sin(a), 0))
    z = neck.z + 0.03 + 0.4 * 0.065 * math.sin(a)
    rays.append((Vector((neck.x, neck.y, z)) + d2 * 0.3, -d2))
pipe("seam_neck", trace(neck_tree, rays), radius=0.0035, cyclic=True)
sx = 0.8 * B["upperarm_l"].head_local.x
N = 90
for s, sign in (("l", 1), ("r", -1)):
    # chest V and back V
    for label, y0, dy, z_bot in (("chest", -1.0, 1.0, shoulder_z - 0.25), ("back", 1.0, -1.0, shoulder_z - 0.22)):
        rays = []
        for i in range(N + 1):
            t = i / N
            x = sign * sx * (1 - t) ** 1.15
            z = (shoulder_z - 0.05) * (1 - t) + z_bot * t
            rays.append(((x, y0, z), (0, dy, 0)))
        pipe(f"seam_{label}_{s}", trace(torso_tree, rays))
    # torso side seam: armpit to belt
    rays = [((sign * 1.0, 0.0, z), (-sign, 0, 0))
            for z in np.linspace(shoulder_z - 0.13, belt_z + 0.03, N)]
    pipe(f"seam_side_{s}", trace(torso_tree, rays))
    # outer leg seam: belt to boot
    leg_tree = bvh_for(wsum(W, [f"thigh_{s}", f"calf_{s}"]) > 0.5)
    th, ca = B[f"thigh_{s}"].head_local, B[f"calf_{s}"].head_local
    rays = []
    for z in np.linspace(belt_z - 0.035, boot_top + 0.012, N):
        a, b = (th, ca) if z > ca.z else (ca, B[f"foot_{s}"].head_local)
        f = (a.z - z) / (a.z - b.z)
        rays.append(((sign * 1.0, a.y + (b.y - a.y) * f, z), (-sign, 0, 0)))
    pipe(f"seam_leg_{s}", trace(leg_tree, rays))
    # top-of-arm seam: shoulder to glove cuff
    arm_tree = bvh_for(wsum(W, ARM[s][:2]) > 0.5)
    sh, el, wr = (B[f"upperarm_{s}"].head_local, B[f"lowerarm_{s}"].head_local, B[f"hand_{s}"].head_local)
    path = [sh.lerp(el, t) for t in np.linspace(0.12, 1, N // 2)]
    path += [el.lerp(wr, t) for t in np.linspace(0, 1, N // 2)][1:]
    cut = wr - (wr - el).normalized() * 0.075
    path = [p for p in path if (p - el).dot(wr - el) < (cut - el).dot(wr - el)]
    rays = []
    for i, p in enumerate(path):
        d = (path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]).normalized()
        up = (Vector((0, 0, 1)) - d * d.z).normalized()
        rays.append((p + up * 0.3, -up))
    pipe(f"seam_arm_{s}", trace(arm_tree, rays))

# skin weights for the added pieces: copy from the nearest body vertex
kd = KDTree(len(co))
for i, p in enumerate(co):
    kd.insert(p, i)
kd.balance()
for ob in parts:
    if ob.vertex_groups:
        continue
    for name in bone_names:
        ob.vertex_groups.new(name=name)
    groups = {name: ob.vertex_groups[name] for name in bone_names}
    for v in ob.data.vertices:
        _, idx, _ = kd.find(v.co)
        for g in me.vertices[idx].groups:
            gname = body.vertex_groups[g.group].name
            if gname in groups and g.weight > 0:
                groups[gname].add([v.index], g.weight, "REPLACE")

activate(body)
for ob in parts:
    ob.select_set(True)
bpy.ops.object.join()

# keep only bone deform groups
for g in list(body.vertex_groups):
    if g.name not in B:
        body.vertex_groups.remove(g)

# ----------------------------------------------------------------------------- 5. exact height
co = get_co(me)
zmin, zmax = co[:, 2].min(), co[:, 2].max()
f = HEIGHT / (zmax - zmin)
co[:, 2] -= zmin
co *= f
set_co(me, co)
activate(rig)
bpy.ops.object.mode_set(mode="EDIT")
for eb in rig.data.edit_bones:
    eb.head = (eb.head - Vector((0, 0, zmin))) * f
    eb.tail = (eb.tail - Vector((0, 0, zmin))) * f
bpy.ops.object.mode_set(mode="OBJECT")
body.parent = rig
body.matrix_parent_inverse = Matrix.Identity(4)
arm_mod = body.modifiers.new("Armature", "ARMATURE")
arm_mod.object = rig
B = {b.name: b for b in rig.data.bones}

# ----------------------------------------------------------------------------- 6. report
co = get_co(me)
W = weights(body)
headv = co[W["head"] > 0.95]
chin_z = headv[headv[:, 1] < np.percentile(headv[:, 1], 15), 2].min()
waist = co[(np.abs(co[:, 2] - (belt_z * f + 0.06)) < 0.01) & (wsum(W, ARM["l"] + ARM["r"]) < 0.05)]
hand_len = (B["middle_03_l"].tail_local - B["hand_l"].head_local).length
unweighted = sum(1 for v in me.vertices if not v.groups)
print(f"[superhero] height={co[:, 2].max() - co[:, 2].min():.4f} m  head={co[:, 2].max() - chin_z:.3f} m "
      f"(1/{(co[:, 2].max() - co[:, 2].min()) / (co[:, 2].max() - chin_z):.2f})  "
      f"shoulder_joints={2 * B['upperarm_l'].head_local.x:.3f} m  waist_w={np.ptp(waist[:, 0]):.3f} m  "
      f"hand={hand_len:.3f} m  tris={sum(len(p.vertices) - 2 for p in me.polygons)}  unweighted={unweighted}")

# ----------------------------------------------------------------------------- 7. save & export (rest pose)
me.validate()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "superhero.blend"))
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, "superhero.glb"), export_format="GLB")
activate(rig)
body.select_set(True)
bpy.ops.export_scene.fbx(filepath=os.path.join(OUT, "superhero.fbx"), use_selection=True,
                         add_leaf_bones=False, mesh_smooth_type="FACE")

# ----------------------------------------------------------------------------- 8. renders
scene.render.engine = "BLENDER_EEVEE"
scene.view_settings.view_transform = "Standard"
scene.render.resolution_x, scene.render.resolution_y = 900, 1200
world = bpy.data.worlds.new("studio")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.82, 0.84, 0.88, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.55
scene.world = world
for name, loc, energy, size in (("key", (-2.5, -3.0, 3.2), 900, 2.5), ("fill", (3.0, -2.0, 1.5), 300, 3.0),
                                ("rim", (1.5, 3.0, 3.0), 700, 2.0)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy
    ld.size = size
    lo = bpy.data.objects.new(name, ld)
    lo.location = loc
    lo.rotation_euler = (Vector((0, 0, 1.0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(lo)
rig.hide_render = True
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
os.makedirs(os.path.join(OUT, "renders"), exist_ok=True)


def shoot(label, ang, ortho=True, z=0.96, dist=6.0, look_z=0.96, lens=50, scale=2.25):
    a = math.radians(ang)
    cam.data.type = "ORTHO" if ortho else "PERSP"
    cam.data.ortho_scale = scale
    cam.data.lens = lens
    cam.location = (math.sin(a) * dist, -math.cos(a) * dist, z)
    cam.rotation_euler = (Vector((0, 0, look_z)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = os.path.join(OUT, "renders", f"{label}.png")
    bpy.ops.render.render(write_still=True)


shoot("front", 0)
shoot("side", 90)
shoot("back", 180)


# lobby-style hero pose (render only; exported files stay in the rest A-pose)
def rotate_world(pb, axis, angle):
    h = pb.head.copy()
    R = Matrix.Translation(h) @ Matrix.Rotation(angle, 4, axis.normalized()) @ Matrix.Translation(-h)
    pb.matrix = R @ pb.matrix
    bpy.context.view_layer.update()


P = rig.pose.bones
for s, sign in (("l", 1), ("r", -1)):
    rotate_world(P[f"upperarm_{s}"], Vector((0, sign, 0)), math.radians(15))
    d = (P[f"lowerarm_{s}"].tail - P[f"lowerarm_{s}"].head).normalized()
    rotate_world(P[f"lowerarm_{s}"], d.cross(Vector((0, -1, 0))), math.radians(25))
    # palm normal: perpendicular to the knuckle line and finger direction, on the thumb's side
    t = P[f"index_01_{s}"].head - P[f"pinky_01_{s}"].head
    f_dir = P[f"middle_01_{s}"].head - P[f"hand_{s}"].head
    palm = t.cross(f_dir).normalized()
    knuckles = (P[f"index_01_{s}"].head + P[f"pinky_01_{s}"].head) / 2
    if palm.dot(P[f"thumb_03_{s}"].tail - knuckles) < 0:
        palm = -palm
    for finger, angles in (("index", (90, 100, 70)), ("middle", (92, 100, 70)), ("ring", (94, 100, 70)),
                           ("pinky", (96, 100, 70)), ("thumb", (25, 35, 30))):
        for i, ang in enumerate(angles, 1):
            pb = P[f"{finger}_0{i}_{s}"]
            d = (pb.tail - pb.head).normalized()
            rotate_world(pb, d.cross(palm), math.radians(ang))
shoot("hero_three_quarter", -28, ortho=False, z=0.85, dist=6.4, look_z=0.98, lens=85)
shoot("hero_face", -20, ortho=False, z=1.72, dist=1.35, look_z=1.70, lens=70)
