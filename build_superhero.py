"""Builds a Fortnite-proportioned, single-colour, eyeless superhero figure.

Run: blender --background --python build_superhero.py
Outputs superhero.blend, superhero.glb, superhero.fbx and preview renders in ./renders.
Units: metres, Z up, character faces -Y. Height ~1.90 m (Fortnite default body).
"""
import math
import os

import bmesh
import bpy
from mathutils import Matrix, Vector

OUT = os.path.dirname(os.path.abspath(__file__))
SUIT_COLOUR = (0.05, 0.16, 0.75, 1.0)  # the one colour; change freely
VOXEL = 0.0045
TARGET_TRIS = 40000

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
BM = bmesh.new()  # every primitive goes in here, then gets voxel-fused


def _rot(deg):
    from mathutils import Euler
    return Euler([math.radians(a) for a in deg]).to_matrix().to_4x4()


def blob(loc, scale, rot=(0, 0, 0), segs=28):
    """Ellipsoid: centre, semi-axes, euler degrees."""
    m = Matrix.Translation(Vector(loc)) @ _rot(rot) @ Matrix.Diagonal((*scale, 1))
    bmesh.ops.create_uvsphere(BM, u_segments=segs, v_segments=segs // 2, radius=1.0, matrix=m)


def frame(axis_z, axis_x):
    z = Vector(axis_z).normalized()
    x = (Vector(axis_x) - z * Vector(axis_x).dot(z)).normalized()
    return x, z.cross(x), z


def framed_blob(centre, axis_z, axis_x, scale):
    """Ellipsoid whose local Z follows axis_z and local X follows axis_x."""
    x, y, z = frame(axis_z, axis_x)
    rot = Matrix((x, y, z)).transposed().to_4x4()
    m = Matrix.Translation(Vector(centre)) @ rot @ Matrix.Diagonal((*scale, 1))
    bmesh.ops.create_uvsphere(BM, u_segments=28, v_segments=14, radius=1.0, matrix=m)


def limb(points, flat=1.0):
    """Tube through [(x, y, z, r), ...]: densely swept spheres, fused smooth by the remesh.
    flat < 1 squashes the cross-section front-to-back (Y) for torso-like shapes."""
    for (a, b) in zip(points, points[1:]):
        pa, pb = Vector(a[:3]), Vector(b[:3])
        n = max(2, int((pb - pa).length / (min(a[3], b[3]) * 0.3)) + 1)
        for i in range(n + 1):
            t = i / n
            r = a[3] + (b[3] - a[3]) * t
            blob(pa.lerp(pb, t), (r, r * flat, r), segs=20)


def torus(centre, axis, major, minor, squash=1.0):
    """Ring around `axis` through `centre` (cuffs, belt)."""
    q = Vector(axis).to_track_quat('Z', 'Y').to_matrix().to_4x4()
    m = Matrix.Translation(Vector(centre)) @ q @ Matrix.Diagonal((1, squash, 1, 1))
    segs, rings = 48, 10
    grid = []
    for i in range(segs):
        a = 2 * math.pi * i / segs
        row = []
        for j in range(rings):
            b = 2 * math.pi * j / rings
            r = major + minor * math.cos(b)
            row.append(BM.verts.new(m @ Vector((r * math.cos(a), r * math.sin(a), minor * math.sin(b)))))
        grid.append(row)
    for i in range(segs):
        for j in range(rings):
            BM.faces.new((grid[i][j], grid[(i + 1) % segs][j],
                          grid[(i + 1) % segs][(j + 1) % rings], grid[i][(j + 1) % rings]))


def chain(a, b, radii):
    """Points evenly spaced from a to b with the given radii."""
    a, b = Vector(a), Vector(b)
    n = len(radii) - 1
    return [(*(a.lerp(b, i / n)), r) for i, r in enumerate(radii)]


# Joint layout shared by the mesh and the skeleton (metres, A-pose).
def arm_joints(s):
    S = Vector((s * 0.2, 0.0, 1.485))
    E = S + Vector((s * 0.72, 0.03, -0.69)).normalized() * 0.29
    fd = Vector((s * 0.66, -0.14, -0.74)).normalized()
    W = E + fd * 0.26
    return S, E, W, fd


def leg_joints(s):
    return (Vector((s * 0.095, 0.0, 0.95)), Vector((s * 0.108, -0.012, 0.52)),
            Vector((s * 0.118, 0.012, 0.095)))


# ---------------------------------------------------------------- head & neck
# Smooth full-face mask, no eyes or mouth: skull, jaw, brow plane, subtle nose ridge.
blob((0, 0.008, 1.765), (0.094, 0.108, 0.118))            # cranium
blob((0, -0.02, 1.705), (0.078, 0.082, 0.072))            # jaw / cheeks
blob((0, -0.052, 1.668), (0.042, 0.034, 0.03))            # chin
blob((0, -0.058, 1.765), (0.07, 0.036, 0.028))            # brow plane
blob((0, -0.088, 1.728), (0.012, 0.014, 0.028), (-12, 0, 0))  # nose ridge
limb([(0, 0.014, 1.5, 0.064), (0, 0.008, 1.6, 0.058), (0, 0.0, 1.69, 0.052)])

# ---------------------------------------------------------------- torso (V-taper hero build)
limb([(0, 0.008, 0.96, 0.15), (0, 0.004, 1.06, 0.135), (0, 0.0, 1.16, 0.13),
      (0, 0.0, 1.27, 0.145), (0, 0.005, 1.37, 0.17), (0, 0.012, 1.45, 0.175),
      (0, 0.018, 1.49, 0.13)], flat=0.66)
blob((0, 0.035, 1.42), (0.16, 0.085, 0.12))               # upper back
for s in (-1, 1):
    blob((s * 0.09, 0.025, 1.53), (0.09, 0.06, 0.045), (0, s * 20, 0))   # traps
    blob((s * 0.07, -0.045, 1.415), (0.1, 0.06, 0.068), (0, s * -10, 0))  # pecs
    blob((s * 0.118, 0.018, 1.34), (0.06, 0.06, 0.12), (0, s * 14, 0))   # lats
    blob((s * 0.065, 0.04, 0.96), (0.078, 0.06, 0.085))                  # glutes
    for z, front in ((1.265, -0.0952), (1.195, -0.0875), (1.125, -0.0868)):
        blob((s * 0.036, front + 0.017, z), (0.04, 0.022, 0.033))        # abs: ~5 mm relief
torus((0, 0.004, 1.05), (0, 0, 1), 0.133, 0.011, squash=0.68)             # belt line

# ---------------------------------------------------------------- arms (A-pose, ~45 deg)
for s in (-1, 1):
    S, E, W, fd = arm_joints(s)
    limb([(s * 0.07, 0.01, 1.5, 0.075), (*S, 0.07)])                       # shoulder girdle
    blob(S + Vector((s * 0.015, 0.0, -0.015)), (0.072, 0.072, 0.085), (0, s * 40, 0))  # deltoid
    limb(chain(S, E, [0.062, 0.064, 0.063, 0.056, 0.05]))                  # upper arm
    blob(S.lerp(E, 0.5) + Vector((0, -0.025, 0)), (0.045, 0.04, 0.09), (0, s * -45, 0))  # bicep
    limb(chain(E, W, [0.05, 0.056, 0.052, 0.045, 0.038]))                  # forearm
    torus(W - fd * 0.03, fd, 0.042, 0.008)                                 # glove cuff

    # hand: chunky, palm toward the thigh like Fortnite's A-pose
    palm_normal = Vector((-s * 0.7, 0.0, -0.7))
    x, y, z = frame(fd, palm_normal)                    # x: palm side, z: along fingers
    side = y if y.y < 0 else -y                         # thumb side (forward)
    palm_c = W + z * 0.055
    framed_blob(palm_c, z, x, (0.028, 0.05, 0.06))
    for off, length in ((0.033, 0.075), (0.011, 0.085), (-0.011, 0.08), (-0.033, 0.066)):
        base = palm_c + z * 0.045 + side * off
        mid = base + z * length * 0.55 + x * 0.008
        tip = mid + z * length * 0.4 + x * 0.024
        limb([(*base, 0.0145), (*mid, 0.013), (*tip, 0.0115)])
    t0 = palm_c + side * 0.035 + x * 0.012 - z * 0.01
    t1 = t0 + side * 0.03 + z * 0.025 + x * 0.014
    t2 = t1 + side * 0.008 + z * 0.035 + x * 0.012
    limb([(*t0, 0.019), (*t1, 0.0155), (*t2, 0.013)])

# ---------------------------------------------------------------- legs
for s in (-1, 1):
    H, K, A = leg_joints(s)
    limb([(s * 0.06, 0.005, 0.98, 0.1), (*H, 0.1)])                        # hip
    limb(chain(H, K, [0.1, 0.094, 0.085, 0.074, 0.062]))                   # thigh
    blob(H.lerp(K, 0.55) + Vector((s * 0.01, -0.03, 0)), (0.065, 0.055, 0.14))  # quads
    limb(chain(K, A, [0.058, 0.06, 0.054, 0.046, 0.042, 0.042]))           # shin
    blob(K.lerp(A, 0.28) + Vector((0, 0.022, 0)), (0.054, 0.05, 0.11))     # calf
    blob(K + Vector((0, -0.035, 0)), (0.042, 0.03, 0.04))                  # kneecap
    torus(Vector((s * 0.118, 0.008, 0.21)), (0, 0, 1), 0.047, 0.008)       # boot cuff
    blob((s * 0.118, 0.025, 0.065), (0.05, 0.055, 0.068))                  # heel
    blob((s * 0.12, -0.05, 0.05), (0.054, 0.125, 0.05))                    # foot
    blob((s * 0.122, -0.145, 0.04), (0.05, 0.05, 0.04))                    # toe

# ---------------------------------------------------------------- fuse into one mesh
me = bpy.data.meshes.new("Superhero")
BM.to_mesh(me)
BM.free()
body = bpy.data.objects.new("Superhero", me)
scene.collection.objects.link(body)
bpy.context.view_layer.objects.active = body
body.select_set(True)

rem = body.modifiers.new("remesh", "REMESH")
rem.mode = "VOXEL"
rem.voxel_size = VOXEL
rem.adaptivity = 0.0
bpy.ops.object.modifier_apply(modifier="remesh")

sm = body.modifiers.new("smooth", "CORRECTIVE_SMOOTH")
sm.factor = 0.5
sm.iterations = 6
sm.use_only_smooth = True
bpy.ops.object.modifier_apply(modifier="smooth")

# flat soles at z = 0
bm = bmesh.new()
bm.from_mesh(body.data)
res = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                             plane_co=(0, 0, 0.004), plane_no=(0, 0, -1), clear_outer=True)
cut_edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
bmesh.ops.holes_fill(bm, edges=cut_edges)
for v in bm.verts:
    v.co.z -= 0.004
bm.to_mesh(body.data)
bm.free()

tris = sum(len(p.vertices) - 2 for p in body.data.polygons)
dec = body.modifiers.new("decimate", "DECIMATE")
dec.ratio = min(1.0, TARGET_TRIS / tris)
bpy.ops.object.modifier_apply(modifier="decimate")
bpy.ops.object.shade_smooth()

# ---------------------------------------------------------------- one-colour suit material
mat = bpy.data.materials.new("Suit")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = SUIT_COLOUR
bsdf.inputs["Roughness"].default_value = 0.42
bsdf.inputs["Coat Weight"].default_value = 0.25
mat.diffuse_color = SUIT_COLOUR
body.data.materials.append(mat)

# ---------------------------------------------------------------- UE-style skeleton (Fortnite uses UE mannequin names)
arm_data = bpy.data.armatures.new("Skeleton")
rig = bpy.data.objects.new("Skeleton", arm_data)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
eb = arm_data.edit_bones


def bone(name, head, tail, parent=None):
    b = eb.new(name)
    b.head, b.tail = head, tail
    if parent:
        b.parent = eb[parent]
        b.use_connect = (Vector(eb[parent].tail) - Vector(head)).length < 1e-4
    return name


bone("pelvis", (0, 0, 0.98), (0, 0, 1.08))
bone("spine_01", (0, 0, 1.08), (0, 0, 1.2), "pelvis")
bone("spine_02", (0, 0, 1.2), (0, 0, 1.34), "spine_01")
bone("spine_03", (0, 0, 1.34), (0, 0.0, 1.52), "spine_02")
bone("neck_01", (0, 0.01, 1.55), (0, 0.0, 1.66), "spine_03")
bone("head", (0, 0.0, 1.66), (0, 0.0, 1.88), "neck_01")
for s, sfx in ((1, "l"), (-1, "r")):
    S, E, W, fd = arm_joints(s)
    bone(f"clavicle_{sfx}", (s * 0.03, 0.0, 1.5), S, "spine_03")
    bone(f"upperarm_{sfx}", S, E, f"clavicle_{sfx}")
    bone(f"lowerarm_{sfx}", E, W, f"upperarm_{sfx}")
    bone(f"hand_{sfx}", W, W + fd * 0.17, f"lowerarm_{sfx}")
    H, K, A = leg_joints(s)
    bone(f"thigh_{sfx}", H, K, "pelvis")
    bone(f"calf_{sfx}", K, A, f"thigh_{sfx}")
    bone(f"foot_{sfx}", A, (s * 0.122, -0.09, 0.03), f"calf_{sfx}")
    bone(f"ball_{sfx}", (s * 0.122, -0.09, 0.03), (s * 0.124, -0.18, 0.03), f"foot_{sfx}")
bpy.ops.object.mode_set(mode="OBJECT")

bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.parent_set(type="ARMATURE_AUTO")
missing = [v.index for v in body.data.vertices if not v.groups]
print(f"[superhero] tris={sum(len(p.vertices) - 2 for p in body.data.polygons)} "
      f"unweighted_verts={len(missing)} height={max(v.co.z for v in body.data.vertices):.3f}")

# ---------------------------------------------------------------- export
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "superhero.blend"))
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, "superhero.glb"), export_format="GLB")
bpy.ops.export_scene.fbx(filepath=os.path.join(OUT, "superhero.fbx"), add_leaf_bones=False)

# ---------------------------------------------------------------- preview renders
rig.hide_render = True
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 700, 1000
scene.render.film_transparent = False
world = bpy.data.worlds.new("w")
world.color = (0.08, 0.08, 0.09)
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.25, 0.26, 0.3, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
for name, rot, energy in (("key", (50, 0, -35), 4.0), ("rim", (60, 0, 160), 3.0)):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = energy
    lo = bpy.data.objects.new(name, ld)
    lo.rotation_euler = [math.radians(a) for a in rot]
    scene.collection.objects.link(lo)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.type = "ORTHO"
cam.data.ortho_scale = 2.15
scene.collection.objects.link(cam)
scene.camera = cam
os.makedirs(os.path.join(OUT, "renders"), exist_ok=True)
for label, ang in (("front", 0), ("three_quarter", 35), ("side", 90), ("back", 180)):
    a = math.radians(ang)
    cam.location = (math.sin(a) * 6, -math.cos(a) * 6, 0.95)
    cam.rotation_euler = (math.radians(90), 0, a)
    scene.render.filepath = os.path.join(OUT, "renders", f"{label}.png")
    bpy.ops.render.render(write_still=True)
