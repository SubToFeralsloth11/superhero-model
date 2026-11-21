# Superhero base model

A one-colour, eyeless superhero figure with Fortnite-style proportions, built from a script.

- Height about 1.88 m, A-pose, Z up, facing -Y (Blender front view)
- One fused, watertight, smooth-shaded mesh (about 40k triangles) with a single `Suit` material
- Rigged to a skeleton that uses the Unreal mannequin bone names (`pelvis`, `spine_01-03`, `neck_01`, `head`, `clavicle/upperarm/lowerarm/hand_l/r`, `thigh/calf/foot/ball_l/r`), with automatic weights

Files: `superhero.glb`, `superhero.fbx`, `superhero.blend`, and previews in `renders/`.

Rebuild (takes about 4 minutes):

```
blender --background --python build_superhero.py
```

To change the colour, edit `SUIT_COLOUR` at the top of the script. To change the mesh density, edit `VOXEL` and `TARGET_TRIS`.
