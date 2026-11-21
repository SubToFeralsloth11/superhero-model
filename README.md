# Superhero base model

A one-colour, eyeless superhero in the style of Fortnite's Boundless superhero outfits (full-mask look), built from a script.

## Dimensions

- **Height: exactly 1.920 m.** Epic's UEFN documentation says "Player characters in Fortnite are 192 cm tall".
- Head is about 1/7.8 of body height; hero build with broad shoulders, flared lats, a narrow waist and big gloved hands.
- A-pose rest pose, Z up, facing -Y (Blender front view).

## What's in it

- An anatomical base body from MakeHuman (MPFB), restyled toward Fortnite proportions.
- A full mask with no eyes or mouth: the fabric stretches over the eye sockets and mouth, and the ears are pressed flat.
- Suit details, all in the same colour:
  - gloves with cuffs
  - boots with cuffs and soles
  - a belt with a buckle
  - raised seam piping: neck ring, chest and back V, side, arm and leg seams
- A hex spandex normal map (`textures/hex_normal.png`).
- Two materials in the same colour: `Suit` (hex fabric) and `Trim` (glossy gloves, boots and belt).
- A UE-style `game_engine` skeleton (53 bones, finger bones included: `pelvis`, `spine_01-03`, `neck_01`, `head`, `clavicle/upperarm/lowerarm/hand_l/r`, `index/middle/ring/pinky/thumb_01-03_l/r`, `thigh/calf/foot/ball_l/r`), fully skinned.
- About 110k triangles.

## Files

- `superhero.glb`, `superhero.fbx` and `superhero.blend`: the model, in the rest A-pose.
- `renders/`: front, side and back views (orthographic), plus a lobby-style hero pose and a close-up of the mask.

## Rebuild

Needs Blender 4.2+ with the MPFB extension installed.

```
blender --background --python build_superhero.py
```

To change the colour, edit `SUIT_COLOUR`. To change the size of the fabric hexes, edit `HEX_SIZE`. Body shape is set in `TARGETS` (MakeHuman target weights).
