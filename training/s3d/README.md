# Structured3D → PlanForge training pairs

Turns each Structured3D house (`Structured3D/scene_xxxxx/annotation_3d.json`, the "Structure annotations"
download) into a (plan drawing, 3D truth) pair:

- `img/scene_xxxxx.png`: the floor plan drawn at 50 px/m with a randomised drafting style
- `mask/scene_xxxxx.npz`: `room` (PlanForge room classes, walls) and `icon` (1 window, 2 door) label masks
- `truth/scene_xxxxx.json`: 3D truth in metres: rooms (polygon, type, ceiling height), walls, openings
  (width, wall thickness, sill, head, which rooms each side, door swing and hinge), storey heights

```
S3D_ROOT=/path/to/Structured3D S3D_OUT=/path/to/out python training/s3d/s3d_pairs.py
```

Licence: Structured3D is for **non-commercial research and educational use only** and may **not be
redistributed in any form**. Keep the data and everything generated from it (images, masks, truth) private,
for example in a private Kaggle dataset. Never upload it to a public Hugging Face repo. Models trained on it
inherit the non-commercial restriction. Cite: Zheng et al., "Structured3D: A Large Photo-realistic Dataset
for Structured 3D Modeling", ECCV 2020.
