To see candidate points of all objects:
```
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa     PYTHONPATH=. python scripts/render_candidates.py 
```
To see candidate points of specified objects:
```
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
python scripts/render_candidates.py --object mug 
```
