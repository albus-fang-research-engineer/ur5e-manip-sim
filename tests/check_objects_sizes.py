from robosuite.models.objects import MujocoXMLObject

for name, path in {"mug": "assets/objects/mug/mug.xml",
                   "teapot": "assets/objects/teapot/teapot.xml"}.items():
    o = MujocoXMLObject(path, name=name, joints=[dict(type="free")],
                        obj_type="all", duplicate_collision_geoms=True)
    print(f"{name:8s} r_h={o.horizontal_radius:.4f}  bottom={o.bottom_offset}  top={o.top_offset}")