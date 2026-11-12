"""Measure the sim 2F-85's collision geometry in the grip-site frame, for
manip_sim.antipodal: in the planner's default state, across its free-air
closing stroke, and closed on test cylinders.

The fingers are tendon-coupled and moved by physics (an underactuated distal
link on a weak coupling spring), so they cannot be posed kinematically. Every
state records each gripper collision geom as an ORIENTED box in the site frame
(center, rotation, half-sizes: the geom's own box, or a mesh geom's local
bounding box) plus the gripper joints, and once per mesh geom the convex hull
MuJoCo collides with, in that box's frame; antipodal.FingerState derives from it
the pads' inner faces, their tilt and the gap between them.

    default    the planner's kinematic state (the env right after reset, not
               stepped): partly open, pads 64 mm apart; it holds the inner
               knuckles at zero, off the physical stroke
    free_air   the arm held still and the finger actuators ramped from open
               to closed in free air, recorded at each actuator target
    on_object  closings on a static cylinder (axis along the site's y, i.e.
               across the closing plane, like the lying marker's barrel under
               a top-down grasp) of each diameter in DIAMETERS_MM, placed at
               the free-air pad center at that gap (moved toward the fingertips
               in 3 mm steps while the closing pads miss it: the pads close
               toed in, so thin objects are reached only near the tips); the
               state after the squeeze settles (ctrl ramped to fully closed,
               as the executor does). pad_grasp marks closings held by the
               pads and fingertips alone, on both sides; those are the closed
               geometry the sampler uses. Free air overstates the closed tilt:
               on the object the distal link gives back 6-9 deg, but the pads
               stay toed in (about 19-22 deg on a 19 mm cylinder)
    comparison the 19 mm cylinder (the marker's barrel) and a 19 mm flat box at
               three contact depths, against the free-air state with the same
               gap where the faces meet the object (both pads tangent to the
               same cylinder; for the box, the same closest approach)

The scene's objects stay where they are (well away from the gripper); the
test objects are added to a copy of the env's model XML.

    scripts/authoring_docker.sh python scripts/measure_gripper_closing.py

Writes manip_sim/data/sim_2f85_closing.json.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

from manip_sim.antipodal import FingerState, Gripper, SIDES, _interp_rows
from manip_sim.planning import ArmKinematics
from manip_sim.scene import load_scene, make_env

OUT = Path("manip_sim/data/sim_2f85_closing.json")
PREFIX = "gripper0_right_"
DIAMETERS_MM = (4, 6, 8, 10, 12, 14, 16, 19, 22, 25, 28, 32, 36, 40, 45, 50, 55, 60,
                65, 70, 75, 80, 85, 90)
SETTLE_STEPS = 150          # per free-air actuator target
OPEN_STEPS = 400            # open before closing on an object
RAMP_STEPS = 125            # ctrl 0 -> fully closed (the executor reaches it in ~5 env steps)
HOLD_STEPS = 700            # squeeze after the ramp


def site_obbs(m, d, sid):
    """{geom: {"c", "R", "h"}} of the gripper's collision geoms, site frame."""
    Rs, ps = d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid]
    out = {}
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not n.startswith(PREFIX) or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
            continue
        Rg, ab = d.geom_xmat[g].reshape(3, 3), m.geom_aabb[g]      # ab: local center, half-size
        c = Rs.T @ (d.geom_xpos[g] + Rg @ ab[:3] - ps)
        out[n[len(PREFIX):].replace("_collision", "")] = {
            "c": np.round(c, 6).tolist(), "R": np.round(Rs.T @ Rg, 6).ravel().tolist(),
            "h": np.round(ab[3:], 6).tolist()}
    return out


def mesh_hulls(m):
    """{geom: [[nx, ny, nz, d], ...]} for the gripper's mesh collision geoms:
    the halfspaces (n . x + d <= 0 inside) of the convex hull MuJoCo collides
    with, in the frame of the geom's oriented box (site_obbs' c and R)."""
    import trimesh
    out = {}
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if (not n.startswith(PREFIX) or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0)):
            continue
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]] - m.geom_aabb[g][:3]
        hull = trimesh.convex.convex_hull(v)
        planes = {tuple(np.round(np.r_[fn, -fn @ hull.vertices[f[0]]], 6))
                  for fn, f in zip(hull.face_normals, hull.faces)}
        out[n[len(PREFIX):].replace("_collision", "")] = sorted(list(p) for p in planes)
    return out


def joints(m, d):
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)[len(PREFIX):]:
            round(float(d.qpos[m.jnt_qposadr[j]]), 6)
            for j in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(PREFIX)}


def describe(obbs):
    s = FingerState.from_json(obbs)
    return {"gap_m": round(s.gap, 5), "gap_center_m": round(s.gap_center, 5),
            "pad_tilt_deg": {k: round(s.pad(k).tilt_deg, 2) for k in SIDES},
            "pad_z_m": [round(v, 5) for v in _faces_z(s)]}


def _faces_z(s):
    zl, zr = s.pad("left").z_extent(), s.pad("right").z_extent()
    return max(zl[0], zr[0]), min(zl[1], zr[1])


class Rig:
    """The env's model, optionally with a static test object at the grip site."""

    def __init__(self, env, kin, obj=None):
        xml = env.model.get_xml()
        if obj is not None:
            i = xml.rindex("</worldbody>")
            xml = xml[:i] + obj + xml[i:]
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = kin.data.qpos
        mujoco.mj_forward(self.m, self.d)
        self.q_arm, self.qa = kin.data.qpos[kin.qpos_ids].copy(), kin.qpos_ids
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, PREFIX + "grip_site")
        self.acts = [a for a in range(self.m.nu)
                     if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").startswith(PREFIX)]

    def step(self, ctrl, n=1):
        for _ in range(n):
            self.d.ctrl[self.acts] = ctrl
            mujoco.mj_step(self.m, self.d)
            self.d.qpos[self.qa] = self.q_arm             # arm held
            self.d.qvel[self.qa] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def touching(self, geom):
        g = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, geom)
        names = set()
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            if g in (c.geom1, c.geom2):
                o = c.geom2 if c.geom1 == g else c.geom1
                names.add((mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, o) or "")[len(PREFIX):]
                          .replace("_collision", ""))
        return sorted(names)


def probe_object(kin, shape, width, z):
    """XML of a static cylinder (axis along site y) or flat box (faces normal
    to site x), `width` across the closing axis, centered at site (0, 0, z)."""
    Rs = kin.data.site_xmat[kin.site_id].reshape(3, 3)
    pos = kin.data.site_xpos[kin.site_id] + Rs @ np.array([0.0, 0.0, z])
    R = Rs @ np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])     # local z -> site y
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    size = f"{width / 2} 0.03" if shape == "cylinder" else f"{width / 2} 0.03 0.03"
    return (f'<body name="probe_obj" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}">'
            f'<geom name="probe_geom" type="{shape}" size="{size}" friction="1 0.005 0.0001"/></body>')


def close_on(env, kin, shape, width, z):
    rig = Rig(env, kin, probe_object(kin, shape, width, z))
    rig.step(0.0, OPEN_STEPS)
    if rig.touching("probe_geom"):
        return None                                       # does not fit the open gripper
    first = None
    for k in range(RAMP_STEPS + HOLD_STEPS):
        rig.step(0.8 * min(1.0, k / RAMP_STEPS))
        if first is None and any("fingerpad" in n or "fingertip" in n for n in rig.touching("probe_geom")):
            first = describe(site_obbs(rig.m, rig.d, rig.sid))["pad_tilt_deg"]
    obbs = site_obbs(rig.m, rig.d, rig.sid)
    return {"object": shape, "width_m": round(width, 5), "object_z_m": round(z, 5),
            **describe(obbs), "pad_tilt_first_contact_deg": first,
            "touching": rig.touching("probe_geom"), "joints": joints(rig.m, rig.d), "obbs": obbs}


def main():
    env, _ = make_env(load_scene("scenes/insert_marker.json"), robot="UR5e", has_renderer=False)
    kin = ArmKinematics(env)

    # the planner's default state, as its kinematic model holds it
    rig = Rig(env, kin)
    default = {**describe(site_obbs(rig.m, rig.d, rig.sid)), "joints": joints(rig.m, rig.d),
               "obbs": site_obbs(rig.m, rig.d, rig.sid)}

    # free-air closing stroke
    lo, hi = rig.m.actuator_ctrlrange[rig.acts[0]]
    free = []
    for target in np.linspace(lo, hi, 41):
        rig.step(target, SETTLE_STEPS)
        obbs = site_obbs(rig.m, rig.d, rig.sid)
        free.append({"ctrl": round(float(target), 4), **describe(obbs), "joints": joints(rig.m, rig.d),
                     "obbs": obbs})
    free.sort(key=lambda r: r["gap_m"])
    free_states = [FingerState.from_json(r["obbs"]) for r in free if r["gap_m"] > 0]
    free_rows = tuple(zip([r["gap_m"] for r in free if r["gap_m"] > 0], free_states))

    def summary(s):
        return {"gap_m": round(s.gap, 5), "pad_tilt_deg": round(s.tilt_deg(), 2),
                "pad_center_z_m": round(0.5 * (s.pad("left").center[2] + s.pad("right").center[2]), 5)}

    def free_at(gap):
        return summary(_interp_rows(free_rows, gap))

    def reach(f, shape, z):
        """How far apart the pad faces are where they meet an object centered
        at site (0, 0, z): for a cylinder, twice the mean distance from its
        axis to the face planes (= its diameter when both pads touch it); for
        a flat box spanning the pads, their closest approach (FingerState.gap)."""
        if shape == "box":
            return f.gap
        P = np.array([0.0, 0.0, z])
        return float(sum(f.pad(k).normal @ (P - f.pad(k).center) for k in SIDES))

    def free_same_gap(shape, z, value):
        rows = tuple(sorted(((reach(f, shape, z), f) for f in free_states), key=lambda r: r[0]))
        return summary(_interp_rows(rows, value))

    # closed on cylinders, each at the free-air pad center for its diameter
    held = []
    for dm in DIAMETERS_MM:
        w = dm * 1e-3
        z0 = free_at(w)["pad_center_z_m"]
        for dz in np.arange(0.0, 0.0151, 0.003):
            r = close_on(env, kin, "cylinder", w, z0 + dz)
            if r is None or any("fingerpad" in n or "fingertip" in n for n in r["touching"]):
                break
        if r is None:
            print(f"  cylinder {dm} mm: does not fit the open gripper, skipped")
            continue
        if not any("fingerpad" in n or "fingertip" in n for n in r["touching"]):
            print(f"  cylinder {dm} mm: the pads never reach it ({r['touching']}), skipped")
            continue
        r["pad_grasp"] = (all(n.endswith(("fingerpad", "fingertip")) for n in r["touching"])
                          and all(any(n.startswith(k) for n in r["touching"]) for k in SIDES))
        held.append(r)
        print(f"  cylinder {dm:3d} mm: gap {r['gap_m']*1e3:5.1f} mm (centers {r['gap_center_m']*1e3:5.1f}), "
              f"tilt L/R {r['pad_tilt_deg']['left']:+.1f}/{r['pad_tilt_deg']['right']:+.1f} deg "
              f"(first contact {r['pad_tilt_first_contact_deg']['left']:+.1f}), object z "
              f"{r['object_z_m']*1e3:+.1f} mm, touching {r['touching']}")

    # free air vs on object at the marker's barrel width
    comparison = []
    for shape in ("cylinder", "box"):
        for dz in (-0.006, 0.0, 0.006):
            fa = free_at(0.019)
            r = close_on(env, kin, shape, 0.019, fa["pad_center_z_m"] + dz)
            if r is None:
                continue
            s = FingerState.from_json(r["obbs"])
            g_obj = reach(s, shape, r["object_z_m"])
            same_gap = free_same_gap(shape, r["object_z_m"], g_obj)
            tilt = 0.5 * (r["pad_tilt_deg"]["left"] + r["pad_tilt_deg"]["right"])
            comparison.append({
                "object": shape, "width_m": 0.019, "object_z_m": r["object_z_m"],
                "on_object": {"gap_m": r["gap_m"], "gap_center_m": r["gap_center_m"],
                              "faces_apart_at_object_m": round(g_obj, 5),
                              "pad_tilt_deg": round(tilt, 2),
                              "pad_tilt_first_contact_deg": r["pad_tilt_first_contact_deg"],
                              "touching": r["touching"]},
                "free_air_same_gap": same_gap,
                "tilt_difference_deg": round(tilt - same_gap["pad_tilt_deg"], 2)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "about": "sim Robotiq 2F-85 collision geoms as oriented boxes in the grip-site frame "
                 "(c: center, R: row-major rotation, h: half-sizes; meters); "
                 "pad tilt positive = toed in; written by scripts/measure_gripper_closing.py",
        "hulls": mesh_hulls(rig.m),
        "default": default, "free_air": free, "on_object": held, "comparison": comparison},
        indent=1) + "\n")

    g = Gripper()
    print(f"default: gap {default['gap_m']*1e3:.1f} mm, tilt {default['pad_tilt_deg']}")
    for r in free[::5]:
        print(f"free air ctrl {r['ctrl']:.2f}: gap {r['gap_m']*1e3:6.1f} mm "
              f"(centers {r['gap_center_m']*1e3:5.1f}), tilt {r['pad_tilt_deg']}")
    for c in comparison:
        o = c["on_object"]
        print(f"{c['object']} 19 mm at z {c['object_z_m']*1e3:+.1f} mm: on object tilt {o['pad_tilt_deg']:+.1f} deg "
              f"(first contact {o['pad_tilt_first_contact_deg']}, faces {o['faces_apart_at_object_m']*1e3:.1f} mm "
              f"apart at it, touching {o['touching']}); free air, same gap: {c['free_air_same_gap']['pad_tilt_deg']:+.1f} "
              f"deg (difference {c['tilt_difference_deg']:+.1f})")
    print(f"{len(free)} free-air states, {len(held)} on-object states "
          f"({sum(r['pad_grasp'] for r in held)} pad grasps) -> {OUT}; "
          f"open gap {g.approach().gap*1e3:.1f} mm, max grasp width {g.max_width*1e3:.1f} mm")


if __name__ == "__main__":
    main()
