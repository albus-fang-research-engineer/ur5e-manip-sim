"""VLM touchpoint-#2 driver for the pour-tea task: one live (or
injected) select_point_axis call per frame role, from the --vlm render
manifests to the role-keyed selections artifact plan_pour_tea.py
consumes. This is the smallest live-loop closure over what already
exists — the render marks, the menu, the parser's accept set, and the
resolver all come from the SAME candidates.json + vlm_subset calls, so
this script only sequences them.

The four pour-tea roles, their objects, and the parts bias handed to
vlm_subset (in the full orchestrator these come from VLM call #1's
StagePlan; here they are the task's fixed structure — call #2 is what
is under test):

    grasp              teapot   handle    where the gripper holds
    transport_active   teapot   spout     the tip carried to the mug
    pour               teapot   spout     the pivot frame of the tilt
    transport_passive  mug      rim       the opening it must reach

Each role's call sends the eight full-res marked views from
outputs/candidates/vlm/<object>/manifest.json and a Vocabulary whose
menu is rebuilt from the same pool + parts filter; the manifest's menu
is cross-checked against the rebuild and a mismatch is a hard stop
(stale renders would desynchronize marks from the accept set).

Every selection is resolved through manip_sim.selection immediately —
a candidate the VLM may licitly pick but that cannot anchor a frame
fails HERE with a typed ResolutionError, not inside the planner.

Output: outputs/selections/pour_tea.json plus a .log.json with the
per-call attempt/rejection/flag records (Client.logs) — the audit
trail of what the model actually said before parsing.

Requires ANTHROPIC_API_KEY. Offline dry runs inject a transport:

    PYTHONPATH=. python scripts/select_frames.py            # live
    PYTHONPATH=. python scripts/select_frames.py --out other.json

Then:

    PYTHONPATH=. python scripts/preview_selections.py \
        outputs/selections/pour_tea.json --render outputs/selections/preview.png
    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/plan_pour_tea.py \
        --selections outputs/selections/pour_tea.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from manip_sim.frames import load_symbols
from manip_sim.selection import (load_pool, menu_from_pool,
                                 resolve_selection, vlm_subset)
from manip_sim.vlm import Client, StageSpec, Vocabulary

OBJECTS = {
    "teapot": Path("assets/objects/teapot"),
    "mug": Path("assets/objects/mug"),
}
VLM_DIR = Path("outputs/candidates/vlm")
OUT = Path("outputs/selections/pour_tea.json")

# ONE menu subset per object, covering all its stage parts — the render
# manifest is per object, and every role on that object shares its menu
# (the parts bias fights mark crowding; the stage text in the prompt is
# what differentiates roles). In the full orchestrator both come from
# VLM call #1's StagePlan; here they are the task's fixed structure —
# call #2 is what is under test.
OBJECT_PARTS: dict[str, tuple[str, ...]] = {
    "teapot": ("handle", "spout"),
    "mug": ("rim",),
}
ROLES: dict[str, tuple[str, StageSpec]] = {
    "grasp": ("teapot", StageSpec(
        index=1, name="grasp teapot handle", active="teapot",
        passive=None, parts=("handle",))),
    "transport_active": ("teapot", StageSpec(
        index=2, name="carry spout tip over the mug opening",
        active="teapot", passive="mug", parts=("spout",))),
    "pour": ("teapot", StageSpec(
        index=3, name="tilt about the spout tip to pour", active="teapot",
        passive="mug", parts=("spout",))),
    "transport_passive": ("mug", StageSpec(
        index=2, name="the mug opening the spout must reach", active="mug",
        passive=None, parts=("rim",))),
}


def role_inputs(obj: str) -> tuple[Vocabulary, list[Path]]:
    """Vocabulary (menu rebuilt from pool + the object's parts filter)
    and the eight view paths, cross-checked against the manifest written
    by render_candidates.py --vlm."""
    parts = OBJECT_PARTS[obj]
    mpath = VLM_DIR / obj / "manifest.json"
    if not mpath.exists():
        raise SystemExit(
            f"[select-frames] {mpath} missing — render first:\n"
            f"  MUJOCO_GL=osmesa PYTHONPATH=. python "
            f"scripts/render_candidates.py --object {obj} --vlm "
            f"--parts {' '.join(parts)}")
    manifest = json.loads(mpath.read_text())
    if tuple(manifest.get("parts_filter", [])) != parts:
        raise SystemExit(
            f"[select-frames] {mpath} was rendered with parts filter "
            f"{manifest.get('parts_filter')} but this task needs "
            f"{list(parts)} — re-render with matching --parts")
    menu = menu_from_pool(vlm_subset(load_pool(OBJECTS[obj]), list(parts)))
    if {str(i): t for i, t in menu.items()} != manifest["menu"]:
        raise SystemExit(
            f"[select-frames] menu rebuilt from candidates.json differs "
            f"from {mpath} — stale render; re-run --vlm")
    vocab = Vocabulary.from_asset_dirs(
        {n: d for n, d in OBJECTS.items()}, menu=menu)
    paths = [Path(v["path"]) for _, v in sorted(manifest["views"].items())]
    return vocab, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), metavar="JSON")
    args = ap.parse_args()

    pools = {n: load_pool(d) for n, d in OBJECTS.items()}
    syms = {n: load_symbols(d) for n, d in OBJECTS.items()}
    client = Client()

    selections: dict[str, dict] = {}
    inputs = {obj: role_inputs(obj) for obj in
              {o for o, _ in ROLES.values()}}
    for role, (obj, stage) in ROLES.items():
        vocab, views = inputs[obj]
        sel = client.select_point_axis(stage, vocab, views)
        rf = resolve_selection(sel, pools[obj], syms[obj])   # typed gate
        print(f"[select-frames] {role}: {rf.frame.comment}")
        selections[role] = {"candidate_id": sel.candidate_id,
                            "axis": sel.axis, "sign": sel.sign,
                            "secondary": sel.secondary,
                            "rationale": sel.rationale}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selections, indent=2) + "\n")
    log = out.with_suffix(".log.json")
    log.write_text(json.dumps([asdict(l) for l in client.logs], indent=2,
                              default=str) + "\n")
    print(f"[select-frames] wrote {out} (+ {log})")
    print("  next: preview_selections.py, then plan_pour_tea.py "
          "--selections")


if __name__ == "__main__":
    main()