"""Which arm of the emission ablation produced an artifact.

The task frames reaching plan_pour_tea come from one of two sources: the
hand-authored frames.json sidecars (the ground-truth arm) or a validated
touchpoint-#2 selections artifact (the VLM arm). Everything downstream --
plan npz, rendered mp4, executed mp4, metrics json -- used to land on the
same filenames, so the second run silently overwrote the first and the
ablation had no artifact-level record of which arm it was looking at.

This module is the one place that distinction is named. The arm is
STAMPED INTO THE PLAN at planning time (`stamp()`), because that is the
only moment it is known for certain; the render and execute scripts read
the stamp back (`resolve_arm()`) rather than trusting a flag that a tired
hand can set wrong at 2am. An explicit --arm that contradicts the stamp
is an error, not an override -- a mislabeled video is exactly the failure
this module exists to prevent.

Layout (arm in the directory AND in the filename, so a file that gets
dragged out of its folder still says what it is):

    outputs/plans/<arm>/pour_tea_full_<arm>.npz
    outputs/videos/<arm>/pour_tea_full_<arm>.mp4
    outputs/videos/<arm>/pour_tea_exec_<arm>.mp4
    outputs/metrics/<arm>/pour_tea_exec_<arm>.json
"""

from pathlib import Path

import numpy as np

ARMS = ("hand", "vlm")
AUTO = "auto"

LABEL = {
    "hand": "hand-authored frames.json (ground-truth arm)",
    "vlm": "VLM-selected frames (selections artifact)",
}

# pre-provenance artifacts, still readable with an explicit --arm
LEGACY_PLAN = Path("outputs/plans/pour_tea_full.npz")


# ------------------------------------------------------------------ naming
def arm_of_run(selections=None) -> str:
    """The arm a *planning* run is on: --selections present -> vlm."""
    return "vlm" if selections else "hand"


def tagged(root, stem: str, suffix: str, arm: str) -> Path:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (expected one of {ARMS})")
    return Path(root) / arm / f"{stem}_{arm}{suffix}"


def plan_path(arm: str) -> Path:
    return tagged("outputs/plans", "pour_tea_full", ".npz", arm)


def video_path(stem: str, arm: str) -> Path:
    return tagged("outputs/videos", stem, ".mp4", arm)


def metrics_path(stem: str, arm: str) -> Path:
    return tagged("outputs/metrics", stem, ".json", arm)


# ------------------------------------------------------------- npz stamping
def stamp(arm: str, selections=None, emissions=None) -> dict:
    """Provenance fields to splat into np.savez(**stamp(...)).

    Stored as 0-d unicode arrays, so np.load reads them back without
    allow_pickle.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (expected one of {ARMS})")
    return {"arm": np.array(arm),
            "selections": np.array(str(selections or "")),
            "emissions": np.array(str(emissions or ""))}


def _read_str(plan, key):
    if key not in getattr(plan, "files", ()):
        return None
    return str(np.asarray(plan[key]).item())


def read_arm(plan):
    """Arm stamped in a loaded npz, or None for a pre-provenance plan."""
    arm = _read_str(plan, "arm")
    if arm is not None and arm not in ARMS:
        raise SystemExit(f"[provenance] plan stamped with unknown arm "
                         f"{arm!r}; expected one of {ARMS}")
    return arm


def read_selections(plan):
    """Selections path stamped in a loaded npz ('' on the hand arm)."""
    return _read_str(plan, "selections")


def read_emissions(plan):
    """Emissions path stamped in a loaded npz ('' when stages 2-3 were
    planned on pour_stages.*; None for a pre-provenance plan)."""
    return _read_str(plan, "emissions")


# ------------------------------------------------------------ CLI resolution
def add_arm_flag(ap) -> None:
    ap.add_argument("--arm", choices=(*ARMS, AUTO), default=AUTO,
                    help="which arm of the emission ablation this run "
                         "belongs to; 'auto' (default) reads the stamp "
                         "written into the plan npz by plan_pour_tea.py. "
                         "Names the output folder and filename suffix.")


def resolve_plan_path(cli_plan, arm: str) -> Path:
    """Pick the plan to load before it has been opened.

    Explicit --plan wins. Otherwise an explicit --arm selects its own
    plan; under --arm auto we take the one arm that has a plan on disk
    and refuse to guess when both do.
    """
    if cli_plan:
        return Path(cli_plan)
    if arm != AUTO:
        return plan_path(arm)

    present = [a for a in ARMS if plan_path(a).exists()]
    if len(present) == 1:
        return plan_path(present[0])
    if len(present) > 1:
        raise SystemExit(
            "[provenance] both arms have a plan on disk "
            f"({', '.join(str(plan_path(a)) for a in present)}); pass "
            "--arm hand or --arm vlm to say which one this run is about.")
    if LEGACY_PLAN.exists():
        return LEGACY_PLAN
    raise SystemExit(
        "[provenance] no plan found under outputs/plans/<arm>/; run "
        "scripts/plan_pour_tea.py (add --selections for the VLM arm) "
        "first, or pass --plan explicitly.")


def resolve_arm(cli_arm: str, plan, path) -> str:
    """Reconcile --arm against the arm stamped in the loaded plan."""
    stamped = read_arm(plan)

    if stamped is None:                       # pre-provenance artifact
        if cli_arm == AUTO:
            raise SystemExit(
                f"[provenance] {path} carries no arm stamp (it predates "
                "provenance). Replan to stamp it, or pass --arm hand / "
                "--arm vlm to say which arm produced it -- guessing here "
                "is how a VLM run ends up filed as ground truth.")
        print(f"[provenance] WARNING: {path} carries no arm stamp; "
              f"trusting --arm {cli_arm}. Replan to make this automatic.")
        return cli_arm

    if cli_arm != AUTO and cli_arm != stamped:
        raise SystemExit(
            f"[provenance] --arm {cli_arm} contradicts {path}, which is "
            f"stamped '{stamped}'. Loading a {stamped}-arm plan and "
            f"filing it under {cli_arm} would mislabel the video; drop "
            "--arm, or point --plan at the other arm's plan.")

    return stamped


def announce(arm: str, plan, path) -> None:
    sels = read_selections(plan)
    detail = f" <- {sels}" if sels else ""
    print(f"[provenance] arm '{arm}': {LABEL[arm]}{detail}")
    print(f"[provenance] plan artifact: {path}")
