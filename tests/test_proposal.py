"""Structural invariants of the interaction point proposal pool, run on
whatever converted meshes are present (skips if none — meshes are
gitignored). Complements the eyeball check in the script's dry run."""

import json
from pathlib import Path

import numpy as np
import pytest

from manip_sim.proposal import (NMS_RADIUS_M, POOL_BUDGET, load_obj,
                                part_masks, propose)

CASES = [(n, Path(f"assets/objects/{n}")) for n in ("teapot", "mug")]


@pytest.mark.parametrize("name,obj_dir", CASES)
def test_pool_invariants(name, obj_dir):
    mesh = obj_dir / "meshes" / f"{name}_visual.obj"
    if not mesh.exists():
        pytest.skip(f"{mesh} not converted")
    spec = json.loads((obj_dir / "frames.json").read_text())
    V, F = load_obj(mesh)
    pool = propose(name, V, F, spec)
    C = pool.candidates
    X = np.array([c["xyz"] for c in C])

    # budget respected (coverage re-admissions may exceed it, but only
    # by the number of parts — bounded and logged)
    assert len(C) <= POOL_BUDGET + len(pool.readmitted)

    # NMS: no two NON-readmitted candidates closer than the merge radius
    n_nms = len(C) - len(pool.readmitted)
    D = np.linalg.norm(X[:n_nms, None] - X[None, :n_nms], axis=-1)
    np.fill_diagonal(D, np.inf)
    assert D.min() >= NMS_RADIUS_M - 1e-9

    # all four classes represented
    assert {c["source"] for c in C} == {"constructed", "part",
                                        "curvature", "fps"}

    # every discovered part region covered by at least one candidate
    parts = set(part_masks(name, V, spec))
    covered = {c.get("part") for c in C}
    assert parts <= covered

    # constructed candidates carry symbols; frames.json points all present
    syms = {c.get("symbol") for c in C if c["source"] == "constructed"}
    assert set(spec.get("points", {})) <= syms

    # ids sequential and stable ordering by class priority
    assert [c["id"] for c in C] == list(range(len(C)))
    prio = {"constructed": 0, "part": 1, "curvature": 2, "fps": 3}
    order = [prio[c["source"]] for c in C]
    assert order == sorted(order)

    # determinism
    pool2 = propose(name, V, F, spec)
    assert all(np.allclose(a["xyz"], b["xyz"])
               for a, b in zip(C, pool2.candidates))