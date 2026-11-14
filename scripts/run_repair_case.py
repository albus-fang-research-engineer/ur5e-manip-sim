"""One repair-study case (docs/deferred.md D9): a known-good specification,
one injected fault, one feedback condition, run until the loop accepts a
specification that plans or its budget is spent; then the final
specification is graded over the planner seeds.

    conditions   typed         Langfold: critic (call #4) on the preview with its
                               typed facts; a planner failure goes to repair
                               (call #5) as a typed record; edits, re-emits
                 untyped       the same loop, every failure detail withheld
                               (captions, compile errors, the planner record)
                 render_check  OmniManip-style (Sec. 3.3, Alg. 1): a label per
                               stage (success / refine / failure); refine writes
                               the stage again near its previous answer (once per
                               stage), failure writes it again from its prompt; a
                               planner failure is a bare "failure": every editable
                               stage is written again
    modes        dry_run       round 1's requests written, no model call
                 replay        canned responses from --responses DIR:
                               <kind>_NN.txt, kind in critic / repair / check /
                               emit, in call order (tests, free)
                 live          PAID. Prints the call bound first; every response
                               is saved under OUT/responses/ (replayable)

Budgets (the same in every condition): --budget rounds that change the
specification (default 2: edits applied, or a stage written again) and
--call-cap VLM requests, parse retries included (default 8). Stopping
early is "not fixed".

    scripts/authoring_docker.sh python scripts/run_repair_case.py \\
        --task pour_tea --selections S.json --emissions E.json \\
        --fault pour_wrong_anchor --condition typed --out OUT --mode replay --responses R

Writes OUT/record.json (manip_sim.repair_study's run record), OUT/fault.json,
OUT/spec_start/, OUT/spec_faulty/, OUT/spec_final/ (+ per-seed plans and
grades), and per round OUT/round_N/ (images, report, prompts, responses).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manip_sim import critic as cr  # noqa: E402
from manip_sim import fault_injection as fi  # noqa: E402
from manip_sim import render_check as rc  # noqa: E402
from manip_sim import repair_loop as rl  # noqa: E402

CATALOG = Path(__file__).resolve().parents[1] / "tasks/repair_study/faults.json"
CONDITIONS = fi.CONDITIONS            # one source: fault_injection (typed, render_check, untyped)
SEEDS = (0, 1, 2)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _w(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=_jsonable) + "\n")


def request_doc(system: str, messages: list) -> dict:
    blocks = []
    for m in messages:
        for b in m["content"]:
            blocks.append({"type": "text", "text": b["text"]} if b["type"] == "text"
                          else {"type": "image"})
    return {"system": system, "blocks": blocks}


def request_text(doc: dict, images: list) -> str:
    it = iter(images)
    parts = ["=== system ===", doc["system"], "=== user ==="]
    for b in doc["blocks"]:
        parts.append(b["text"] if b["type"] == "text" else f"[image: {next(it, '?')}]")
    return "\n".join(parts) + "\n"


# ------------------------------------------------------------------ transports

class Replay:
    """Canned responses <kind>_NN.txt, in order; running out is an error."""

    def __init__(self, d: Path, kind: str):
        self.files = sorted(Path(d).glob(f"{kind}_*.txt"))
        self.kind, self.i = kind, 0

    def __call__(self, payload: dict) -> str:
        if self.i >= len(self.files):
            raise RuntimeError(f"replay: no {self.kind} response #{self.i + 1}")
        self.i += 1
        return self.files[self.i - 1].read_text()


class Record:
    """Live transport that saves every raw response, for replay."""

    def __init__(self, d: Path, kind: str):
        from manip_sim.vlm import _urllib_transport
        self.send, self.d, self.kind, self.i = _urllib_transport, Path(d), kind, 0

    def __call__(self, payload: dict) -> str:
        raw = self.send(payload)
        self.i += 1
        self.d.mkdir(parents=True, exist_ok=True)
        (self.d / f"{self.kind}_{self.i:02d}.txt").write_text(raw)
        return raw


# ------------------------------------------------------------------ the world

def load_world(task: str):
    from manip_sim.frames import load_symbols
    from manip_sim.scene import load_scene, make_env
    from manip_sim.state import PoseReader
    from manip_sim.tsr import pose_from_pos_quat_wxyz
    from scripts.run_authoring_ablation import TASKS
    t = TASKS[task]
    scene = load_scene(t["scene"], t["grounding"])
    env, objs = make_env(scene, robot="UR5e", has_renderer=False)
    reader = PoseReader(env, list(objs))
    poses = {n: pose_from_pos_quat_wxyz(*reader.pose(n)) for n in objs}
    env.close()
    symbols = {n: load_symbols(d) for n, d in scene.asset_dirs.items()}
    bound, synth = None, frozenset()
    if t["stage_plan"]:
        from scripts.plan_stages import load_bindings
        b = load_bindings(t["stage_plan"])
        bound = {r: st.index for r, (_, st) in b.roles.items()}
        synth = frozenset(b.entry_synthesized)
    return scene, poses, symbols, bound, synth, t


def _stage_sha(state, role) -> str:
    em = state.emissions[role]
    sel = state.selections.get(role)
    blob = json.dumps(asdict(em), sort_keys=True, default=str) + f"|{getattr(sel, 'candidate_id', None)}"
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _spec_from_state(spec: fi.Spec, state) -> fi.Spec:
    return replace(spec, selections=dict(state.selections), objects=dict(state.objects),
                   emissions={**spec.emissions, **state.emissions})


class Case:
    def __init__(self, a):
        from manip_sim.selection import load_pool
        from manip_sim.vlm import Client, Vocabulary
        self.a, self.out = a, Path(a.out)
        self.start = fi.load_spec(a.selections, a.emissions)
        self.scene, self.poses, self.symbols, bound, self.synth, self.t = load_world(a.task)
        self.asset_dirs = self.scene.asset_dirs
        spec, self.fault = self.start, None
        if a.fault:
            f = next((x for x in fi.load_catalog(CATALOG) if x.id == a.fault), None)
            if f is None:
                raise SystemExit(f"no fault {a.fault!r} in {CATALOG}")
            pools = {o: load_pool(self.asset_dirs[o]) for o in set(self.start.objects.values())}
            spec, self.fault = fi.inject(self.start, f, pools)
            _w(self.out / "fault.json", asdict(self.fault))
            print(f"[case] fault {self.fault.fault_id}: {self.fault.pointer}: "
                  f"{self.fault.before} -> {self.fault.after}")
        fi.write_spec(self.start, self.out / "spec_start")
        fi.write_spec(spec, self.out / "spec_faulty")
        self.spec = spec
        self.role_index = rl.role_index_for(spec, bound)
        self.state = rl.spec_state(spec, self.asset_dirs)
        self.vocab = Vocabulary.from_asset_dirs(self.asset_dirs)
        # the task's stage plan (run_authoring_ablation.TASKS) first; the path an
        # emissions artifact recorded only if it still exists; else the fixed stages
        sp = next((p for p in (self.t["stage_plan"], spec.extra.get("stage_plan"))
                   if p and Path(p).exists()), None)
        self.stages = rl.stage_specs(sp)
        self.arm = spec.extra.get("arm") or "framed-chained"
        self.colors = rl.object_colors(self.asset_dirs)
        self.task_text = spec.extra.get("task") or a.task.replace("_", " ")
        self.clients = {}
        if a.mode != "dry_run":
            for kind in ("critic", "repair", "check", "emit"):
                t = (Replay(a.responses, kind) if a.mode == "replay"
                     else Record(self.out / "responses", kind))
                self.clients[kind] = Client(transport=t)
        self.rounds: list[dict] = []
        self.n_preview, self.last_plan = 0, None
        self.changes = 0                      # rounds that changed the specification
        self.final = {"accepted": False, "planned_in_loop": False, "grades": [], "error": None}

    # -- accounting ---------------------------------------------------
    def n_calls(self) -> int:
        return sum(l.attempts for c in self.clients.values() for l in c.logs)

    def calls_since(self, marks: dict) -> list[dict]:
        out = []
        for kind, c in self.clients.items():
            for l in c.logs[marks.get(kind, 0):]:
                use = [u for u in (l.usage or []) if u]
                out.append({"touchpoint": l.touchpoint, "attempts": l.attempts,
                            "input_tokens": sum(u.get("input_tokens", 0) for u in use) or None,
                            "output_tokens": sum(u.get("output_tokens", 0) for u in use) or None,
                            "rejections": list(l.rejections)})
        return out

    def marks(self) -> dict:
        return {k: len(c.logs) for k, c in self.clients.items()}

    def room(self, need: int = 1) -> bool:
        return self.n_calls() + need <= self.a.call_cap

    def stage_names(self) -> dict:
        return {r: em.name for r, em in self.state.emissions.items()}

    def mover_roles(self) -> list:
        return [r for r in self.spec.roles if self.state.emissions[r].passive is not None]

    # -- one preview --------------------------------------------------
    def preview(self, rd: Path):
        import scripts.render_preview as rp
        chain = rl.compile_chain(self.state, self.spec.roles, self.poses, self.symbols,
                                 self.asset_dirs, self.role_index, self.synth)
        previews = rl.preview_inputs(self.state, chain, self.symbols)
        manifest = rl.render_round(self.scene, chain, previews, rd / "images", self.symbols,
                                   gripper=rp.sim_gripper())
        report = rl.preview_report(self.state, chain, previews)
        _w(rd / "report.json", report)
        _w(rd / "chain.json", [{"role": c.role, "stage": c.emission.name,
                                "compiled": c.compiled is not None, "error": c.error}
                               for c in chain])
        return chain, manifest, report

    def editable(self, chain):
        ctx = self.state.context_roles()
        return [c for c in chain if c.role not in ctx]

    def round_rec(self, kind, rd, marks, chain=None, **kw) -> dict:
        rec = {"round": len(self.rounds) + 1, "kind": kind, "dir": str(rd),
               "calls": self.calls_since(marks),
               "stage_sha": {self.state.emissions[r].name: _stage_sha(self.state, r)
                             for r in self.state.emissions},
               **kw}
        if chain is not None:
            rec["compiled"] = {c.emission.name: True if c.compiled is not None else c.error
                               for c in chain if c.emission.passive is not None}
        self.rounds.append(rec)
        return rec

    def write_spec(self, name: str) -> Path:
        d = self.out / name
        fi.write_spec(_spec_from_state(self.spec, self.state), d)
        return d

    def next_round_dir(self) -> Path:
        """One folder per preview round (round_1, round_2, ...)."""
        self.n_preview += 1
        return self.out / f"round_{self.n_preview}"

    def plan(self, rd: Path) -> dict:
        d = self.write_spec(rd.name + "_spec")
        res = rl.plan_and_grade(self.a.task, d, 0, self.stage_names(), self.mover_roles())
        self.last_plan = (d, res)          # reused as seed 0 of the final grade
        print(f"[case] planned seed 0: " + ("ok" if res["planned"] else
              f"FAILED {(res['failure'] or {}).get('type')}") +
              (f", grader {'pass' if res['success'] else 'fail'}" if res["planned"] else ""))
        return res

    # -- call #3 again ------------------------------------------------
    def rewrite(self, role: str, chain, feedback, rd: Path) -> dict:
        c = next(x for x in chain if x.role == role)
        stage = self.stages[role]
        inputs = rl.call3_inputs(role, stage, c, self.state, chain, self.scene, self.symbols,
                                 self.asset_dirs, self.role_index, self.arm,
                                 rd / f"call3_{role}")
        em, rec = rl.rerun_call3(self.clients["emit"], stage, self.vocab, inputs, c,
                                 self.symbols, feedback=feedback, room=self.room)
        if em is not None:
            self.state = replace(self.state, emissions={**self.state.emissions, role: em})
        rec["role"] = role
        return rec

    # -- conditions ---------------------------------------------------
    def run_langfold(self, untyped: bool):
        ledger = cr.LoopLedger(budget=self.a.budget)
        while True:
            rd = self.next_round_dir()
            chain, manifest, report = self.preview(rd)
            if untyped:
                views = rl.withheld_views(manifest, rd / "images", chain, self.colors)
                rep = rl.withheld_report(report)
            else:
                views = rl.views_from_manifest(manifest, rd / "images", chain, self.colors)
                rep = report
            if not self.room():
                return self.stop("call cap reached before the critic")
            m = self.marks()
            try:
                verdict = self.clients["critic"].critique_preview_v2(
                    rep, views, self.vocab, self.state, ledger)
            except Exception as e:           # noqa: BLE001 -- recorded, the run ends
                self.round_rec("critic", rd, m, chain, error=str(e))
                return self.stop(f"critic: {e}", error=True)
            outcome = self._admit(ledger, verdict)
            rec = self.round_rec("critic", rd, m, chain, verdict=verdict.verdict,
                                 diagnosis=verdict.diagnosis,
                                 edits=[self._edit(e) for e in verdict.edits])
            if outcome == "accept":
                res = self.plan(rd)
                rec["planned"] = {"ok": res["planned"], "failure": res["failure"]}
                if res["planned"]:
                    self.final.update(accepted=True, planned_in_loop=True)
                    return
                if ledger.rounds_used >= ledger.budget:
                    return self.stop("round budget spent (no repair call made)")
                if not self.room():
                    return self.stop("call cap reached before repair")
                m = self.marks()
                try:
                    fix = self.clients["repair"].repair_v2(
                        rep, res["failure"], self.vocab, self.state, ledger,
                        reason_withheld=untyped)
                except Exception as e:       # noqa: BLE001
                    self.round_rec("repair", rd, m, error=str(e))
                    return self.stop(f"repair: {e}", error=True)
                outcome = self._admit(ledger, fix)
                rrec = self.round_rec("repair", rd, m, diagnosis=fix.diagnosis,
                                      edits=[self._edit(e) for e in fix.edits])
                if outcome != "apply":
                    return self.stop("round budget spent (repair)")
                self._apply(ledger, fix, rd, chain, untyped, rrec)
                continue
            if outcome != "apply":
                return self.stop("round budget spent (critic)")
            self._apply(ledger, verdict, rd, chain, untyped, rec)

    def _admit(self, ledger, verdict):
        try:
            return ledger.admit(verdict)
        except cr.EditRefusal as e:
            return {"type": "preview_rejected", "refusal": e.reason}

    @staticmethod
    def _edit(e) -> dict:
        return {"target": e.target, "action": e.action, "token": e.token,
                "before": e.before, "after": e.after, "direction": e.direction}

    def _apply(self, ledger, verdict, rd, chain, untyped, rec):
        res = cr.apply_edits(self.state, verdict.edits, verdict.diagnosis)
        ledger.record(verdict)
        self.state = res.state
        self.changes += 1
        reemits = []
        for role in [r for r in self.spec.roles if r in res.reemit]:
            chain = rl.compile_chain(self.state, self.spec.roles, self.poses, self.symbols,
                                     self.asset_dirs, self.role_index, self.synth)
            c = next((x for x in chain if x.role == role), None)
            if c is None or c.no_entry or not self.room():
                reemits.append({"role": role, "error": "no entry or call cap; not re-emitted"})
                continue
            text = (rl.REEMIT_COMPILED.format(diagnosis=res.reemit[role])
                    if c.compiled is not None or untyped else
                    rl.REEMIT_UNCOMPILED.format(error=c.error, diagnosis=res.reemit[role]))
            m = self.marks()
            r = self.rewrite(role, chain, [(rl.emission_raw(c.emission), text)], rd)
            r["calls"] = self.calls_since(m)
            reemits.append(r)
        if reemits:
            rec["reemit"] = reemits
            for r in reemits:
                rec["calls"] += r.get("calls", [])

    def run_render_check(self):
        refined: set[str] = set()
        passed: set[tuple] = set()      # (role, stage sha, entry pose) already labeled success
        while True:
            rd = self.next_round_dir()
            chain, manifest, _ = self.preview(rd)
            rejected = None
            for c in self.editable(chain):
                m = self.marks()
                key = (c.role, _stage_sha(self.state, c.role),
                       hashlib.sha256(np.round(np.asarray(c.poses[c.emission.active]), 6)
                                      .tobytes()).hexdigest()[:12])
                if key in passed:
                    # Alg. 1 accepts a stage once: an unchanged stage at an unchanged
                    # entry is not checked (or paid for) again
                    continue
                if c.compiled is None:
                    # no region to render: a failure with no model call (logged)
                    chk = rc.Check(c.emission.name, "failure", "auto: the stage has no "
                                   "compiled region to render")
                else:
                    if not self.room():
                        return self.stop("call cap reached before a check")
                    view = rc.StageView(c.emission.name, c.emission.active, c.emission.passive,
                                        rl.stage_rows(c.emission),
                                        rl.stage_images(manifest, rd / "images", c.role))
                    try:
                        chk = rc.check_stage(self.clients["check"], self.task_text, view)
                    except Exception as e:   # noqa: BLE001
                        self.round_rec("check", rd, m, chain, stage=c.emission.name, error=str(e))
                        return self.stop(f"check: {e}", error=True)
                self.round_rec("check", rd, m, chain, stage=c.emission.name,
                               labels=[{"stage": chk.stage, "label": chk.label}],
                               reasoning=chk.reasoning)
                if chk.label == "success":
                    passed.add(key)
                else:
                    rejected = (c, rc.action_for(chk.label, c.role in refined))
                    break
            if rejected is None:
                res = self.plan(rd)
                self.rounds[-1]["planned"] = {"ok": res["planned"]}
                if res["planned"]:
                    self.final.update(accepted=True, planned_in_loop=True)
                    return
                targets = [(c, "regenerate") for c in self.editable(chain)]   # a bare failure
            else:
                targets = [rejected]
            if self.changes >= self.a.budget:
                return self.stop("round budget spent")
            self.changes += 1
            for c, action in targets:
                if not self.room():
                    return self.stop("call cap reached before a rewrite")
                fb = None
                if action == "regenerate_near":
                    refined.add(c.role)
                    fb = [(rl.emission_raw(c.emission), rc.REFINE_TURN)]
                chain_now = rl.compile_chain(self.state, self.spec.roles, self.poses,
                                             self.symbols, self.asset_dirs, self.role_index,
                                             self.synth)
                m = self.marks()
                r = self.rewrite(c.role, chain_now, fb, rd)
                self.round_rec("regenerate", rd, m, stage=c.emission.name, action=action,
                               rewrite=r)

    def stop(self, why: str, error: bool = False):
        print(f"[case] stop: {why}")
        self.final["stopped"] = why
        if error:
            self.final["error"] = why

    # -- run ----------------------------------------------------------
    def run(self):
        """The record is written whatever happens: an exception anywhere (a
        planner or grader crash, an edit the applier refuses, a render error)
        ends the run as outcome "error", never as a missing record."""
        try:
            self._run()
        except Exception as e:                # noqa: BLE001 -- recorded below
            import traceback
            traceback.print_exc()
            self.final["error"] = f"{type(e).__name__}: {e}"
        self._write_record()

    def _run(self):
        if self.a.condition == "render_check":
            self.run_render_check()
        else:
            self.run_langfold(untyped=self.a.condition == "untyped")
        self.write_spec("spec_final")
        if self.final["accepted"] and not self.final["error"]:
            # the loop's own seed-0 plan is the final specification's seed 0 (the
            # loop accepted that specification and planned it); the other seeds
            # plan in parallel beside it
            from concurrent.futures import ThreadPoolExecutor
            d, res0 = self.last_plan
            with ThreadPoolExecutor(len(SEEDS) - 1) as ex:
                rest = list(ex.map(lambda s: rl.plan_and_grade(self.a.task, d, s,
                                                               self.stage_names(),
                                                               self.mover_roles()), SEEDS[1:]))
            self.final["grades"] = [res0] + rest
            self.final["graded_in"] = str(d)

    def _write_record(self):
        from manip_sim.repair_study import sha
        final_spec = _spec_from_state(self.spec, self.state)
        record = {"fault": asdict(self.fault) if self.fault else None,
                  "condition": self.a.condition, "repeat": self.a.repeat,
                  "mode": self.a.mode, "model": _model(), "temperature": _temperature(),
                  "call_cap": self.a.call_cap, "round_budget": self.a.budget,
                  "start": {"selections": self.a.selections, "emissions": self.a.emissions},
                  "start_sha": sha(Path(self.a.selections), Path(self.a.emissions)),
                  "rounds": self.rounds, "final": self.final,
                  "diff": fi.spec_diff(self.start, final_spec), "n_calls": self.n_calls()}
        _w(self.out / "record.json", record)
        from manip_sim.repair_study import outcome
        print(f"[case] {self.a.condition}: {outcome(record)} after {self.n_calls()} call(s), "
              f"{self.changes} change round(s) -> {self.out / 'record.json'}")


def _model():
    from manip_sim import vlm
    return vlm.MODEL


def _temperature():
    from manip_sim import vlm
    return vlm.TEMPERATURE


def dry_run(a) -> int:
    from manip_sim.vlm import build_critic_prompt_v2
    case = Case(a)
    rd = case.out / "round_1"
    chain, manifest, report = case.preview(rd)
    for c in chain:
        print(f"[case] {c.role}: " + ("compiled" if c.compiled is not None
                                      else c.error or "context (not compiled)"))
    if a.condition in ("typed", "untyped"):
        if a.condition == "untyped":
            views = rl.withheld_views(manifest, rd / "images", chain, case.colors)
            report = rl.withheld_report(report)
        else:
            views = rl.views_from_manifest(manifest, rd / "images", chain, case.colors)
        system, messages = build_critic_prompt_v2(report, views, case.vocab)
        doc = request_doc(system, messages)
        _w(rd / "prompt.json", doc)
        (rd / "prompt.txt").write_text(request_text(doc, [str(p) for _, p in views]))
        print(f"[case] {a.condition}: critic prompt with {len(views)} image(s) -> {rd / 'prompt.txt'}")
    else:
        for c in case.editable(chain):
            view = rc.StageView(c.emission.name, c.emission.active, c.emission.passive,
                                rl.stage_rows(c.emission),
                                rl.stage_images(manifest, rd / "images", c.role))
            system, messages = rc.build_check_prompt(case.task_text, view)
            doc = request_doc(system, messages)
            _w(rd / f"check_{c.role}.json", doc)
            (rd / f"check_{c.role}.txt").write_text(request_text(doc, [str(p) for p in view.images]))
            print(f"[case] render_check: {c.role} with {len(view.images)} image(s)")
    print(f"[case] dry run -> {case.out}; no model call made")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True,
                    choices=("pour_tea", "insert_marker", "insert_marker_upright"))
    ap.add_argument("--selections", required=True)
    ap.add_argument("--emissions", required=True)
    ap.add_argument("--fault", default=None, help="fault id in tasks/repair_study/faults.json")
    ap.add_argument("--condition", choices=CONDITIONS, default="typed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("dry_run", "replay", "live"), default="dry_run")
    ap.add_argument("--responses", default=None, help="replay: dir of <kind>_NN.txt")
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--call-cap", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=0)
    a = ap.parse_args(argv)
    if a.mode == "dry_run":
        return dry_run(a)
    if a.mode == "replay" and not a.responses:
        raise SystemExit("--mode replay needs --responses DIR")
    if a.mode == "live":
        print(f"[case] LIVE (paid): at most {a.call_cap} VLM request(s), parse retries "
              f"included; model {_model()}, temperature {_temperature()}")
    Case(a).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
