import os, pickle, argparse
import numpy as np
import zmq
from gsnet import create_detector

PICKLE_PROTOCOL = 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", default="/opt/anygrasp/checkpoints/checkpoint_detection.tar")
    p.add_argument("--max_gripper_width", type=float, default=0.085)  # match your gripper
    p.add_argument("--gripper_height", type=float, default=0.03)
    p.add_argument("--top_down_grasp", action="store_true")
    p.add_argument("--debug", action="store_true")
    cfgs = p.parse_args()
    cfgs.max_gripper_width = max(0.0, min(0.1, cfgs.max_gripper_width))

    detector = create_detector(cfgs)
    if detector is None:
        raise RuntimeError("create_detector failed (license validation or checkpoint issue)")

    port = os.environ.get("GRASP_PORT", "5666")
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://*:{port}")
    print(f"[grasp_server] listening on :{port}", flush=True)

    while True:
        raw = sock.recv()
        print(f"[grasp_server] recv {len(raw)} bytes", flush=True)

        # The REP socket is strict-state: it MUST send exactly one reply per
        # recv, or the next recv desyncs and every later request fails. So the
        # recv/unpickle lives inside the guard, and every failure path replies.
        try:
            req = pickle.loads(raw)
            points = req["points"].astype(np.float32)

            # lims -> region_steering mask (workspace filtering moved into the mask API)
            region_steering = req.get("region_steering")
            lims = req.get("lims")
            if region_steering is None and lims is not None:
                xmin, xmax, ymin, ymax, zmin, zmax = lims
                region_steering = (
                    (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
                    (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
                    (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
                )

            optional_params = {
                "dense_grasp": req.get("dense_grasp", False),
                "collision_detection": req.get("collision_detection", True),
                "region_steering": region_steering,
                "approach_steering": req.get("approach_steering",
                                             [0, 0, 1] if cfgs.top_down_grasp else None),
                "approach_thresh": req.get("approach_thresh",
                                           np.pi / 6 if cfgs.top_down_grasp else np.pi),
            }

            gg = detector.get_grasp(points, optional_params)

            if gg is None or len(gg) == 0:
                sock.send(pickle.dumps({"ok": True, "n": 0}, protocol=PICKLE_PROTOCOL))
                continue
            if not optional_params["dense_grasp"]:
                gg = gg.nms()
            gg = gg.sort_by_score()
            sock.send(pickle.dumps({
                "ok": True, "n": len(gg),
                "translations": gg.translations,
                "rotations": gg.rotation_matrices,
                "widths": gg.widths, "depths": gg.depths, "scores": gg.scores,
            }, protocol=PICKLE_PROTOCOL))
        except Exception as e:
            print(f"[grasp_server] bad request: {e!r}", flush=True)
            try:
                sock.send(pickle.dumps({"ok": False, "error": repr(e)}, protocol=PICKLE_PROTOCOL))
            except zmq.error.ZMQError:
                # send failed because we're not in a send-allowed state (recv
                # never completed cleanly). Reset the socket so the next
                # request isn't dead on arrival.
                sock.close(linger=0)
                sock = zmq.Context.instance().socket(zmq.REP)
                sock.bind(f"tcp://*:{port}")
                print("[grasp_server] socket reset after framing error", flush=True)

    # unreachable, but close cleanly if the loop ever exits
    sock.close(linger=0)


if __name__ == "__main__":
    main()