import os, pickle, argparse
import numpy as np
import zmq
from gsnet import AnyGrasp

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", default="/opt/anygrasp/checkpoints/checkpoint_detection.tar")
    p.add_argument("--max_gripper_width", type=float, default=0.085)  # match your gripper
    p.add_argument("--gripper_height", type=float, default=0.03)
    p.add_argument("--top_down_grasp", action="store_true")
    p.add_argument("--debug", action="store_true")
    cfgs = p.parse_args()
    cfgs.max_gripper_width = max(0.0, min(0.1, cfgs.max_gripper_width))

    ag = AnyGrasp(cfgs)
    ag.load_net()

    port = os.environ.get("GRASP_PORT", "5555")
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://*:{port}")
    print(f"[grasp_server] listening on :{port}", flush=True)

    while True:
        req = pickle.loads(sock.recv())
        try:
            gg, cloud = ag.get_grasp(
                req["points"].astype(np.float32),
                req["colors"].astype(np.float32),
                lims=req.get("lims"),
                apply_object_mask=req.get("apply_object_mask", True),
                dense_grasp=req.get("dense_grasp", False),
                collision_detection=req.get("collision_detection", True),
            )
            if gg is None or len(gg) == 0:
                sock.send(pickle.dumps({"ok": True, "n": 0}))
                continue
            gg = gg.nms().sort_by_score()
            sock.send(pickle.dumps({
                "ok": True, "n": len(gg),
                "translations": gg.translations,
                "rotations": gg.rotation_matrices,
                "widths": gg.widths, "depths": gg.depths, "scores": gg.scores,
            }))
        except Exception as e:
            sock.send(pickle.dumps({"ok": False, "error": repr(e)}))

if __name__ == "__main__":
    main()