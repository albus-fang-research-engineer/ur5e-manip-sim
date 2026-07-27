import os, pickle
import zmq

class GraspClient:
    def __init__(self, addr=None, timeout_ms=30000):
        self.addr = addr or os.environ.get("ANYGRASP_ADDR", "tcp://grasp:5555")
        self.timeout_ms = timeout_ms
        self._connect()

    def _connect(self):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def get_grasps(self, points, colors, lims=None, **kw):
        self.sock.send(pickle.dumps({"points": points, "colors": colors, "lims": lims, **kw}))
        try:
            rep = pickle.loads(self.sock.recv())
        except zmq.error.Again:
            self._connect()  # REQ socket is dead after a timeout; rebuild
            raise RuntimeError(
                f"grasp service not responding at {self.addr} — "
                "run: docker compose --profile grasp up -d grasp")
        if not rep["ok"]:
            raise RuntimeError(f"grasp server error: {rep['error']}")
        return rep