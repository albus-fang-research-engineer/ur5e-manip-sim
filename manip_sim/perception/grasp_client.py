import os, pickle
import zmq

# Pin protocol 4 so a modern sim wing (default proto 5) can't emit frames an
# older AnyGrasp interpreter chokes on. 4 is readable by Python >= 3.4.
PICKLE_PROTOCOL = 4


class GraspClient:
    def __init__(self, addr=None, timeout_ms=30000):
        self.addr = addr or os.environ.get("ANYGRASP_ADDR", "tcp://grasp:5666")
        self.timeout_ms = timeout_ms
        self._connect()

    def _connect(self):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def get_grasps(self, points, colors, lims=None, **kw):
        blob = pickle.dumps(
            {"points": points, "colors": colors, "lims": lims, **kw},
            protocol=PICKLE_PROTOCOL,
        )
        print(f"[client] sending {len(blob)} bytes", flush=True)
        try:
            self.sock.send(blob)
            raw = self.sock.recv()
        except zmq.error.Again:
            self._connect()  # REQ socket is dead after a timeout; rebuild
            raise RuntimeError(
                f"grasp service not responding at {self.addr} — it may have "
                "crashed on the request. Check: docker compose --profile grasp "
                "logs grasp")
        rep = pickle.loads(raw)
        if not rep["ok"]:
            raise RuntimeError(f"grasp server error: {rep['error']}")
        return rep