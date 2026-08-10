"""ZMQ client for the PointSO sidecar (ur5e-manip-hardware, port 5668).

Same shape as GraspClient, but the pointso server speaks msgpack +
msgpack_numpy, not pickle. Point clouds are Nx6 float32 (xyz metric,
rgb in [0,1]); predictions are unit vectors expressed in the SAME frame
the points were sent in — feed canonical/body-frame mesh points, get a
body-frame semantic direction; rotate the cloud by the body pose first if
you want the answer in world frame.
"""

import os

import numpy as np
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()


class PointSOClient:
    def __init__(self, addr=None, timeout_ms=60000):
        self.addr = addr or os.environ.get("POINTSO_ADDR", "tcp://pointso:5668")
        self.timeout_ms = timeout_ms
        self._connect()

    def _connect(self):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def _request(self, payload):
        self.sock.send(msgpack.packb(payload, use_bin_type=True))
        try:
            rep = msgpack.unpackb(self.sock.recv(), raw=False)
        except zmq.error.Again:
            self._connect()  # REQ socket is dead after a timeout; rebuild
            raise RuntimeError(
                f"pointso service not responding at {self.addr} — in "
                "ur5e-manip-hardware run: docker compose up -d pointso")
        if not rep.get("ok"):
            raise RuntimeError(f"pointso server error: {rep.get('error')}")
        return rep

    def ping(self):
        return self._request({"cmd": "ping"})["ok"]

    def orient(self, pcd, instruction):
        """pcd: Nx6 float32. Returns a (3,) unit vector."""
        pcd = np.ascontiguousarray(pcd, np.float32)
        assert pcd.ndim == 2 and pcd.shape[1] == 6, "pcd must be Nx6 (xyz+rgb)"
        rep = self._request(
            {"cmd": "orient", "pcd": pcd, "instruction": instruction})
        return np.asarray(rep["direction"], np.float32).reshape(3)

    def orient_batch(self, pcd, instructions):
        """pcd: Nx6 float32. Returns (M, 3) unit vectors."""
        pcd = np.ascontiguousarray(pcd, np.float32)
        assert pcd.ndim == 2 and pcd.shape[1] == 6, "pcd must be Nx6 (xyz+rgb)"
        rep = self._request(
            {"cmd": "orient_batch", "pcd": pcd,
             "instructions": list(instructions)})
        return np.asarray(rep["directions"], np.float32).reshape(-1, 3)
