#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime web monitor for Kinova Gen3 end-effector pose.

The pose uses the same calibrated forward-kinematics convention as this folder:
- joint feedback from BaseCyclic.RefreshFeedback()
- Kinematic_fcn.Kinematic(theta_rad, Z_TOOL)
- fixed ZYX Euler angles returned as [Z, Y, X]

Open the page in a browser, click the copy box, then press Ctrl+C.
The JSON API is available at /api/pose for external tools.
"""

import argparse
import errno
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from Kinematic_fcn import Kinematic

import utilities

Z_TOOL = -0.16746
DEFAULT_POLL_HZ = 20.0
DEFAULT_WEB_HOST = "localhost"
DEFAULT_WEB_PORT = 8088


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class PoseState:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "ok": False,
            "message": "waiting for first robot feedback",
            "timestamp": None,
        }

    def set(self, data):
        with self._lock:
            self._data = data

    def get(self):
        with self._lock:
            return dict(self._data)


class PosePoller(threading.Thread):
    def __init__(self, base_cyclic, actuator_count, state, poll_hz):
        super().__init__(daemon=True)
        self.base_cyclic = base_cyclic
        self.actuator_count = actuator_count
        self.state = state
        self.period = 1.0 / max(float(poll_hz), 0.1)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                fb = self.base_cyclic.RefreshFeedback()
                joints_deg = [float(fb.actuators[i].position) for i in range(self.actuator_count)]
                theta = np.deg2rad(np.array(joints_deg, dtype=float))
                position_m, euler_zyx_rad, quaternion_wxyz, _ = Kinematic(theta, Z_TOOL)

                euler_zyx_deg = np.rad2deg(euler_zyx_rad)
                payload = {
                    "ok": True,
                    "message": "ok",
                    "timestamp": time.time(),
                    "time_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "position_m": {
                        "x": float(position_m[0]),
                        "y": float(position_m[1]),
                        "z": float(position_m[2]),
                    },
                    "euler_zyx_deg": {
                        "z": float(euler_zyx_deg[0]),
                        "y": float(euler_zyx_deg[1]),
                        "x": float(euler_zyx_deg[2]),
                    },
                    "euler_zyx_rad": {
                        "z": float(euler_zyx_rad[0]),
                        "y": float(euler_zyx_rad[1]),
                        "x": float(euler_zyx_rad[2]),
                    },
                    "quaternion_wxyz": {
                        "w": float(quaternion_wxyz[0]),
                        "x": float(quaternion_wxyz[1]),
                        "y": float(quaternion_wxyz[2]),
                        "z": float(quaternion_wxyz[3]),
                    },
                    "joints_deg": joints_deg,
                    "joints_rad": [float(v) for v in theta],
                    "copy_text": format_copy_text(position_m, euler_zyx_deg, quaternion_wxyz),
                    "csv_header": "timestamp,x_m,y_m,z_m,euler_z_deg,euler_y_deg,euler_x_deg,q_w,q_x,q_y,q_z",
                }
                self.state.set(payload)
            except Exception as exc:
                self.state.set({
                    "ok": False,
                    "message": str(exc),
                    "timestamp": time.time(),
                    "time_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                })

            elapsed = time.perf_counter() - started
            self._stop_event.wait(max(0.0, self.period - elapsed))


def format_copy_text(position_m, euler_zyx_deg, quaternion_wxyz):
    values = [
        f"x={position_m[0]:.9f}",
        f"y={position_m[1]:.9f}",
        f"z={position_m[2]:.9f}",
        f"euler_Z={euler_zyx_deg[0]:.6f}deg",
        f"euler_Y={euler_zyx_deg[1]:.6f}deg",
        f"euler_X={euler_zyx_deg[2]:.6f}deg",
        f"qw={quaternion_wxyz[0]:.9f}",
        f"qx={quaternion_wxyz[1]:.9f}",
        f"qy={quaternion_wxyz[2]:.9f}",
        f"qz={quaternion_wxyz[3]:.9f}",
    ]
    return ", ".join(values)


def format_csv_line(data):
    if not data.get("ok"):
        return ""
    p = data["position_m"]
    e = data["euler_zyx_deg"]
    q = data["quaternion_wxyz"]
    return ",".join([
        str(data.get("timestamp", "")),
        f"{p['x']:.9f}", f"{p['y']:.9f}", f"{p['z']:.9f}",
        f"{e['z']:.6f}", f"{e['y']:.6f}", f"{e['x']:.6f}",
        f"{q['w']:.9f}", f"{q['x']:.9f}", f"{q['y']:.9f}", f"{q['z']:.9f}",
    ])


def make_handler(state):
    class PoseHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[{time.strftime('%H:%M:%S')}] {self.client_address[0]} {fmt % args}")

        def send_json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_html(self):
            body = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            path = urlparse(self.path).path
            data = state.get()
            if path == "/" or path == "/index.html":
                self.send_html()
                return
            if path == "/api/pose":
                data["csv_line"] = format_csv_line(data)
                self.send_json(data)
                return
            if path == "/api/text":
                text = data.get("copy_text", "") if data.get("ok") else data.get("message", "no data")
                body = (text + "\n").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_json({"ok": False, "message": "not found"}, status=404)

    return PoseHandler


PAGE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Kinova 末端实时位姿</title>
  <style>
    :root { color-scheme: light; --ink:#13211b; --muted:#5f6f66; --line:#d9e2dc; --card:#fffdf7; --accent:#1c7c54; --bg1:#edf6ef; --bg2:#fff7e3; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; color:var(--ink); background: radial-gradient(circle at 20% 10%, #d8f1df 0, transparent 30%), linear-gradient(135deg, var(--bg1), var(--bg2)); }
    main { width:min(1080px, 94vw); margin:0 auto; padding:34px 0 42px; }
    .hero { display:flex; justify-content:space-between; gap:18px; align-items:flex-end; margin-bottom:20px; }
    h1 { margin:0; font-size:clamp(28px, 4vw, 48px); letter-spacing:-0.04em; }
    .sub { margin:8px 0 0; color:var(--muted); }
    .pill { padding:8px 12px; border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.65); font-weight:700; white-space:nowrap; }
    .grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:14px; }
    .card { background:rgba(255,253,247,.86); border:1px solid var(--line); border-radius:22px; padding:18px; box-shadow:0 18px 50px rgba(26,50,35,.10); backdrop-filter: blur(8px); }
    .wide { grid-column:1 / -1; }
    .label { color:var(--muted); font-size:13px; margin-bottom:8px; }
    .num { font-size:clamp(24px, 4vw, 42px); font-weight:800; letter-spacing:-0.04em; font-variant-numeric: tabular-nums; }
    textarea { width:100%; min-height:92px; resize:vertical; border:1px solid var(--line); border-radius:16px; padding:14px; font:16px/1.55 Consolas, "Cascadia Mono", monospace; color:var(--ink); background:#fffefb; outline:none; }
    textarea:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(28,124,84,.14); }
    button { border:0; border-radius:14px; padding:12px 16px; margin-top:10px; background:var(--accent); color:white; font-weight:800; cursor:pointer; }
    pre { overflow:auto; margin:0; font:14px/1.55 Consolas, "Cascadia Mono", monospace; color:#203129; }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; justify-content:space-between; }
    .hint { color:var(--muted); font-size:13px; }
    @media (max-width: 760px) { .hero { display:block; } .pill { display:inline-block; margin-top:12px; } .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <h1>Kinova 末端实时位姿</h1>
      <p class="sub">采用当前目录的标定正运动学：关节反馈 → <code>Kinematic(theta, z_tool)</code> → 固定 ZYX 欧拉角。</p>
    </div>
    <div id="status" class="pill">连接中...</div>
  </section>

  <section class="grid">
    <div class="card"><div class="label">X / m</div><div class="num" id="x">--</div></div>
    <div class="card"><div class="label">Y / m</div><div class="num" id="y">--</div></div>
    <div class="card"><div class="label">Z / m</div><div class="num" id="z">--</div></div>
    <div class="card"><div class="label">Euler Z / deg</div><div class="num" id="ez">--</div></div>
    <div class="card"><div class="label">Euler Y / deg</div><div class="num" id="ey">--</div></div>
    <div class="card"><div class="label">Euler X / deg</div><div class="num" id="ex">--</div></div>

    <div class="card wide">
      <div class="row"><div class="label">点击文本框自动全选，然后 Ctrl+C 复制</div><div class="hint" id="updated">--</div></div>
      <textarea id="copyBox" readonly spellcheck="false">等待数据...</textarea>
      <button id="copyBtn">复制当前位姿</button>
      <span class="hint" id="copyHint"></span>
    </div>

    <div class="card wide">
      <div class="label">/api/pose JSON</div>
      <pre id="jsonBox">{}</pre>
    </div>
  </section>
</main>
<script>
const ids = Object.fromEntries(['status','x','y','z','ez','ey','ex','copyBox','jsonBox','updated','copyBtn','copyHint'].map(id => [id, document.getElementById(id)]));
function fmt(v, n) { return Number.isFinite(v) ? v.toFixed(n) : '--'; }
async function refresh() {
  try {
    const r = await fetch('/api/pose?ts=' + Date.now());
    const data = await r.json();
    ids.jsonBox.textContent = JSON.stringify(data, null, 2);
    if (!data.ok) throw new Error(data.message || 'no data');
    ids.status.textContent = '实时读取中';
    ids.status.style.color = '#17633f';
    ids.x.textContent = fmt(data.position_m.x, 6);
    ids.y.textContent = fmt(data.position_m.y, 6);
    ids.z.textContent = fmt(data.position_m.z, 6);
    ids.ez.textContent = fmt(data.euler_zyx_deg.z, 3);
    ids.ey.textContent = fmt(data.euler_zyx_deg.y, 3);
    ids.ex.textContent = fmt(data.euler_zyx_deg.x, 3);
    ids.copyBox.value = data.copy_text + '\n' + data.csv_header + '\n' + data.csv_line;
    ids.updated.textContent = '更新: ' + data.time_iso;
  } catch (err) {
    ids.status.textContent = '读取失败: ' + err.message;
    ids.status.style.color = '#9b2c2c';
  }
}
ids.copyBox.addEventListener('focus', () => ids.copyBox.select());
ids.copyBox.addEventListener('click', () => ids.copyBox.select());
ids.copyBtn.addEventListener('click', async () => {
  ids.copyBox.select();
  try { await navigator.clipboard.writeText(ids.copyBox.value); ids.copyHint.textContent = '已复制'; }
  catch { document.execCommand('copy'); ids.copyHint.textContent = '已选中，可按 Ctrl+C'; }
});
refresh();
setInterval(refresh, 200);
</script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime Kinova end-effector pose web monitor")
    parser.add_argument("--ip", type=str, required=True, help="Robot IP address. Use your own Kinova Gen3 IP.")
    parser.add_argument("-u", "--username", type=str, required=True, help="Robot username")
    parser.add_argument("-p", "--password", type=str, required=True, help="Robot password")
    parser.add_argument("--web-host", type=str, default=DEFAULT_WEB_HOST, help="Web bind host")
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT, help="Web bind port")
    parser.add_argument("--poll-hz", type=float, default=DEFAULT_POLL_HZ, help="Robot feedback polling frequency")
    args = parser.parse_args()
    if args.ip in {"<ROBOT_IP>", "<YOUR_ROBOT_IP>", "YOUR_ROBOT_IP"}:
        parser.error("Please replace the robot IP placeholder with your own Kinova Gen3 IP.")
    return args


def main():
    args = parse_args()
    state = PoseState()
    server = None
    poller = None

    try:
        server = ReusableThreadingHTTPServer((args.web_host, args.web_port), make_handler(state))
    except OSError as exc:
        if exc.errno in (errno.EADDRINUSE, 10048):
            print(f"端口 {args.web_port} 已被占用，可能已有 realtime_pose_web.py 在运行。")
            print('请先停止旧服务：pkill -f realtime_pose_web.py')
            print(f"或者换一个端口：--web-port {args.web_port + 1}")
            raise SystemExit(1) from exc
        raise

    conn_args = argparse.Namespace(ip=args.ip, username=args.username, password=args.password)
    try:
        with utilities.DeviceConnection.createTcpConnection(conn_args) as router_tcp:
            with utilities.DeviceConnection.createUdpConnection(conn_args) as router_udp:
                base = BaseClient(router_tcp)
                base_cyclic = BaseCyclicClient(router_udp)
                actuator_count = base.GetActuatorCount().count
                if actuator_count != 7:
                    raise RuntimeError(f"需要7轴机械臂，当前为 {actuator_count} 轴")

                poller = PosePoller(base_cyclic, actuator_count, state, args.poll_hz)
                poller.start()

                print(f"Web pose monitor: http://{args.web_host}:{args.web_port}")
                print("API endpoint: /api/pose")
                print("Press Ctrl+C to stop.")
                server.serve_forever()
    finally:
        if poller is not None:
            poller.stop()
        if server is not None:
            server.server_close()


if __name__ == "__main__":
    main()
