#!/usr/bin/env python3
"""Record an Isaac Lab render driven by the offline explicit-time planner trace."""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path("/home/kinova/ssd1/YanWen/yanwen_exp_kinematic/isaac")

parser = argparse.ArgumentParser(description="Record a Kinova Gen3 N7 Isaac Lab motion video from planner data.")
parser.add_argument("--trace", type=str, default=str(ROOT / "data" / "offline_planner_trace.npz"))
parser.add_argument("--output", type=str, default=str(ROOT / "videos" / "kinova_gen3_isaac_motion.mp4"))
parser.add_argument("--duration", type=float, default=None, help="Planner duration in seconds. Default uses the trace duration.")
parser.add_argument("--pre-roll", type=float, default=2.0, help="Static seconds before planner time 0. Video starts at -pre-roll.")
parser.add_argument("--fps", type=int, default=24, help="Output video FPS.")
parser.add_argument("--width", type=int, default=960, help="Camera width.")
parser.add_argument("--height", type=int, default=540, help="Camera height.")
parser.add_argument("--crf", type=int, default=31, help="ffmpeg H.264 CRF. Larger means smaller file.")
parser.add_argument("--camera-eye", type=float, nargs=3, default=(2.25, -2.55, 1.70), help="World camera position.")
parser.add_argument("--camera-target", type=float, nargs=3, default=(0.0, 0.0, 0.48), help="World point kept near the image center.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab_assets import KINOVA_GEN3_N7_CFG


def look_at_quat_wxyz(eye, target, up=(0.0, 0.0, 1.0)):
    """Return OpenGL camera quaternion where local -Z points at target."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rot = np.column_stack((right, true_up, -forward))
    trace = np.trace(rot)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * s, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(rot)))
        if i == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quat = np.array([(rot[2, 1] - rot[1, 2]) / s, 0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array([(rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array([(rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s])
    quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat)


def ensure_trace(trace_path):
    trace = Path(trace_path)
    if trace.exists():
        return trace
    cmd = [sys.executable, str(ROOT / "examples" / "offline_convergence_demo.py")]
    subprocess.run(cmd, check=True)
    if not trace.exists():
        raise FileNotFoundError(f"Planner trace was not created: {trace}")
    return trace


def load_planner_trace(trace_path):
    trace = np.load(ensure_trace(trace_path))
    time = np.asarray(trace["time"], dtype=float)
    theta = np.asarray(trace["theta_unwrapped" if "theta_unwrapped" in trace.files else "theta"], dtype=float)
    qdot = np.asarray(trace["qdot"], dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 7:
        raise ValueError(f"Expected theta shape (N, 7), got {theta.shape}")
    return time, theta, qdot


def sample_trace(time, theta, qdot, planner_t):
    if planner_t <= time[0]:
        return theta[0].copy(), np.zeros(7, dtype=float)
    planner_t = np.clip(planner_t, time[0], time[-1])
    q = np.array([np.interp(planner_t, time, theta[:, j]) for j in range(7)], dtype=float)
    dq = np.array([np.interp(planner_t, time, qdot[:, j]) for j in range(7)], dtype=float)
    return q, dq


def main():
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.stem + "_raw.mp4")

    time, theta, qdot = load_planner_trace(args_cli.trace)
    planner_duration = float(time[-1]) if args_cli.duration is None else min(float(args_cli.duration), float(time[-1]))
    pre_roll = max(0.0, float(args_cli.pre_roll))
    video_duration = pre_roll + planner_duration

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=list(args_cli.camera_eye), target=list(args_cli.camera_target))

    ground_cfg = sim_utils.GroundPlaneCfg(size=(3.0, 3.0))
    ground_cfg.func("/World/Ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.82, 0.86, 0.90))
    light_cfg.func("/World/DomeLight", light_cfg)
    prim_utils.create_prim("/World/KinovaOrigin", "Xform", translation=(0.0, 0.0, 0.0))

    robot_cfg = KINOVA_GEN3_N7_CFG.replace(prim_path="/World/KinovaOrigin/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 0.0)
    robot = Articulation(cfg=robot_cfg)

    eye = tuple(args_cli.camera_eye)
    target = tuple(args_cli.camera_target)
    camera_cfg = CameraCfg(
        prim_path="/World/RenderCamera",
        update_period=1.0 / args_cli.fps,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.2,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at_quat_wxyz(eye, target), convention="opengl"),
    )
    camera = Camera(cfg=camera_cfg)

    sim.reset()
    q0, dq0 = sample_trace(time, theta, qdot, -pre_roll)
    q0_t = torch.tensor(q0, dtype=robot.data.default_joint_pos.dtype, device=robot.data.default_joint_pos.device).unsqueeze(0)
    dq0_t = torch.tensor(dq0, dtype=robot.data.default_joint_vel.dtype, device=robot.data.default_joint_vel.device).unsqueeze(0)
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
    robot.write_joint_state_to_sim(q0_t, dq0_t)
    robot.reset()
    camera.reset()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, float(args_cli.fps), (args_cli.width, args_cli.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {raw_path}")

    sim_dt = sim.get_physics_dt()
    render_interval = max(1, round((1.0 / args_cli.fps) / sim_dt))
    steps = int(video_duration / sim_dt)
    frames = 0

    for step in range(steps + 1):
        video_t = step * sim_dt
        planner_t = video_t - pre_roll
        q, dq = sample_trace(time, theta, qdot, planner_t)
        q_t = torch.tensor(q, dtype=robot.data.default_joint_pos.dtype, device=robot.data.default_joint_pos.device).unsqueeze(0)
        dq_t = torch.tensor(dq, dtype=robot.data.default_joint_vel.dtype, device=robot.data.default_joint_vel.device).unsqueeze(0)
        robot.write_joint_state_to_sim(q_t, dq_t)
        robot.write_data_to_sim()
        sim.step(render=(step % render_interval == 0))
        robot.update(sim_dt)
        camera.update(sim_dt)

        if step % render_interval == 0 and "rgb" in camera.data.output:
            rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            frames += 1

    writer.release()

    ffmpeg = "/usr/bin/ffmpeg"
    if os.path.exists(ffmpeg):
        tmp = output_path.with_name(output_path.stem + "_h264.mp4")
        cmd = [ffmpeg, "-y", "-i", str(raw_path), "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args_cli.crf), "-movflags", "+faststart", str(tmp)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp.replace(output_path)
        raw_path.unlink(missing_ok=True)
    else:
        raw_path.replace(output_path)

    size_mb = output_path.stat().st_size / (1024.0 * 1024.0)
    print(f"Saved Isaac Kinova Gen3 video: {output_path}")
    print(f"Trace: {args_cli.trace}")
    print(f"Planner time: 0.00 to {planner_duration:.2f} s; video time: {-pre_roll:.2f} to {planner_duration:.2f} s")
    print(f"Frames: {frames}, size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
    os._exit(0)
