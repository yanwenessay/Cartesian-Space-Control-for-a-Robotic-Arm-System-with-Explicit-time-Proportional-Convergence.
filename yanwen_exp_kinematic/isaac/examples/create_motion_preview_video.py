#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source_explicit_time_kinematic'))

from planning_main import planning
from Kinematic_fcn import Ttrans7, Kinematic
import constant as cont


def tool_transform(z_tool):
    return np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0, -1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0,  z_tool],
        [0.0,  0.0,  0.0,  1.0],
    ], dtype=float)


def joint_positions(theta):
    t_0_i, t_0_7 = Ttrans7(theta)
    pts = [np.zeros(3)]
    for i in range(7):
        pts.append(t_0_i[i, :3, 3].copy())
    pts.append((t_0_7 @ tool_transform(cont.z_tool))[:3, 3].copy())
    return np.asarray(pts)


def main():
    out_dir = ROOT / 'videos'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'kinova_motion_preview.mp4'

    theta = np.deg2rad(np.array([10, -20, 10, 120, 60, 10, 70], dtype=float))
    target_position = np.array([0.224, -0.094, 0.717], dtype=float)
    target_zyx = np.deg2rad(np.array([45.830, -18.501, 164.665], dtype=float))
    dt = 0.01
    duration = 10.0

    frames = []
    pos_err = []
    att_err = []
    for k in range(int(duration / dt)):
        qdot = np.asarray(planning(target_position, target_zyx, theta), dtype=float)
        theta = np.arctan2(np.sin(theta + qdot * dt), np.cos(theta + qdot * dt))
        if k % 5 == 0:
            frames.append(joint_positions(theta))
            p, phi, _, _ = Kinematic(theta, cont.z_tool)
            r_cur = R.from_euler('ZYX', phi).as_matrix()
            r_des = R.from_euler('ZYX', target_zyx).as_matrix()
            rot_err = R.from_matrix(r_cur @ r_des.T).as_rotvec()
            pos_err.append(np.linalg.norm(p - target_position) * 1000.0)
            att_err.append(np.linalg.norm(rot_err) * 180.0 / np.pi)

    all_pts = np.vstack(frames + [target_position.reshape(1, 3)])
    center = all_pts.mean(axis=0)
    radius = max(0.45, np.max(np.linalg.norm(all_pts - center, axis=1)) * 1.15)

    fig = plt.figure(figsize=(6.4, 4.8), dpi=110)
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], '-o', lw=3, markersize=4, color='#1f77b4')
    target = ax.scatter([target_position[0]], [target_position[1]], [target_position[2]], s=50, color='red', label='target')
    txt = ax.text2D(0.03, 0.94, '', transform=ax.transAxes)
    ax.legend(loc='upper right')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=24, azim=-58)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(-0.05, center[2] - radius), center[2] + radius)
    ax.set_title('Kinova Gen3 analytical pose-feedback motion preview')

    def update(i):
        pts = frames[i]
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        txt.set_text(f't={i*5*dt:4.2f}s  pos={pos_err[i]:.3f} mm  att={att_err[i]:.3f} deg')
        return line, txt

    ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    writer = FFMpegWriter(fps=20, codec='libx264', bitrate=650, extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    ani.save(out_path, writer=writer)
    plt.close(fig)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f'Saved {out_path} ({size_mb:.2f} MB)')


if __name__ == '__main__':
    main()

