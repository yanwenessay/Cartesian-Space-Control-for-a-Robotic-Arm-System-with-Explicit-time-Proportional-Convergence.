#!/usr/bin/env python3
"""Offline explicit-time pose-feedback demo and plot generator."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source_explicit_time_kinematic"))

from planning_main import planning
from Kinematic_fcn import Kinematic
from inverse_kinematics import inverse_kinematics
import constant as cont


def wrap_to_pi(x):
    return np.arctan2(np.sin(x), np.cos(x))


def geometric_singular_values(theta):
    _, s = inverse_kinematics(theta, np.zeros(6), cont.z_tool)
    return s


def equalize_3d_axes(ax, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 0.05)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_wrapped_joint_angles(ax, t, theta_rad):
    theta_deg = np.rad2deg(wrap_to_pi(theta_rad))
    for j in range(theta_deg.shape[1]):
        y = theta_deg[:, j].copy()
        jumps = np.abs(np.diff(y)) > 180.0
        y[1:][jumps] = np.nan
        ax.plot(t, y, label=f"J{j + 1}")
    ax.set_ylim(-185.0, 185.0)
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.set_ylabel("Normalized joint angle (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)


def active_target(t, switch_time, targets):
    if t < switch_time:
        return targets[0]
    return targets[1]


def main():
    fig_dir = ROOT / "figures"
    data_dir = ROOT / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    theta = np.deg2rad(np.array([10, -20, 10, 120, 60, 10, 70], dtype=float))
    theta0 = theta.copy()
    targets = [
        {
            "name": "target 1",
            "position": np.array([0.224, -0.094, 0.717], dtype=float),
            "zyx": np.deg2rad(np.array([45.830, -18.501, 164.665], dtype=float)),
        },
        {
            "name": "target 2",
            "position": np.array([0.258, -0.066, 0.694], dtype=float),
            "zyx": np.deg2rad(np.array([50.830, -21.501, 171.665], dtype=float)),
        },
    ]
    dt = 0.01
    duration_s = 10.0
    switch_time_s = 5.0

    time_hist = []
    active_target_index = []
    pos_err_norm = []
    att_err_norm = []
    pos_err_vec = []
    att_err_vec = []
    theta_hist = []
    qdot_hist = []
    s_hist = []
    pos_hist = []
    zyx_hist = []

    n_steps = int(duration_s / dt)
    for k in range(n_steps + 1):
        t_now = k * dt
        target_index = 0 if t_now < switch_time_s else 1
        target = targets[target_index]
        target_position = target["position"]
        target_zyx = target["zyx"]

        p, phi, _, _ = Kinematic(theta, cont.z_tool)
        r_cur = R.from_euler("ZYX", phi).as_matrix()
        r_des = R.from_euler("ZYX", target_zyx).as_matrix()
        rot_err = R.from_matrix(r_cur @ r_des.T).as_rotvec()

        time_hist.append(t_now)
        active_target_index.append(target_index)
        pos_hist.append(p.copy())
        zyx_hist.append(phi.copy())
        theta_hist.append(theta.copy())
        pos_err = p - target_position
        pos_err_vec.append(pos_err.copy())
        att_err_vec.append(rot_err.copy())
        pos_err_norm.append(np.linalg.norm(pos_err) * 1000.0)
        att_err_norm.append(np.linalg.norm(rot_err) * 180.0 / np.pi)
        s_hist.append(geometric_singular_values(theta))

        qdot = np.asarray(planning(target_position, target_zyx, theta), dtype=float)
        qdot_hist.append(qdot.copy())
        if k < n_steps:
            theta = wrap_to_pi(theta + qdot * dt)

    t = np.asarray(time_hist)
    theta_hist = np.asarray(theta_hist)
    theta_unwrapped = np.unwrap(theta_hist, axis=0)
    qdot_hist = np.asarray(qdot_hist)
    pos_hist = np.asarray(pos_hist)
    zyx_hist = np.asarray(zyx_hist)
    pos_err_vec = np.asarray(pos_err_vec)
    att_err_vec = np.asarray(att_err_vec)
    s_hist = np.asarray(s_hist)
    active_target_index = np.asarray(active_target_index)
    target_positions = np.vstack([x["position"] for x in targets])
    target_zyx = np.vstack([x["zyx"] for x in targets])

    np.savez(
        data_dir / "offline_planner_trace.npz",
        time=t,
        theta=theta_hist,
        theta_unwrapped=theta_unwrapped,
        qdot=qdot_hist,
        position=pos_hist,
        zyx=zyx_hist,
        target_positions=target_positions,
        target_zyx=target_zyx,
        active_target_index=active_target_index,
        theta0=theta0,
        pos_error=pos_err_vec,
        att_error_rotvec=att_err_vec,
        singular_values=s_hist,
        dt=np.array(dt),
        duration_s=np.array(duration_s),
        switch_time_s=np.array(switch_time_s),
        pre_roll_s=np.array(2.0),
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, pos_err_norm, lw=2)
    axes[0].set_ylabel("Position error to active target (mm)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, att_err_norm, lw=2, color="tab:orange")
    axes[1].set_ylabel("Attitude error to active target (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    for ax in axes:
        ax.axvline(switch_time_s, color="0.25", ls="--", lw=1.2, alpha=0.8)
    fig.suptitle("Two-stage analytical pose-feedback convergence")
    fig.savefig(fig_dir / "pose_error_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    labels_pos = ["ex", "ey", "ez"]
    labels_att = ["erx", "ery", "erz"]
    for i in range(3):
        axes[0].plot(t, pos_err_vec[:, i] * 1000.0, label=labels_pos[i])
        axes[1].plot(t, att_err_vec[:, i] * 180.0 / np.pi, label=labels_att[i])
    axes[0].set_ylabel("Position error component (mm)")
    axes[1].set_ylabel("Attitude error component (deg)")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.axvline(switch_time_s, color="0.25", ls="--", lw=1.2, alpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=8)
    fig.savefig(fig_dir / "pose_error_components.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for j in range(7):
        ax.plot(t, np.rad2deg(qdot_hist[:, j]), label=f"J{j + 1}")
    ax.axvline(switch_time_s, color="0.25", ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint velocity (deg/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.savefig(fig_dir / "joint_velocity_commands.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_wrapped_joint_angles(ax, t, theta_hist)
    ax.axvline(switch_time_s, color="0.25", ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("Time (s)")
    fig.savefig(fig_dir / "joint_angle_history.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i in range(s_hist.shape[1]):
        ax.plot(t, s_hist[:, i], alpha=0.55, lw=1.2)
    ax.plot(t, np.min(s_hist, axis=1), lw=2.4, color="black", label="min")
    ax.axvline(switch_time_s, color="0.25", ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Singular value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(fig_dir / "analytical_jacobian_singular_values.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2], lw=2, label="EE path")
    ax.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], s=65, c=["red", "tab:green"], label="targets")
    ax.text(*target_positions[0], "  target 1", color="red")
    ax.text(*target_positions[1], "  target 2", color="tab:green")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    equalize_3d_axes(ax, np.vstack([pos_hist, target_positions]))
    fig.savefig(fig_dir / "end_effector_path_3d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    dist = np.linalg.norm(target_positions[1] - target_positions[0])
    dori = np.linalg.norm(R.from_matrix(R.from_euler("ZYX", target_zyx[1]).as_matrix() @ R.from_euler("ZYX", target_zyx[0]).as_matrix().T).as_rotvec())
    print(f"Saved figures to {fig_dir}")
    print(f"Saved trajectory data to {data_dir / 'offline_planner_trace.npz'}")
    print(f"Target switch time: {switch_time_s:.2f} s")
    print(f"Target distance: {dist * 1000.0:.2f} mm")
    print(f"Target attitude distance: {np.rad2deg(dori):.2f} deg")
    print(f"Final position error: {pos_err_norm[-1]:.6f} mm")
    print(f"Final attitude error: {att_err_norm[-1]:.6f} deg")
    print(f"Steps: {len(t)}")


if __name__ == "__main__":
    main()
