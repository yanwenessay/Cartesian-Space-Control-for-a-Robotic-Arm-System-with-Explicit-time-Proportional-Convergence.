"""
Closed-loop simulation of the Lyapunov-based Cartesian space controller.

Demonstrates pose-error convergence for a UR5-style 6-DOF robotic arm
under the pose-level indirect Cartesian space control model (Eq. 6):

    ε̇_e = J_m θ̇_p + ΔV − V_ed

Usage
-----
Run directly::

    python simulation.py

or call :func:`run_simulation` from another script.

The script produces two figures saved to ``results/``:

* ``pose_error_norm.png``   – ‖εₑ(t)‖ vs. time
* ``lyapunov_function.png`` – 𝒱(t) = ½‖εₑ‖² vs. time
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display required)
import matplotlib.pyplot as plt

from robot_model import build_ur5_model
from lyapunov_controller import CartesianSpaceController


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

def run_simulation(
    T_f=5.0,
    dt=0.01,
    K0=2.0,
    theta_init=None,
    theta_desired=None,
    explicit_time=True,
    delta_V_noise_std=0.0,
):
    """
    Simulate the closed-loop Lyapunov Cartesian space controller.

    The simulation integrates the joint angles forward in time using the
    Euler method.  At each step the controller computes joint velocities
    ``θ̇_p`` that drive the pose error ``εₑ`` to zero.

    Parameters
    ----------
    T_f : float
        Desired convergence time [s] (used only when ``explicit_time=True``).
    dt : float
        Integration time step [s].
    K0 : float
        Base control-gain scalar.
    theta_init : array-like, shape (6,) or None
        Initial joint angles [rad].  Defaults to a non-zero configuration.
    theta_desired : array-like, shape (6,) or None
        Desired joint angles [rad].  Defaults to the home configuration
        ``[0, -π/2, 0, -π/2, 0, 0]``.
    explicit_time : bool
        If ``True`` (default), use the time-varying gain
        ``K(t) = K₀ / (T_f − t)`` for explicit-time convergence.
        If ``False``, use the constant gain ``K₀``.
    delta_V_noise_std : float
        Standard deviation of Gaussian noise added to ``ΔV`` (velocity
        tracking error) to simulate real-world imperfect tracking.

    Returns
    -------
    results : dict
        ``'time'``           – 1-D array of time steps
        ``'epsilon_norm'``   – ‖εₑ(t)‖  at each step
        ``'lyapunov'``       – 𝒱(t)     at each step
        ``'lyapunov_dot'``   – 𝒱̇(t)    at each step
        ``'theta_history'``  – joint angles at each step (shape n_steps × 6)
    """
    robot = build_ur5_model()

    # Default configurations
    if theta_init is None:
        theta_init = np.array([0.3, -1.2, 0.8, -1.0, 0.5, 0.2])
    if theta_desired is None:
        theta_desired = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])

    theta_init = np.asarray(theta_init, dtype=float)
    theta_desired = np.asarray(theta_desired, dtype=float)

    # Build controller
    T_f_ctrl = T_f if explicit_time else None
    controller = CartesianSpaceController(robot, K=K0, T_f=T_f_ctrl, damping=1e-3)

    # Pre-compute desired pose
    T_desired = robot.forward_kinematics(theta_desired)
    # Desired end-effector velocity is zero (static target)
    V_ed = np.zeros(6)

    # Simulation state
    theta = theta_init.copy()
    sim_time = T_f + dt  # run slightly past T_f to show post-convergence behaviour
    n_steps = int(np.ceil(sim_time / dt))

    # Storage
    time_arr = np.zeros(n_steps)
    epsilon_norm_arr = np.zeros(n_steps)
    lyapunov_arr = np.zeros(n_steps)
    lyapunov_dot_arr = np.zeros(n_steps)
    theta_history = np.zeros((n_steps, robot.n_joints))

    rng = np.random.default_rng(seed=0)

    for k in range(n_steps):
        t = k * dt

        # Current end-effector pose and pose error
        T_current = robot.forward_kinematics(theta)
        epsilon_e = robot.pose_error(T_current, T_desired)

        # Optional velocity tracking noise (ΔV ≈ 0 in ideal conditions)
        delta_V = np.zeros(6)
        if delta_V_noise_std > 0.0:
            delta_V = rng.normal(0.0, delta_V_noise_std, size=6)

        # Control law → planned joint velocities
        theta_dot_p, J_m = controller.compute_joint_velocity(
            theta, epsilon_e, V_ed, delta_V=delta_V, t=t
        )

        # Stability diagnostics
        V_val = controller.lyapunov_value(epsilon_e)
        V_dot = controller.lyapunov_derivative(
            epsilon_e, theta_dot_p, J_m, delta_V, V_ed
        )

        # Store
        time_arr[k] = t
        epsilon_norm_arr[k] = np.linalg.norm(epsilon_e)
        lyapunov_arr[k] = V_val
        lyapunov_dot_arr[k] = V_dot
        theta_history[k] = theta.copy()

        # Euler integration: θ(k+1) = θ(k) + dt · θ̇_p(k)
        theta = theta + dt * theta_dot_p

    return {
        "time": time_arr,
        "epsilon_norm": epsilon_norm_arr,
        "lyapunov": lyapunov_arr,
        "lyapunov_dot": lyapunov_dot_arr,
        "theta_history": theta_history,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ensure_results_dir():
    os.makedirs("results", exist_ok=True)


def plot_results(results, T_f, explicit_time, output_dir="results"):
    """
    Generate and save diagnostic plots.

    Parameters
    ----------
    results : dict
        Output of :func:`run_simulation`.
    T_f : float
        Convergence time [s] (used to annotate the plot).
    explicit_time : bool
        Whether explicit-time mode was active.
    output_dir : str
        Directory where PNG files are written.
    """
    os.makedirs(output_dir, exist_ok=True)

    time = results["time"]
    label_suffix = f"(T_f={T_f}s)" if explicit_time else "(exponential)"

    # --- Pose error norm ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, results["epsilon_norm"], linewidth=2, label=f"||eps_e(t)|| {label_suffix}")
    if explicit_time:
        ax.axvline(T_f, color="red", linestyle="--", linewidth=1, label=f"T_f = {T_f}s")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pose error norm")
    ax.set_title("End-effector Pose Error Convergence")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pose_error_norm.png"), dpi=150)
    plt.close(fig)

    # --- Lyapunov function ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, results["lyapunov"], linewidth=2, color="tab:orange",
            label=f"V(t) = 0.5*||eps_e||^2 {label_suffix}")
    if explicit_time:
        ax.axvline(T_f, color="red", linestyle="--", linewidth=1, label=f"T_f = {T_f}s")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Lyapunov function V")
    ax.set_title("Lyapunov Function (Stability Indicator)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lyapunov_function.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to '{output_dir}/'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate the Lyapunov Cartesian space controller."
    )
    parser.add_argument("--T_f",       type=float, default=5.0,
                        help="Convergence time [s] (default: 5.0)")
    parser.add_argument("--dt",        type=float, default=0.01,
                        help="Integration time step [s] (default: 0.01)")
    parser.add_argument("--K0",        type=float, default=2.0,
                        help="Base control gain (default: 2.0)")
    parser.add_argument("--no-explicit-time", action="store_true",
                        help="Use constant gain (exponential convergence)")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Std-dev of velocity tracking noise ΔV (default: 0)")
    args = parser.parse_args()

    explicit_time = not args.no_explicit_time

    print("=" * 60)
    print("Lyapunov Cartesian Space Controller – Simulation")
    print("=" * 60)
    print(f"  T_f          = {args.T_f} s")
    print(f"  dt           = {args.dt} s")
    print(f"  K0           = {args.K0}")
    print(f"  Explicit-time= {explicit_time}")
    print(f"  ΔV noise std = {args.noise_std}")
    print("-" * 60)

    results = run_simulation(
        T_f=args.T_f,
        dt=args.dt,
        K0=args.K0,
        explicit_time=explicit_time,
        delta_V_noise_std=args.noise_std,
    )

    t_final = results["time"][-1]
    e_final = results["epsilon_norm"][-1]
    print(f"  Simulation complete.  t_final={t_final:.3f}s  ‖εₑ‖_final={e_final:.6f}")

    plot_results(results, T_f=args.T_f, explicit_time=explicit_time)
