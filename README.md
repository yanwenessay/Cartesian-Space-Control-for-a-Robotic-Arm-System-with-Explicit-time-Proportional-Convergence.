# Cartesian-Space-Control-for-a-Robotic-Arm-System-with-Explicit-time-Proportional-Convergence.
Cartesian Space Control and Joint Tracking Control for a Robotic Arm System with Explicit-time Proportional Convergence[J/OL]. IEEE/CAA Journal of Automatica Sinica, [2026-2-27]. https://doi.org/10.1109/JAS.2026.125963

---

## Overview

This repository implements the **pose-level indirect Cartesian space control model** proposed in the above paper.

The core idea is to introduce a Lyapunov control framework into the joint angular velocity planning for a robotic arm, enabling velocity-level pose-error feedback with guaranteed convergence.

### Core equation (Eq. 6)

The velocity-level pose error satisfies:

```
ε̇_e = J_m θ̇_p + ΔV − V_ed
```

| Symbol | Meaning |
|--------|---------|
| `ε̇_e` | Time derivative of end-effector pose error (`current − desired`) |
| `J_m` | Geometric Jacobian at the current joint configuration |
| `θ̇_p` | Planned joint angular velocity — the **control input** |
| `ΔV = V_e − V_ep` | End-effector spatial velocity tracking error |
| `V_ep` | Planned end-effector spatial velocity |
| `V_ed` | Desired end-effector spatial velocity |

### Lyapunov-based control law

The control input is derived by requiring `𝒱̇ ≤ 0` on the quadratic Lyapunov candidate `𝒱 = ½‖ε_e‖²`:

```
θ̇_p = J_m⁺ (V_ed − ΔV − K(t) ε_e)
```

which drives `ε̇_e = −K(t) ε_e` and guarantees asymptotic convergence.

### Explicit-time proportional convergence

With a desired convergence time `T_f` and base gain `K₀`, the time-varying gain

```
K(t) = K₀ / (T_f − t),   t ∈ [0, T_f)
```

produces the closed-form error trajectory:

```
ε_e(t) = ε_e(0) · ((T_f − t) / T_f)^{K₀}
```

so `ε_e(T_f) = 0` exactly.

---

## File Structure

```
.
├── robot_model.py          # DH-parameter FK, geometric Jacobian, pose error
├── lyapunov_controller.py  # Lyapunov-based Cartesian space controller (Eq. 6)
├── simulation.py           # Closed-loop simulation & result plots
├── requirements.txt        # Python dependencies
└── tests/
    ├── test_robot_model.py          # Unit tests for kinematics
    └── test_lyapunov_controller.py  # Unit tests for controller & convergence
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulation (explicit-time convergence, T_f = 5 s)
python simulation.py

# Run all unit tests
python -m pytest tests/ -v
```

### Simulation options

```
--T_f FLOAT          Desired convergence time in seconds (default: 5.0)
--dt FLOAT           Integration time step in seconds (default: 0.01)
--K0 FLOAT           Base control gain (default: 2.0)
--no-explicit-time   Use constant gain (exponential convergence instead)
--noise-std FLOAT    Std-dev of velocity tracking noise ΔV (default: 0)
```

---

## Simulation Results

The plots below were produced by `python simulation.py` for a UR5-style 6-DOF robot arm driven from a non-zero initial configuration to the home pose.

**Pose error norm ‖ε_e(t)‖** — converges to zero exactly at T_f = 5 s:

![Pose error convergence](https://github.com/user-attachments/assets/eb3ba269-edce-4469-8354-02c29ae28b02)

**Lyapunov function V(t) = ½‖ε_e‖²** — strictly decreasing, confirming stability:

![Lyapunov function](https://github.com/user-attachments/assets/d3410ea0-af9d-4602-a19a-8b7651be0016)

---

## Reference

> Cartesian Space Control and Joint Tracking Control for a Robotic Arm System with Explicit-time Proportional Convergence.  
> *IEEE/CAA Journal of Automatica Sinica*, 2026.  
> DOI: [10.1109/JAS.2026.125963](https://doi.org/10.1109/JAS.2026.125963)

