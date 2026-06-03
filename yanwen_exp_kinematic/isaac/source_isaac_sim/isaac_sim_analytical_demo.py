#!/usr/bin/env python3
"""Isaac Sim integration example for the analytical pose-feedback planner."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source_explicit_time_kinematic'))

from planning_main import planning
from Kinematic_fcn import Kinematic
import constant as cont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--usd', required=True, help='Robot USD path')
    parser.add_argument('--prim-path', default='/World/Kinova')
    parser.add_argument('--target-position', nargs=3, type=float, default=[0.224, -0.094, 0.717])
    parser.add_argument('--target-zyx-deg', nargs=3, type=float, default=[45.830, -18.501, 164.665])
    parser.add_argument('--max-speed-deg-s', type=float, default=20.0)
    parser.add_argument('--duration', type=float, default=12.0)
    parser.add_argument('--headless', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from isaacsim import SimulationApp
    except Exception:
        from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({'headless': args.headless})

    try:
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.types import ArticulationAction
    except Exception:
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.types import ArticulationAction

    world = World(stage_units_in_meters=1.0)
    add_reference_to_stage(args.usd, args.prim_path)
    robot = Articulation(prim_paths_expr=args.prim_path, name='kinova_gen3')
    world.scene.add(robot)
    world.reset()

    target_position = np.asarray(args.target_position, dtype=float)
    target_zyx = np.deg2rad(np.asarray(args.target_zyx_deg, dtype=float))
    # Joint velocity is intentionally unlimited in this release demo.
    dt = world.get_physics_dt()
    steps = int(args.duration / dt)

    for step in range(steps):
        world.step(render=not args.headless)
        if not world.is_playing():
            continue
        q = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)[:7]
        qdot = np.asarray(planning(target_position, target_zyx, q), dtype=float)
        # No joint-velocity clipping.
        robot.apply_action(ArticulationAction(joint_velocities=qdot))
        if step % max(1, int(0.5 / dt)) == 0:
            p, phi, _, _ = Kinematic(q, cont.z_tool)
            pos_err = np.linalg.norm(p - target_position) * 1000.0
            print(f't={step*dt:6.2f}s | pos_err={pos_err:8.3f} mm | |qdot|={np.linalg.norm(qdot):.4f} rad/s')

    simulation_app.close()


if __name__ == '__main__':
    main()

