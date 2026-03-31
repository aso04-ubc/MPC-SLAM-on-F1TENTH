# MPC Interface Spec

## Inputs

- `/odom` (`nav_msgs/msg/Odometry`)
  - Required.
  - Used for position, heading, longitudinal speed, and steering estimation from yaw rate.
- `path_csv` parameter
  - Optional CSV centerline reference.
  - Supported columns: `x`, `y`, optional `v_ref`.
  - If omitted and `reference_mode=auto`, the controller falls back to constant-speed straight-line tracking.

## Outputs

- `/mpc/drive` (`ackermann_msgs/msg/AckermannDriveStamped`)
  - Raw MPC command for debugging or direct integration.
- `/drive_control` (`dev_b7_interfaces/msg/DriveControlMessage`)
  - Enabled by default.
  - Used to integrate with the repo's safety arbitration pipeline.

## Parameters

- `reference_mode`
  - `auto`, `path_csv`, or `constant_speed`
- `path_csv`
  - Path to the CSV centerline file.
- `target_speed`
  - Fallback speed target when the CSV does not provide `v_ref`.
- `command_priority`
  - Priority attached to `/drive_control`.
- `stop_on_invalid_reference`
  - Publishes a safe stop when no usable reference is available.
- `stop_on_solver_failure`
  - Publishes a safe stop when the QP backend returns no usable solution.

## Solver Contract

- State: `[x, y, psi, v, delta]`
- Control: `[a, delta_dot]`
- Backend: CVXPY + OSQP
- Linearization: stagewise linearized bicycle model over the horizon
