# MPC Assumptions

- The current project does not yet provide a finished planning topic, so the MPC controller accepts a CSV centerline as its practical path source.
- Steering angle is not directly measured from odometry in this repo, so the node estimates steering from yaw rate and speed, then carries the commanded steering between odometry updates.
- The controller uses a kinematic bicycle model and linearizes it stage by stage for a QP-based MPC loop.
- If `cvxpy` / `osqp` are unavailable at runtime, the node safely stops when `stop_on_solver_failure=true`.
- The existing safety package remains the final arbitration layer for commands sent to the car.
