# Gap Following Node Inspection

Reviewed files:
- `gap_following/main_node.py`
- `gap_following/gap_utils.py`
- `gap_following/PID_control.py`

## Findings

### 2) High: `min_safe_distance` default is a tuple, not a float
- `main_node.py:108` has a trailing comma:
  - `self.min_safe_distance = 0.25,`
- This creates `(0.25,)` and can break math/type expectations when defaults are used.
- Relevant lines:
  - `main_node.py:108`
  - `main_node.py:122`

### 3) Medium: `cost_diff` tuning is not actually used
- `main_node.py` loads and updates `cost_diff`, but gap scoring uses a hardcoded `30`.
- This makes `cost_diff` in `config.json` ineffective.
- Relevant lines:
  - `main_node.py:79`
  - `main_node.py:160`
  - `gap_utils.py:71`

### 4) Medium: Odometry speed is collected but ignored in control logic
- `odom_callback` stores `self.current_speed`, but speed command is computed only from steering.
- Potential missed opportunity for speed-aware stability/safety behavior.
- Relevant lines:
  - `main_node.py:132`
  - `main_node.py:220`

### 5) Medium: PID steering saturation may conflict with configured max steering
- `PIDControl` default `steering_limit=0.7` clamps output even when node config sets `MAX_STEERING_ANGLE=1.4`.
- This can artificially cap steering authority in tight turns.
- Relevant lines:
  - `PID_control.py:23`
  - `PID_control.py:107`
  - `main_node.py:153`

## Integration Check
- Publish path and message type match safety mux expectations:
  - Publishes `dev_b7_interfaces/msg/DriveControlMessage` on `DriveControlMessage.BUILTIN_TOPIC_NAME_STRING` (`/drive_control`).
  - Safety node subscribes to the same built-in topic and uses highest `priority`.

