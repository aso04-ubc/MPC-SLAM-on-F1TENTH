# Contribution Report

**Project:** MPC & SLAM

**Team Members:** Amir Tajaddoditalab, Augustin So, Fengwei Huang, Guanxu Zhou, Leo Liu, Yiran Wang

---
##  Amir Tajaddoditalab
**Primary Focus:** MPC Controller implementation and tuning
* **Code implementation**
* Devloped the MPC controller frame work along with the cost function as well as reactive path planning
* Validates MPC Controller with sim and further tuned parameters on track on the real physical car

### Key Contributions
* **Code implementations**
* Helped bringup PID controller library to be used by the wall following algorithm
* Tested varying factors such as different controller gains in order to achieve a fast step response
* Stress tested the car under different conditions such as starting the car at a positional angle to ensure the PID controller and adjust steering angle to achieve the desired path

---
##  Augustin So
**Primary Focus:** SLAM error correction, LiDAR interface functions

### Key Contributions
* **Code Implementation:**
* Implemented ICP to correct IMU errors with minimal introduction of noise
* Map building, live SLAM through vehicle position and yaw estimates, calculating occupancy grid
* Interfacing path planning and controller
* **Testing:**
* SLAM debugging in simulation and on hardware
* Verification of ICP correctness, debugging

---
## Fengwei Huang

**Primary Focus:** MPC-related development, documentation, and testing support

### Key Contributions
- **MPC-related development:** Contributed to an alternative trajectory-tracking MPC branch, including work on vehicle modeling, discretization, local linearization, and reference handling.
- **Documentation:** Helped write and revise the README, including implementation details, testing notes, and system-level descriptions.
- **Testing support:** Assisted with routine testing, validation, parameter checking, and debugging under different configurations.

---
## Guanxu Zhou
**Primary Focus:** semantic perception pipeline, mapping deployment, and planner optimization

### Key Contributions
* **YOLO annotation, training, and testing:** Contributed to dataset annotation, model training, and testing workflows for the YOLO model.
* **Semantic perception research:** Explored sensor-fusion and localization-improvement directions relevant to the semantic-perception pipeline.
* **Mapping deployment:** Worked on transforming the mapping model into an on-car working version.
* **Parameter tuning:** Participated in system-level tuning to improve runtime performance and reliability.
* **High-level planner updates:** Contributed to modifications and refinement of the high-level planner.

---
##  Leo Liu
**Primary Focus:** high-level planning research, system integration documentation

### Key Contributions
* **High-level planning research:** Contributed to high-level planning research, including track-level planning structure and decomposition of planning responsibilities across mapping, planning, and control modules.
* **Interface design:** Defined how planner outputs should be interfaced with the controller and simulation workflow, including expected path representation, frame consistency, and topic-level integration assumptions.
* **Documentation:** Expanded technical documentation to clarify planning/control data flow, system architecture decisions, and integration rationale for team implementation.

---
## Yiran Wang
**Primary Focus:** safety node architecture, performance tuning & diagnostics

### Key Contributions
* **Safety node architecture:** Implemented a high-performance, low-latency safety node for real-time message prioritization and forwarding.
* **Custom interfaces & topics:** Defined custom message types and topics to facilitate seamless system integration.
* **Parameter tuning & debugging:** Performed extensive fine-tuning of control parameters and assisted in overall system debugging.
* **System diagnostics:** Analyzed sensor recordings and feedback to identify and analyze technical problems.
