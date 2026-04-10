# Contribution Report

**Project:** Milestone 1 - Wall Following

**Team Members:** Amir Tajaddoditalab, Augustin So, Fengwei Huang, Guanxu Zhou, Leo Liu, Yiran Wang

---
##  Amir Tajaddoditalab
**Primary Focus:** PID Controller and tuning

### Key Contributions
* **Code implementations**
* Helped bringup PID controller library to be used by the wall following algorithm
* Tested varying factors such as different controller gains in order to achieve a fast step response
* Stress tested the car under different conditions such as starting the car at a positional angle to ensure the PID controller and adjust steering angle to achieve the desired path

---
##  Augustin So
**Primary Focus:** distance and angle calculation relative to walls, LiDAR interface functions

### Key Contributions
* **Code Implementation:**
* Implemented error calculation from relative angle and distance
* Attempted different wall following strategies to compare effectiveness
* Tested a variety of parameters for k_p, k_d, k_i
* Implemented proper usage of proportional, integral, derivative terms
* **Testing:**
* Performed testing on calculations using self.get_logger to verify angle and distance correctness during simulation
* Tested a variety of starting setups to verify the response (overdamped, underdamped, critically damped)

---
## Fengwei Huang

**Primary Focus:** MPC-related development, documentation, and testing support

### Key Contributions
- **MPC-related development:** Contributed to an alternative trajectory-tracking MPC branch, including work on vehicle modeling, discretization, local linearization, and reference handling.
- **Documentation:** Helped write and revise the README, including implementation details, testing notes, and system-level descriptions.
- **Testing support:** Assisted with routine testing, validation, parameter checking, and debugging under different configurations.

---
## Guanxu Zhou
**Primary Focus:** wall follow node data process implementation

### Key Contributions
* **Code Implementation:**
  * Implemented the data process part of wall following node
  * Integrated PID control with data process
* **Debug**
  * Test out different parameter for pid
  * Performed Long-Duration Stress Test
* **Project Management**
  * Doing code review on pull requests
  * Defined project scope and delegated tasks.
  * Set up the project scaffolding

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
