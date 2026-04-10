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
##  Fengwei Huang
**Primary Focus:** PID controller implementation

### Key Contributions
* **Code implementations**
  * Implemented the PID controller used by the wall following algorithm
  * Defined the wall-following control error based on lateral distance and heading angle
  * Structured the controller as a standalone module for integration with the wall-follow node
* **Stability**
  * Added basic safeguards to improve control stability and prevent unsafe steering behavior


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
**Primary Focus:** data processing, documentation

### Key Contributions
* **Data Processing:** Discussed data processing algorithms and integration with PID control for implementation.
* **Documentation:** Wrote README.md explaining detailed technical aspects such as data processing, Kalman filtering, safety, drive control, etc.
* **Testing:** Stress tested the car under multiple scenarios for final verification.

---
## Yiran Wang
**Primary Focus:** safety node architecture, performance tuning & diagnostics

### Key Contributions
* **Safety node architecture:** Implemented a high-performance, low-latency safety node for real-time message prioritization and forwarding.
* **Custom interfaces & topics:** Defined custom message types and topics to facilitate seamless system integration.
* **Parameter tuning & debugging:** Performed extensive fine-tuning of control parameters and assisted in overall system debugging.
* **System diagnostics:** Analyzed sensor recordings and feedback to identify and analyze technical problems.
