# Contribution Report

**Project:** Milestone 1 - Wall Following

**Team Members:** Amir Tajaddoditalab, Augustin So, Fengwei Huang, Guanxu Zhou, Leo Liu, Yiran Wang
Yiran Wang

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
**Primary Focus:**

### Key Contributions
* 

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
**Primary Focus:**

### Key Contributions
* 

---

##  Yiran Wang
**Primary Focus:**

### Key Contributions
* 

