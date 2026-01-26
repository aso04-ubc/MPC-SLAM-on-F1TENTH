## Algorithm Explanation ##

The angle and distance difference were used between two lasers to find the orientation of the car relative to the wall. The formulae used were

$$
\alpha = \tan^{-1}\left(\frac{a\sin(\theta)-b}{a\cos(\theta)}\right)
$$

$$
CD = b\cos(\alpha)
$$

Using the distance from the wall on both sides (named CD in the above equation), y was calculated by subtracting the distances on both sides of the car.

$$
y = \mathrm{dist}_{\text{left}} - \mathrm{dist}_{\text{right}}
$$

The total error formula was given as follows

$$
\Theta_d = -(y + L \cdot \sin(\alpha))
$$

where L is a chosen distance in front of the car. We selected 1.5m. The steering angle is set to $\Theta_d$

Next, PID was implemented keeping track of the previous error and storing inside self.previous_error. The change in error over time is given with the formula

$$
\frac{de(t)}{dt} = \frac{e(t) - e(t-\Delta t)}{\Delta t}
$$

due to noisy data coming from the LiDAR sensor, a low pass filter was applied to the signal to smooth out the data and prevent jitter during driving caused by the derivative term.

Finally, integral was implemented by constantly adding the error in each frame over time to self.integral. self.integral is capped at a maximum and minimum of +/- 1.0 to prevent overcorrection after the car is in an error state.

a drive control node was implemented for node priority ordering, giving maximum priority to emergency braking when it is activated.
