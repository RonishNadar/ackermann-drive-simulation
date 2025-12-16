# Ackermann Drive Simulation

**A high-fidelity Python simulation of a Double-Track Ackermann vehicle featuring Rigid Body Dynamics and Sliding Mode Control (SMC).**

This repository implements a complete robotics stack—from Motion Planning to Control to Dynamics simulation—verifying the stability of a vehicle tracking a Cubic Spline path under realistic physical constraints.

## 📂 Repository Structure

```text
ackermann-drive-simulation/
├── Controller/
│   └── controller_sim.py    # Main Sliding Mode Controller (SMC) implementation
├── Dynamics/
│   └── ...                  # Rigid Body physics engine & tire models
├── Kinematics/
│   └── kinematics_sim.py    # Inverse kinematics & geometry validation
├── Planner/
│   └── ...                  # Path generation (Cubic Spline interpolation)
└── README.md

```

## 🚀 Key Features* **Double-Track Ackermann Geometry:** Unlike simplified bicycle models, this project models all four wheels independently, accounting for track width (T) and calculating unique steering angles for inner/outer wheels.
* **Rigid Body Dynamics Engine:** Simulates the vehicle using Newton-Euler equations, accounting for:
* Inertia matrices (\mathbf{M}) and Coriolis forces (\mathbf{C}(\nu)).
* Linear/Rotational Damping (Air drag & friction).
* **Linear Tire Model:** Calculates lateral tire forces (F_y = -C_{\alpha}\alpha) based on slip angles.


* **Sliding Mode Controller (SMC):**
* **Lateral Control:** Tracks cross-track error (e_y) and heading error (e_{\psi}) using a smoothed reaching law (\tanh) to eliminate chattering.
* **Longitudinal Control:** Schedules speed based on path curvature (\kappa) to ensure safe cornering.


* **Stability Analysis:** Includes real-time logging of Lyapunov function candidates to empirically verify system stability.

## 🛠️ System Requirements###Prerequisites* **OS:** Linux (Recommended), Windows, or macOS.
* **Python:** 3.8+
* **FFmpeg:** Required if you want to save animations as `.mp4`.

### Python DependenciesInstall the required libraries using pip:

```bash
pip install numpy matplotlib scipy

```

*(Optional)* If you are on Linux and want smoother video export:

```bash
sudo apt install ffmpeg

```

## 💻 Usage ### 1. Run the Full Control SimulationTo see the robot track a Figure-8 path with the Sliding Mode Controller:

```bash
python Controller/controller_sim.py

```

**Output:**

* A real-time animation of the vehicle.
* `controller_sim_animation.mp4`: Saved video of the run.
* **Stability Plots:** Graphs showing the decay of tracking errors and Lyapunov candidates.

### 2. Run the Kinematics ValidationTo verify the Inverse Kinematics mapping (Global Velocity \to Wheel Speeds):

```bash
python Kinematics/kinematics_sim.py

```

## 📚 Technical Details###1. Kinematics (Double-Track)The system maps a desired inertial velocity [\dot{X}, \dot{Y}, \dot{\psi}]^T to individual wheel angular velocities (\omega) using the Jacobian matrix for a double-track model. This ensures correct differential steering where outer wheels spin faster than inner wheels during turns.

### 2. Dynamics (Newton-Euler)The physics engine solves the following equation of motion at every time step (dt=0.01s):

* \boldsymbol{\tau}_{tires}: Derived from a linear tire model F_y = -C_{\alpha}\alpha.
* \mathbf{D}\mathbf{\nu}: Linearly proportional damping (drag).

### 3. Control Strategy (SMC)The controller drives the robot along a path defined by a Cubic Spline.

* **Sliding Surface:** s_{lat} = e_{\psi} + \lambda e_y
* **Control Law:** A continuous approximation of the switching law to prevent high-frequency jitter (chattering):



##📊 Results & ValidationThe simulation outputs comprehensive plots to validate performance:

1. **Trajectory:** Visual confirmation of path tracking (e.g., Figure-8).
2. **Lyapunov Stability:** Plots of V = \frac{1}{2}s^2 demonstrating that error energy decays to zero.
3. **Actuation:** Verification that steering and torque commands remain within physical saturation limits (\pm 45^\circ, 8 Nm).

##📝 LicenseThis project is open-source. Feel free to use it for educational or research purposes.

**Author:** Ronish Nadar
**GitHub:** [https://github.com/RonishNadar/ackermann-drive-simulation](https://github.com/RonishNadar/ackermann-drive-simulation)