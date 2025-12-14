import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# Vehicle parameters
# ============================================================
L = 0.35          # wheelbase [m]
t = 0.28          # track width [m]
lf = 0.18         # CG -> front axle [m]
lr = L - lf       # CG -> rear axle  [m]

m  = 8.0          # mass [kg]
Iz = 0.45         # yaw inertia [kg*m^2]
g  = 9.81

# Tire model parameters (linear + saturation)
Cf = 250.0        # cornering stiffness front per wheel [N/rad]
Cr = 280.0        # cornering stiffness rear  per wheel [N/rad]
mu = 0.9          # friction coefficient

# Longitudinal "drive" model
Cdrag = 0.6
vx_min = 0.05

# ============================================================
# Simulation settings & Squircle Generation
# ============================================================
dt = 0.01
T = 20.0          # Runtime [s] (adjusted to allow full lap)
N = int(T/dt)
tt = np.linspace(0, T, N)

v_x_des = 0.8     # Target speed [m/s]
Kvx = 30.0        # Cruise control gain
Fx_max_total = 35.0

# --- SQUIRCLE PARAMETERS ---
R_sq = 2.0        # "Radius" (half-width) of the shape
n_order = 4.0     # 4.0 = standard squircle
laps = 1.0        # How many laps to drive in T seconds
period = T / laps

# Generate the parameter 'theta' over time
# We shift by -pi/2 so we start at (0, -R) driving East, or 0 to start at (R,0) driving North.
# Let's start at theta=0 => Position (R, 0), Heading North.
theta_traj = np.linspace(0, 2 * np.pi * laps, N)

# Parametric Squircle Equations:
# x = R * sgn(cos t) * |cos t|^(2/n)
# y = R * sgn(sin t) * |sin t|^(2/n)
p = 2.0 / n_order
xs = R_sq * np.sign(np.cos(theta_traj)) * np.abs(np.cos(theta_traj)) ** p
ys = R_sq * np.sign(np.sin(theta_traj)) * np.abs(np.sin(theta_traj)) ** p

# Compute Derivatives for Curvature (Kappa)
dx_dt = np.gradient(xs, dt)
dy_dt = np.gradient(ys, dt)
ddx_dt = np.gradient(dx_dt, dt)
ddy_dt = np.gradient(dy_dt, dt)

# Curvature k = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5
numerator = dx_dt * ddy_dt - dy_dt * ddx_dt
denominator = (dx_dt**2 + dy_dt**2)**1.5
kappa = numerator / (denominator + 1e-6)

# Calculate Steering Angle from Curvature (Kinematic Bicycle Approx)
# delta = arctan(L * k)
delta_profile = np.arctan(L * kappa)

# Clamp steering to physical limits
delta_max = np.deg2rad(25.0)
delta_profile = np.clip(delta_profile, -delta_max, delta_max)

# ============================================================
# Drawing geometry
# ============================================================
body_overhang_rear  = 0.05
body_overhang_front = 0.05
wheel_len, wheel_wid = 0.16, 0.06

# Wheel positions in body frame (origin at CG)
p_FL = np.array([+lf, +t/2])
p_FR = np.array([+lf, -t/2])
p_RL = np.array([-lr, +t/2])
p_RR = np.array([-lr, -t/2])
wheel_pos = np.stack([p_FL, p_FR, p_RL, p_RR], axis=0)
wheel_names = ["FL", "FR", "RL", "RR"]

# ============================================================
# Helper functions
# ============================================================
def R2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],[s, c]])

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def ackermann_split(delta, L, t):
    td = np.tan(delta)
    if abs(td) < 1e-8:
        return delta, delta
    Rmid = L / td
    # Avoid singular Rmid inside track width
    if abs(Rmid) <= t/2 + 1e-6:
        Rmid = np.sign(Rmid) * (t/2 + 1e-6)
    dFL = np.arctan2(L, Rmid - t/2)
    dFR = np.arctan2(L, Rmid + t/2)
    return dFL, dFR

def box_corners(center_xy, heading, length, width):
    l, w = length/2.0, width/2.0
    local = np.array([[+l,+w],[+l,-w],[-l,-w],[-l,+w],[+l,+w]])
    return (R2(heading) @ local.T).T + center_xy

def body_outline(center_xy, heading):
    Lb = lf + lr + body_overhang_front + body_overhang_rear
    x_center = (lf + body_overhang_front - (lr + body_overhang_rear)) / 2.0
    body_center_R = np.array([x_center, 0.0])
    return box_corners(center_xy + (R2(heading) @ body_center_R), heading, Lb, t)

def clamp_friction(Fx, Fy, Fz, mu):
    """Friction circle: sqrt(Fx^2 + Fy^2) <= mu*Fz."""
    Fmax = mu * Fz
    mag = np.hypot(Fx, Fy)
    if mag <= Fmax or mag < 1e-12:
        return Fx, Fy
    s = Fmax / mag
    return Fx*s, Fy*s

# ============================================================
# Normal loads (static)
# ============================================================
Fz_front_axle = m * g * (lr / L)
Fz_rear_axle  = m * g * (lf / L)
Fz = np.array([
    0.5 * Fz_front_axle,  # FL
    0.5 * Fz_front_axle,  # FR
    0.5 * Fz_rear_axle,   # RL
    0.5 * Fz_rear_axle,   # RR
])

# ============================================================
# Explicit matrices for dynamics (BODY coordinates)
# ============================================================
M = np.diag([m, m, Iz])  # 3x3

def C_matrix(nu):
    vx, vy, r = nu
    return np.array([
        [0.0,  m*r, 0.0],
        [-m*r, 0.0, 0.0],
        [0.0,  0.0, 0.0]
    ])

def H_i(xi, yi):
    return np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-yi, xi]
    ])

def Bx_By(wheel_heading):
    Bx = np.zeros((3, 4))
    By = np.zeros((3, 4))
    for i in range(4):
        xi, yi = wheel_pos[i]
        Hi = H_i(xi, yi)
        di = wheel_heading[i]
        a = np.array([np.cos(di), np.sin(di)]) 
        b = np.array([-np.sin(di), np.cos(di)])
        Bx[:, i] = Hi @ a
        By[:, i] = Hi @ b
    return Bx, By

# ============================================================
# State initialization
# ============================================================
x   = np.zeros(N)
y   = np.zeros(N)
psi = np.zeros(N)

vx  = np.zeros(N)
vy  = np.zeros(N)
r_yaw = np.zeros(N)

# Set Initial State to match the Squircle Start
vx[0]  = v_x_des
x[0]   = xs[0]
y[0]   = ys[0]

# Calculate initial heading from trajectory derivative
initial_heading = np.arctan2(dy_dt[0], dx_dt[0])
psi[0] = initial_heading

Fx_w = np.zeros((N, 4))
Fy_w = np.zeros((N, 4))
alpha_w = np.zeros((N, 4))
delta_act_log = np.zeros(N)

# ============================================================
# Simulation loop
# ============================================================
for k in range(N-1):
    
    # 1. READ STEERING FROM SQUIRCLE PROFILE
    delta_act = delta_profile[k]
    delta_act_log[k] = delta_act
    v_ref = v_x_des

    # 2. ACKERMANN GEOMETRY
    dFL, dFR = ackermann_split(delta_act, L, t)
    wheel_heading = np.array([dFL, dFR, 0.0, 0.0])

    # 3. LONGITUDINAL CONTROL (Cruise Control)
    Fx_cmd_total = Kvx * (v_ref - vx[k]) - Cdrag * vx[k]
    Fx_cmd_total = np.clip(Fx_cmd_total, -Fx_max_total, Fx_max_total)
    Fx_cmd = np.array([0.0, 0.0, 0.5*Fx_cmd_total, 0.5*Fx_cmd_total]) # RWD

    # 4. TIRE VELOCITIES
    v_i = np.zeros((4, 2))
    for i in range(4):
        xi, yi = wheel_pos[i]
        v_i[i, 0] = vx[k] - r_yaw[k] * yi
        v_i[i, 1] = vy[k] + r_yaw[k] * xi

    # 5. TIRE FORCES
    Fx_i = np.zeros(4)
    Fy_i = np.zeros(4)

    for i in range(4):
        vxi = np.sign(v_i[i, 0]) * max(abs(v_i[i, 0]), vx_min)
        vyi = v_i[i, 1]

        alpha = wrap_pi(wheel_heading[i] - np.arctan2(vyi, vxi))
        alpha_w[k, i] = alpha

        Cc = Cf if i < 2 else Cr
        Fy = Cc * alpha
        Fx = Fx_cmd[i]

        Fx, Fy = clamp_friction(Fx, Fy, Fz[i], mu)
        Fx_i[i] = Fx
        Fy_i[i] = Fy

    Fx_w[k, :] = Fx_i
    Fy_w[k, :] = Fy_i

    # 6. DYNAMICS SOLVER
    nu = np.array([vx[k], vy[k], r_yaw[k]])
    Bx, By = Bx_By(wheel_heading)
    tau = (Bx @ Fx_i) + (By @ Fy_i)
    C_mat = C_matrix(nu)
    
    # M * nu_dot + C * nu = tau
    nu_dot = np.linalg.solve(M, (tau - C_mat @ nu))

    # 7. INTEGRATION
    vx[k+1]    = vx[k]    + nu_dot[0] * dt
    vy[k+1]    = vy[k]    + nu_dot[1] * dt
    r_yaw[k+1] = r_yaw[k] + nu_dot[2] * dt

    # World Kinematics
    x_dot = vx[k]*np.cos(psi[k]) - vy[k]*np.sin(psi[k])
    y_dot = vx[k]*np.sin(psi[k]) + vy[k]*np.cos(psi[k])
    psi_dot = r_yaw[k]

    x[k+1]   = x[k]   + x_dot * dt
    y[k+1]   = y[k]   + y_dot * dt
    psi[k+1] = wrap_pi(psi[k] + psi_dot * dt)

# Fill last steps for logging
delta_act_log[-1] = delta_act_log[-2]
Fx_w[-1, :] = Fx_w[-2, :]
Fy_w[-1, :] = Fy_w[-2, :]

# ============================================================
# Figure 1: Animation
# ============================================================
fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
ax_anim.set_aspect('equal')
ax_anim.set_title("Squircle Tracking (Dynamic Model)")
ax_anim.set_xlabel("X [m]")
ax_anim.set_ylabel("Y [m]")

# Set bounds based on squircle radius
pad = 1.0
ax_anim.set_xlim(-R_sq - pad, R_sq + pad)
ax_anim.set_ylim(-R_sq - pad, R_sq + pad)

(trace,) = ax_anim.plot([], [], 'b-', lw=1.5, label="Actual Path")
(ref_line,) = ax_anim.plot(xs, ys, 'r--', lw=1.5, alpha=0.5, label="Desired Squircle")

(body_line,) = ax_anim.plot([], [], 'k-', lw=2)
(FL_w_line,) = ax_anim.plot([], [], 'k-', lw=3)
(FR_w_line,) = ax_anim.plot([], [], 'k-', lw=3)
(RL_w_line,) = ax_anim.plot([], [], 'k-', lw=3)
(RR_w_line,) = ax_anim.plot([], [], 'k-', lw=3)
ax_anim.legend(loc="upper right")

def update_anim(i):
    # Only update every n frames to speed up animation render if needed
    # i = i * 2 
    if i >= N: return
    
    pos = np.array([x[i], y[i]])
    heading = psi[i]
    dFL, dFR = ackermann_split(delta_act_log[i], L, t)

    body_poly = body_outline(pos, heading)
    body_line.set_data(body_poly[:,0], body_poly[:,1])

    Rg = R2(heading)
    cFL = pos + (Rg @ p_FL)
    cFR = pos + (Rg @ p_FR)
    cRL = pos + (Rg @ p_RL)
    cRR = pos + (Rg @ p_RR)

    FL_poly = box_corners(cFL, heading + dFL, wheel_len, wheel_wid)
    FR_poly = box_corners(cFR, heading + dFR, wheel_len, wheel_wid)
    RL_poly = box_corners(cRL, heading,       wheel_len, wheel_wid)
    RR_poly = box_corners(cRR, heading,       wheel_len, wheel_wid)

    FL_w_line.set_data(FL_poly[:,0], FL_poly[:,1])
    FR_w_line.set_data(FR_poly[:,0], FR_poly[:,1])
    RL_w_line.set_data(RL_poly[:,0], RL_poly[:,1])
    RR_w_line.set_data(RR_poly[:,0], RR_poly[:,1])

    trace.set_data(x[:i+1], y[:i+1])
    return trace, body_line, FL_w_line, FR_w_line, RL_w_line, RR_w_line

# Decimate frames slightly for smoother realtime viewing if N is huge
ani = FuncAnimation(fig_anim, update_anim, frames=range(0, N, 4), 
                    interval=dt*4*1000, blit=True, repeat=False)

# ============================================================
# Figure 2: States
# ============================================================
fig_st, axs_st = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axs_st[0].plot(tt, np.rad2deg(delta_act_log))
axs_st[0].set_ylabel("Steering [deg]")
axs_st[0].grid(True)
axs_st[0].set_title("Control Inputs & Yaw Rate")

axs_st[1].plot(tt, vx, label="vx")
axs_st[1].plot(tt, vy, label="vy")
axs_st[1].set_ylabel("Velocity [m/s]")
axs_st[1].legend()
axs_st[1].grid(True)

axs_st[2].plot(tt, r_yaw)
axs_st[2].set_ylabel("Yaw Rate [rad/s]")
axs_st[2].set_xlabel("Time [s]")
axs_st[2].grid(True)

plt.tight_layout()
plt.show()