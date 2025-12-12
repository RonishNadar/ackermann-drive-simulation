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
# Simulation settings
# ============================================================
dt, T = 0.01, 15.0
N = int(T/dt)
tt = np.linspace(0, T, N)

v_x_des = 0.8

delta_max = np.deg2rad(25.0)
delta_sq  = delta_max

side_len = 1.6

Kvx = 30.0
Fx_max_total = 35.0

# ============================================================
# Drawing geometry
# ============================================================
body_overhang_rear  = 0.05
body_overhang_front = 0.05
wheel_len, wheel_wid = 0.16, 0.06

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
    Fmax = mu * Fz
    mag = np.hypot(Fx, Fy)
    if mag <= Fmax or mag < 1e-12:
        return Fx, Fy
    scale = Fmax / mag
    return Fx * scale, Fy * scale

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
# State arrays
# ============================================================
x   = np.zeros(N)
y   = np.zeros(N)
psi = np.zeros(N)

vx  = np.zeros(N)
vy  = np.zeros(N)
r   = np.zeros(N)

vx[0] = v_x_des

Fx_w = np.zeros((N, 4))
Fy_w = np.zeros((N, 4))
alpha_w = np.zeros((N, 4))
delta_act_log = np.zeros(N)

# ============================================================
# Square steering state machine (instant steering)
# ============================================================
STRAIGHT = 0
TURN = 1
mode = STRAIGHT
side_idx = 0
x_start, y_start = x[0], y[0]
psi_turn_start = psi[0]

# ============================================================
# Simulation loop
# ============================================================
for k in range(N-1):
    if side_idx >= 4:
        delta_act = 0.0
        v_ref = 0.0
    else:
        if mode == STRAIGHT:
            delta_act = 0.0
            v_ref = v_x_des
            dist = np.hypot(x[k] - x_start, y[k] - y_start)
            if dist >= side_len:
                mode = TURN
                psi_turn_start = psi[k]
        else:
            delta_act = delta_sq
            v_ref = v_x_des
            dpsi = wrap_pi(psi[k] - psi_turn_start)
            if dpsi >= (np.pi/2):
                mode = STRAIGHT
                side_idx += 1
                x_start, y_start = x[k], y[k]

    delta_act = np.clip(delta_act, -delta_max, +delta_max)
    delta_act_log[k] = delta_act

    dFL, dFR = ackermann_split(delta_act, L, t)
    wheel_heading = np.array([dFL, dFR, 0.0, 0.0])

    Fx_cmd_total = Kvx * (v_ref - vx[k]) - Cdrag * vx[k]
    Fx_cmd_total = np.clip(Fx_cmd_total, -Fx_max_total, Fx_max_total)
    Fx_cmd = np.array([0.0, 0.0, 0.5*Fx_cmd_total, 0.5*Fx_cmd_total])

    v_i = np.zeros((4, 2))
    for i in range(4):
        xi, yi = wheel_pos[i]
        v_i[i, 0] = vx[k] - r[k] * yi
        v_i[i, 1] = vy[k] + r[k] * xi

    Fx_i = np.zeros(4)
    Fy_i = np.zeros(4)

    for i in range(4):
        vxi = np.sign(v_i[i, 0]) * max(abs(v_i[i, 0]), vx_min)
        vyi = v_i[i, 1]
        alpha = wrap_pi(wheel_heading[i] - np.arctan2(vyi, vxi))
        alpha_w[k, i] = alpha

        C = Cf if i < 2 else Cr
        Fy = C * alpha
        Fx = Fx_cmd[i]
        Fx, Fy = clamp_friction(Fx, Fy, Fz[i], mu)

        Fx_i[i] = Fx
        Fy_i[i] = Fy

    Fx_w[k, :] = Fx_i
    Fy_w[k, :] = Fy_i

    Fxb = 0.0
    Fyb = 0.0
    Mzb = 0.0

    for i in range(4):
        th_i = wheel_heading[i]
        c, s = np.cos(th_i), np.sin(th_i)
        Fx_bi = c*Fx_i[i] - s*Fy_i[i]
        Fy_bi = s*Fx_i[i] + c*Fy_i[i]
        Fxb += Fx_bi
        Fyb += Fy_bi
        xi, yi = wheel_pos[i]
        Mzb += xi * Fy_bi - yi * Fx_bi

    vx_dot = (Fxb / m) + r[k] * vy[k]
    vy_dot = (Fyb / m) - r[k] * vx[k]
    r_dot  = Mzb / Iz

    x_dot = vx[k]*np.cos(psi[k]) - vy[k]*np.sin(psi[k])
    y_dot = vx[k]*np.sin(psi[k]) + vy[k]*np.cos(psi[k])
    psi_dot = r[k]

    vx[k+1]  = vx[k]  + vx_dot * dt
    vy[k+1]  = vy[k]  + vy_dot * dt
    r[k+1]   = r[k]   + r_dot  * dt
    psi[k+1] = wrap_pi(psi[k] + psi_dot * dt)
    x[k+1]   = x[k]   + x_dot  * dt
    y[k+1]   = y[k]   + y_dot  * dt

delta_act_log[-1] = delta_act_log[-2]
Fx_w[-1, :] = Fx_w[-2, :]
Fy_w[-1, :] = Fy_w[-2, :]
alpha_w[-1, :] = alpha_w[-2, :]

# ============================================================
# Figure 1: Animation (separate window)
# ============================================================
fig_anim, ax_anim = plt.subplots(figsize=(7, 7))
ax_anim.set_aspect('equal')
ax_anim.set_title("4-Wheel Dynamic Ackermann (Double-Track) - Animation")
ax_anim.set_xlabel("X [m]")
ax_anim.set_ylabel("Y [m]")
pad = 0.6
ax_anim.set_xlim(x.min()-pad, x.max()+pad)
ax_anim.set_ylim(y.min()-pad, y.max()+pad)

(trace,) = ax_anim.plot([], [], lw=2, label="path")
(body_line,) = ax_anim.plot([], [], lw=2)

(FL_w_line,) = ax_anim.plot([], [], lw=3, label="FL")
(FR_w_line,) = ax_anim.plot([], [], lw=3, label="FR")
(RL_w_line,) = ax_anim.plot([], [], lw=3, label="RL")
(RR_w_line,) = ax_anim.plot([], [], lw=3, label="RR")
ax_anim.legend(loc="upper right")

def update_anim(i):
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

ani = FuncAnimation(fig_anim, update_anim, frames=N, interval=dt*1000,
                    blit=False, repeat=False)

# ============================================================
# Figure 2: Body velocities + yaw rate (separate window, 3 subplots)
# ============================================================
fig_vel, axs_vel = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
fig_vel.suptitle("Body Velocities and Yaw Rate")

axs_vel[0].plot(tt, vx)
axs_vel[0].set_ylabel("vx [m/s]")
axs_vel[0].grid(True)

axs_vel[1].plot(tt, vy)
axs_vel[1].set_ylabel("vy [m/s]")
axs_vel[1].grid(True)

axs_vel[2].plot(tt, r)
axs_vel[2].set_ylabel("r [rad/s]")
axs_vel[2].set_xlabel("Time [s]")
axs_vel[2].grid(True)

fig_vel.tight_layout(rect=[0, 0, 1, 0.95])

# ============================================================
# Figure 3: Wheel longitudinal forces Fx (separate window, 4 subplots)
# ============================================================
fig_fx, axs_fx = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
fig_fx.suptitle("Wheel Longitudinal Forces $F_x$")

for i in range(4):
    axs_fx[i].plot(tt, Fx_w[:, i])
    axs_fx[i].set_ylabel(f"{wheel_names[i]} [N]")
    axs_fx[i].grid(True)

axs_fx[-1].set_xlabel("Time [s]")
fig_fx.tight_layout(rect=[0, 0, 1, 0.95])

# ============================================================
# Figure 4: Wheel lateral forces Fy (separate window, 4 subplots)
# ============================================================
fig_fy, axs_fy = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
fig_fy.suptitle("Wheel Lateral Forces $F_y$")

for i in range(4):
    axs_fy[i].plot(tt, Fy_w[:, i])
    axs_fy[i].set_ylabel(f"{wheel_names[i]} [N]")
    axs_fy[i].grid(True)

axs_fy[-1].set_xlabel("Time [s]")
fig_fy.tight_layout(rect=[0, 0, 1, 0.95])

# ============================================================
# Show everything
# ============================================================
if __name__ == "__main__":
    plt.show()
    # To save animation (needs ffmpeg):
    # ani.save("dynamics_sim.mp4", dpi=120, writer="ffmpeg")
