import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- Parameters ----------------
# Geometry & mass properties
L, t, r = 0.35, 0.28, 0.06            # wheelbase, track, wheel radius
m, Iz   = 8.0, 0.45                   # mass [kg], yaw inertia [kg m^2]

# Motion and sim
dt, T   = 0.01, 12.0                  # timestep, total time
v_x     = 0.8                         # desired body-frame forward speed
delta_max = np.deg2rad(18.0)          # max bicycle steering
S_period = 6.0                        # steering sine period

# Mild viscous damping (global)
d_x, d_y, d_th = 0.8, 0.8, 0.05

# Controller gains (global coordinates)
Kp = np.diag([5.0, 5.0, 4.0])
Kd = np.diag([2.0, 2.0, 1.6])

# Torque solver / actuator limits
lambda_damp = 0.2                     # Tikhonov damping for B
tau_max     = 2.0                     # wheel torque limit [N·m]

# Visuals
body_overhang_rear  = 0.05
body_overhang_front = 0.05
wheel_len, wheel_wid = 0.16, 0.06

# Wheel centers in robot frame (origin: rear-axle midpoint, x forward, y left)
p_FL = np.array([L,  t/2]); p_FR = np.array([L, -t/2])
p_RL = np.array([0., t/2]); p_RR = np.array([0., -t/2])

# ---------------- Helper functions ----------------
def R2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s],[s, c]])

def TG2R(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def ackermann_split(delta, L, t):
    td = np.tan(delta)
    if abs(td) < 1e-8:
        return delta, delta
    Rmid = L/td
    if abs(Rmid) <= t/2 + 1e-6:
        Rmid = np.sign(Rmid)*(t/2 + 1e-6)
    return np.arctan2(L, Rmid - t/2), np.arctan2(L, Rmid + t/2)

def A_matrix(dFL, dFR, L, t):
    return np.array([
        [np.cos(dFL), np.sin(dFL), -(t/2)*np.cos(dFL) + L*np.sin(dFL)],
        [np.cos(dFR), np.sin(dFR), +(t/2)*np.cos(dFR) + L*np.sin(dFR)],
        [1.0,         0.0,         -(t/2)],
        [1.0,         0.0,         +(t/2)],
    ])

def B_matrix(th, dFL, dFR, L, t, r):
    A = A_matrix(dFL, dFR, L, t)
    return (1.0/r) * TG2R(th).T @ A.T    # 3x4

def damped_least_squares(B, y, lam):
    # argmin_tau ||B tau - y||^2 + lam ||tau||^2
    # (B^T B + lam I) tau = B^T y
    BtB = B.T @ B
    return np.linalg.solve(BtB + lam*np.eye(B.shape[1]), B.T @ y)

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def box_corners(center_xy, heading, length, width):
    l, w = length/2.0, width/2.0
    local = np.array([[+l,+w],[+l,-w],[-l,-w],[-l,+w],[+l,+w]])
    return (R2(heading) @ local.T).T + center_xy

def body_outline(center_xy, heading):
    Lb = L + body_overhang_front + body_overhang_rear
    x_center = (L + body_overhang_front - body_overhang_rear)/2.0
    body_center_R = np.array([x_center, 0.0])
    return box_corners(center_xy + (R2(heading) @ body_center_R),
                       heading, Lb, t)

# ---------------- Reference S-path ----------------
N = int(T/dt)
tt = np.linspace(0, T, N)
q_ref    = np.zeros((N,3))
qdot_ref = np.zeros((N,3))
qdd_ref  = np.zeros((N,3))

th = 0.0
for k in range(N):
    tk  = tt[k]
    dlt = delta_max*np.sin(2*np.pi*tk/S_period)
    dlt_dot = delta_max*(2*np.pi/S_period)*np.cos(2*np.pi*tk/S_period)
    thdot = v_x*np.tan(dlt) / L
    thdd  = (v_x/L)*(1/np.cos(dlt)**2)*dlt_dot
    if k>0:
        th = q_ref[k-1,2] + thdot*dt
    q_ref[k]    = [q_ref[k-1,0] + np.cos(th)*v_x*dt if k>0 else 0.0,
                   q_ref[k-1,1] + np.sin(th)*v_x*dt if k>0 else 0.0,
                   th]
    qdot_ref[k] = [np.cos(th)*v_x, np.sin(th)*v_x, thdot]
    qdd_ref[k]  = [-np.sin(th)*thdot*v_x,
                    np.cos(th)*thdot*v_x,
                    thdd]

# ---------------- Plant/dynamics simulation ----------------
M = np.diag([m, m, Iz])

q    = q_ref[0].copy()
qdot = qdot_ref[0].copy()

q_log     = np.zeros((N,3))
qdot_log  = np.zeros((N,3))
tau_log   = np.zeros((N,4))
vt_log    = np.zeros((N,4))
deltaFL_v = np.zeros(N)
deltaFR_v = np.zeros(N)

for k in range(N):
    tk = tt[k]
    # steering used to define input directions
    dlt = delta_max*np.sin(2*np.pi*tk/S_period)
    dFL, dFR = ackermann_split(dlt, L, t)
    deltaFL_v[k], deltaFR_v[k] = dFL, dFR

    # input matrix
    B = B_matrix(q[2], dFL, dFR, L, t, r)  # 3x4

    # external viscous wrench (global)
    Dg = np.diag([d_x, d_y, d_th])
    Wext = - Dg @ qdot

    # PD tracking in generalized coords (global)
    e    = q_ref[k]    - q
    e[2] = wrap_pi(e[2])
    edot = qdot_ref[k] - qdot
    wrench_des = M @ qdd_ref[k] + Kp @ e + Kd @ edot + Wext

    # damped least-squares torque solve + saturation
    tau = damped_least_squares(B, wrench_des, lambda_damp)
    tau = np.clip(tau, -tau_max, tau_max)

    # integrate dynamics: qdd = M^{-1}(B tau + Wext)
    qdd = np.linalg.solve(M, B @ tau + Wext)
    qdot += qdd * dt
    q    += qdot * dt

    # log
    q_log[k]    = q
    qdot_log[k] = qdot
    tau_log[k]  = tau

    # wheel longitudinal speeds (for plots)
    A = A_matrix(dFL, dFR, L, t)
    vR = TG2R(q[2]) @ qdot
    vt_log[k] = A @ vR

# ---------------- One-window Visualization ----------------
fig, axs = plt.subplots(2, 2, figsize=(13, 8))
ax_anim, ax_xyz, ax_vt, ax_tau = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

# Animation axes
ax_anim.set_aspect('equal')
ax_anim.set_title("Ackermann Dynamics Tracking S-Path (w/ steering wheels)")
ax_anim.set_xlabel("X [m]"); ax_anim.set_ylabel("Y [m]")
pad = 0.6
xmin, xmax = min(q_log[:,0].min(), q_ref[:,0].min())-pad, max(q_log[:,0].max(), q_ref[:,0].max())+pad
ymin, ymax = min(q_log[:,1].min(), q_ref[:,1].min())-pad, max(q_log[:,1].max(), q_ref[:,1].max())+pad
ax_anim.plot(q_ref[:,0], q_ref[:,1], 'k--', lw=1.2, label='ref path')
ax_anim.set_xlim(xmin, xmax); ax_anim.set_ylim(ymin, ymax)

(trace,) = ax_anim.plot([], [], 'b-', lw=2, label='actual path')
(body_line,) = ax_anim.plot([], [], 'k-', lw=2)
(FL_w_line,) = ax_anim.plot([], [], 'r-', lw=3, label='FL')
(FR_w_line,) = ax_anim.plot([], [], 'g-', lw=3, label='FR')
(RL_w_line,) = ax_anim.plot([], [], 'b-', lw=3, label='RL')
(RR_w_line,) = ax_anim.plot([], [], 'm-', lw=3, label='RR')
ax_anim.legend(loc='upper right')

def update_anim(i):
    pos = q_log[i,:2]; heading = q_log[i,2]
    body_poly = body_outline(pos, heading)
    body_line.set_data(body_poly[:,0], body_poly[:,1])

    Rg = R2(heading)
    cFL = pos + (Rg @ p_FL); cFR = pos + (Rg @ p_FR)
    cRL = pos + (Rg @ p_RL); cRR = pos + (Rg @ p_RR)

    FL_poly = box_corners(cFL, heading + deltaFL_v[i], wheel_len, wheel_wid)
    FR_poly = box_corners(cFR, heading + deltaFR_v[i], wheel_len, wheel_wid)
    RL_poly = box_corners(cRL, heading,                    wheel_len, wheel_wid)
    RR_poly = box_corners(cRR, heading,                    wheel_len, wheel_wid)

    FL_w_line.set_data(FL_poly[:,0], FL_poly[:,1])
    FR_w_line.set_data(FR_poly[:,0], FR_poly[:,1])
    RL_w_line.set_data(RL_poly[:,0], RL_poly[:,1])
    RR_w_line.set_data(RR_poly[:,0], RR_poly[:,1])

    trace.set_data(q_log[:i+1,0], q_log[:i+1,1])
    return trace, body_line, FL_w_line, FR_w_line, RL_w_line, RR_w_line

ani = FuncAnimation(fig, update_anim, frames=N, interval=dt*1000, blit=False, repeat=False)

# State tracking
ax_xyz.set_title("State Tracking")
ax_xyz.plot(tt, q_ref[:,0], 'k--', label='x_ref'); ax_xyz.plot(tt, q_log[:,0], 'b-',  label='x')
ax_xyz.plot(tt, q_ref[:,1], 'k-.', label='y_ref'); ax_xyz.plot(tt, q_log[:,1], 'g-',  label='y')
ax_xyz.plot(tt, q_ref[:,2], 'k:',  label='theta_ref'); ax_xyz.plot(tt, q_log[:,2], 'r-',label='theta')
ax_xyz.set_xlabel("Time [s]"); ax_xyz.legend(ncols=2, fontsize=9)

# Wheel speeds and torques
ax_vt.set_title("Wheel Longitudinal Speeds $v_t$")
ax_vt.plot(tt, vt_log[:,0], 'r-', label='FL')
ax_vt.plot(tt, vt_log[:,1], 'g-', label='FR')
ax_vt.plot(tt, vt_log[:,2], 'b-', label='RL')
ax_vt.plot(tt, vt_log[:,3], 'm-', label='RR')
ax_vt.set_xlabel("Time [s]"); ax_vt.set_ylabel("$v_t$ [m/s]"); ax_vt.legend()

ax_tau.set_title("Wheel Torques $\\tau_w$ (clipped)")
ax_tau.plot(tt, tau_log[:,0], 'r-', label='FL')
ax_tau.plot(tt, tau_log[:,1], 'g-', label='FR')
ax_tau.plot(tt, tau_log[:,2], 'b-', label='RL')
ax_tau.plot(tt, tau_log[:,3], 'm-', label='RR')
ax_tau.set_xlabel("Time [s]"); ax_tau.set_ylabel("Torque [N·m]"); ax_tau.legend()

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
    # To save animation (needs ffmpeg):
    # ani.save("dynamics_sim.mp4", dpi=120, writer="ffmpeg")