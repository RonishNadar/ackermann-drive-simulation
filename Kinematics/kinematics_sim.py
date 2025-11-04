import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- Parameters ----------------
L, t, r = 0.35, 0.28, 0.06  # wheelbase, track width, wheel radius
dt, T = 0.02, 16.0
v_x = 0.8
delta_max = np.deg2rad(18.0)
S_period = 8.0

# Visual geometry (purely for drawing)
body_overhang_rear = 0.05      # rear overhang (drawing only)
body_overhang_front = 0.05     # front overhang (drawing only)
wheel_len = 0.16               # wheel rectangle length (along rolling direction)
wheel_wid = 0.06               # wheel rectangle width  (across rolling direction)

# Wheel centers in robot frame (origin at rear-axle midpoint, x forward, y left)
p_FL = np.array([L,  t/2])
p_FR = np.array([L, -t/2])
p_RL = np.array([0., t/2])
p_RR = np.array([0.,-t/2])

def R2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def ackermann_split(delta, L, t):
    """Split bicycle steering angle into per-wheel FL/FR (Ackermann geometry)."""
    td = np.tan(delta)
    if abs(td) < 1e-8:
        return delta, delta
    Rmid = L / td
    if abs(Rmid) <= t/2 + 1e-6:  # avoid singular geometry
        Rmid = np.sign(Rmid) * (t/2 + 1e-6)
    dFL = np.arctan2(L, Rmid - t/2)
    dFR = np.arctan2(L, Rmid + t/2)
    return dFL, dFR

def box_corners(center_xy, heading, length, width):
    """
    Return Nx2 closed polygon of a rectangle centered at 'center_xy',
    with local +x along 'heading', dimensions (length, width).
    """
    l = length/2.0
    w = width/2.0
    # local box corners (closed)
    box_local = np.array([[+l, +w],
                          [+l, -w],
                          [-l, -w],
                          [-l, +w],
                          [+l, +w]])
    Rg = R2(heading)
    return (Rg @ box_local.T).T + center_xy

def body_outline(center_xy, heading):
    """
    Draw the chassis as a rectangle extending from -rear_overhang to L+front_overhang.
    Centered laterally at y=0 in the body frame, then rotated+translated.
    """
    Lb = L + body_overhang_front + body_overhang_rear
    # Put geometric center of the body footprint at (x = L/2 + (front - rear)/2)
    x_center = (L + body_overhang_front - body_overhang_rear) / 2.0
    body_center_R = np.array([x_center, 0.0])
    return box_corners(center_xy + (R2(heading) @ body_center_R),
                       heading, Lb, t)

# ---------------- Simulate ----------------
N = int(T/dt)
tt = np.linspace(0, T, N)
x = np.zeros(N); y = np.zeros(N); th = np.zeros(N)
v_t = np.zeros((N, 4)); omega_w = np.zeros((N, 4))
P_FL = np.zeros((N, 2)); P_FR = np.zeros((N, 2))
P_RL = np.zeros((N, 2)); P_RR = np.zeros((N, 2))
delta_FL = np.zeros(N); delta_FR = np.zeros(N)

for k in range(N-1):
    # S-shaped bicycle steering
    delta = delta_max * np.sin(2*np.pi * tt[k] / S_period)
    dFL, dFR = ackermann_split(delta, L, t)
    delta_FL[k], delta_FR[k] = dFL, dFR

    # Kinematics (Vy=0)
    omega = v_x * np.tan(delta) / L
    x[k+1] = x[k] + np.cos(th[k]) * v_x * dt
    y[k+1] = y[k] + np.sin(th[k]) * v_x * dt
    th[k+1] = th[k] + omega * dt

    # Wheel speeds (longitudinal and angular)
    A = np.array([
        [np.cos(dFL), np.sin(dFL), -(t/2)*np.cos(dFL) + L*np.sin(dFL)],
        [np.cos(dFR), np.sin(dFR), +(t/2)*np.cos(dFR) + L*np.sin(dFR)],
        [1.0,         0.0,         -(t/2)],
        [1.0,         0.0,         +(t/2)],
    ])
    vR = np.array([v_x, 0.0, omega])
    w = (1.0/r) * (A @ vR)
    omega_w[k,:] = w
    v_t[k,:] = r*w

    # Wheel centers in world
    Rg = R2(th[k])
    pos = np.array([x[k], y[k]])
    P_FL[k,:] = pos + (Rg @ p_FL)
    P_FR[k,:] = pos + (Rg @ p_FR)
    P_RL[k,:] = pos + (Rg @ p_RL)
    P_RR[k,:] = pos + (Rg @ p_RR)

# Final step fill
delta = delta_max * np.sin(2*np.pi * tt[-1] / S_period)
delta_FL[-1], delta_FR[-1] = ackermann_split(delta, L, t)
Rg = R2(th[-1]); pos = np.array([x[-1], y[-1]])
P_FL[-1,:] = pos + (Rg @ p_FL); P_FR[-1,:] = pos + (Rg @ p_FR)
P_RL[-1,:] = pos + (Rg @ p_RL); P_RR[-1,:] = pos + (Rg @ p_RR)

# ---------------- Figure Layout (single window) ----------------
fig, axs = plt.subplots(2, 2, figsize=(13, 8))
ax_anim, ax_pos, ax_vt, ax_om = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

# Animation axes
ax_anim.set_aspect('equal')
ax_anim.set_title("Ackermann S-shaped Motion (with steering wheels)")
ax_anim.set_xlabel("X [m]"); ax_anim.set_ylabel("Y [m]")
pad = 0.6
ax_anim.set_xlim(x.min()-pad, x.max()+pad)
ax_anim.set_ylim(y.min()-pad, y.max()+pad)

# Path trace
(trace,) = ax_anim.plot([], [], 'b-', lw=2, label='Path')

# Chassis outline (as a line we update each frame)
(body_line,) = ax_anim.plot([], [], 'k-', lw=2)

# Wheel outlines (as separate polylines)
(FL_wheel_line,) = ax_anim.plot([], [], 'r-', lw=3, label='FL')
(FR_wheel_line,) = ax_anim.plot([], [], 'g-', lw=3, label='FR')
(RL_wheel_line,) = ax_anim.plot([], [], 'b-', lw=3, label='RL')
(RR_wheel_line,) = ax_anim.plot([], [], 'm-', lw=3, label='RR')
ax_anim.legend(loc='upper right')

# Static time plots
ax_pos.set_title("Wheel Positions (X and Y)")
ax_pos.set_xlabel("Time [s]"); ax_pos.set_ylabel("Position [m]")
ax_pos.plot(tt, P_FL[:,0], 'r-', label='FL_x'); ax_pos.plot(tt, P_FL[:,1], 'r--', label='FL_y')
ax_pos.plot(tt, P_FR[:,0], 'g-', label='FR_x'); ax_pos.plot(tt, P_FR[:,1], 'g--', label='FR_y')
ax_pos.plot(tt, P_RL[:,0], 'b-', label='RL_x'); ax_pos.plot(tt, P_RL[:,1], 'b--', label='RL_y')
ax_pos.plot(tt, P_RR[:,0], 'm-', label='RR_x'); ax_pos.plot(tt, P_RR[:,1], 'm--', label='RR_y')
ax_pos.legend(ncols=2, fontsize=9)

ax_vt.set_title("Wheel Longitudinal Speeds $v_t$")
ax_vt.set_xlabel("Time [s]"); ax_vt.set_ylabel("$v_t$ [m/s]")
ax_vt.plot(tt, v_t[:,0], 'r-', label='FL')
ax_vt.plot(tt, v_t[:,1], 'g-', label='FR')
ax_vt.plot(tt, v_t[:,2], 'b-', label='RL')
ax_vt.plot(tt, v_t[:,3], 'm-', label='RR')
ax_vt.legend()

ax_om.set_title("Wheel Angular Speeds $\\omega_w$")
ax_om.set_xlabel("Time [s]"); ax_om.set_ylabel("$\\omega$ [rad/s]")
ax_om.plot(tt, omega_w[:,0], 'r-', label='FL')
ax_om.plot(tt, omega_w[:,1], 'g-', label='FR')
ax_om.plot(tt, omega_w[:,2], 'b-', label='RL')
ax_om.plot(tt, omega_w[:,3], 'm-', label='RR')
ax_om.legend()

def update(frame):
    pos = np.array([x[frame], y[frame]])
    heading = th[frame]

    # Body outline (rectangle)
    body_poly = body_outline(pos, heading)
    body_line.set_data(body_poly[:,0], body_poly[:,1])

    # Wheel headings:
    #   Rear wheels aligned with body
    #   Front wheels aligned with body + steering
    Rg = R2(heading)
    cFL = pos + (Rg @ p_FL)
    cFR = pos + (Rg @ p_FR)
    cRL = pos + (Rg @ p_RL)
    cRR = pos + (Rg @ p_RR)

    FL_poly = box_corners(cFL, heading + delta_FL[frame], wheel_len, wheel_wid)
    FR_poly = box_corners(cFR, heading + delta_FR[frame], wheel_len, wheel_wid)
    RL_poly = box_corners(cRL, heading,                    wheel_len, wheel_wid)
    RR_poly = box_corners(cRR, heading,                    wheel_len, wheel_wid)

    FL_wheel_line.set_data(FL_poly[:,0], FL_poly[:,1])
    FR_wheel_line.set_data(FR_poly[:,0], FR_poly[:,1])
    RL_wheel_line.set_data(RL_poly[:,0], RL_poly[:,1])
    RR_wheel_line.set_data(RR_poly[:,0], RR_poly[:,1])

    # Trace of COM path
    trace.set_data(x[:frame+1], y[:frame+1])

    return (trace, body_line,
            FL_wheel_line, FR_wheel_line, RL_wheel_line, RR_wheel_line)

ani = FuncAnimation(fig, update, frames=N, interval=dt*1000,
                    blit=False, repeat=False)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
    # To save animation (needs ffmpeg):
    # ani.save("kinematics_sim.mp4", dpi=120, writer="ffmpeg")
