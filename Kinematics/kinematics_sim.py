import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- Parameters ----------------
L, t, r_wheel = 0.35, 0.28, 0.06  # wheelbase, track width, wheel radius
dt, T = 0.02, 16.0
v_cmd = 0.8
delta_max = np.deg2rad(40.0)

# Visual geometry (purely for drawing)
body_overhang_rear = 0.05
body_overhang_front = 0.05
wheel_len = 0.16
wheel_wid = 0.06

# Wheel centers in robot frame (origin at rear-axle midpoint, x forward, y left)
p_FL = np.array([L,  t/2])
p_FR = np.array([L, -t/2])
p_RL = np.array([0., t/2])
p_RR = np.array([0.,-t/2])

wheel_pts = [p_FL, p_FR, p_RL, p_RR]
wheel_names = ['FL', 'FR', 'RL', 'RR']

def R2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def ackermann_split(delta, L, t):
    """Split bicycle steering angle into per-wheel FL/FR (Ackermann geometry)."""
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
    l = length/2.0
    w = width/2.0
    box_local = np.array([[+l, +w],
                          [+l, -w],
                          [-l, -w],
                          [-l, +w],
                          [+l, +w]])
    Rg = R2(heading)
    return (Rg @ box_local.T).T + center_xy

def body_outline(center_xy, heading):
    Lb = L + body_overhang_front + body_overhang_rear
    x_center = (L + body_overhang_front - body_overhang_rear) / 2.0
    body_center_R = np.array([x_center, 0.0])
    return box_corners(center_xy + (R2(heading) @ body_center_R),
                       heading, Lb, t)

# ---------------- Square-path steering schedule ----------------
delta_sq = delta_max
omega_turn_guess = v_cmd * np.tan(delta_sq) / L
T_turn = (np.pi/2.0) / max(abs(omega_turn_guess), 1e-6)

T_side = T / 4.0
T_straight = max(T_side - T_turn, 0.1)  # guard

def steering_square(t_now):
    if t_now >= T:
        return 0.0
    side_idx = int(t_now // T_side)
    phase = t_now - side_idx * T_side
    return 0.0 if phase < T_straight else delta_sq

# ---------------- Double-track kinematics solver ----------------
def solve_body_twist_from_wheels(v_cmd, dFL, dFR, L, t):
    """
    Solve for body twist [vx, vy, omega] at rear-axle midpoint using:
      1) Rear wheels enforce v_y = 0 at their contact frames (since delta=0)
         => for RL: vy + omega*x_RL = 0 (x_RL=0) -> vy = 0
         => for RR: vy + omega*x_RR = 0 (x_RR=0) -> vy = 0
         This pins vy to ~0 in ideal no-slip.
      2) Enforce commanded forward speed along body x: vx = v_cmd
      3) Use front wheel no-lateral-slip constraints:
         0 = -vx*sin(d_i) + vy*cos(d_i) + omega*x_i*cos(d_i)
                               + omega*y_i*sin(d_i)  (derived from rotating v_i)
    We build linear constraints A*[vx, vy, omega] = b and least-squares solve.

    Note: This is still kinematic (no tire slip dynamics), but it's double-track
    because each wheel contributes a constraint.
    """
    # Wheel positions (x,y) in body frame
    xFL, yFL = L,  t/2
    xFR, yFR = L, -t/2
    xRL, yRL = 0.,  t/2
    xRR, yRR = 0., -t/2

    A = []
    b = []

    # (a) Command vx = v_cmd
    A.append([1.0, 0.0, 0.0])
    b.append(v_cmd)

    # (b) Rear no-lateral-slip constraints (delta=0):
    # wheel lateral velocity in wheel frame for rear wheels is:
    # v_y_wheel = vy + omega*x (since delta=0 and wheel x along body x)
    A.append([0.0, 1.0, xRL])
    b.append(0.0)
    A.append([0.0, 1.0, xRR])
    b.append(0.0)

    # (c) Front no-lateral-slip constraints (delta = dFL/dFR):
    # For wheel at (x,y), body point velocity: v_i = [vx - omega*y, vy + omega*x]
    # Wheel-frame lateral component = [-sin d, cos d] · v_i = 0
    # => -sin d*(vx - omega*y) + cos d*(vy + omega*x) = 0
    # => (-sin d)*vx + (cos d)*vy + omega*( sin d*y + cos d*x ) = 0
    def add_front_constraint(d, xw, yw):
        A.append([-np.sin(d), np.cos(d), (np.sin(d)*yw + np.cos(d)*xw)])
        b.append(0.0)

    add_front_constraint(dFL, xFL, yFL)
    add_front_constraint(dFR, xFR, yFR)

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # least squares
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    vx, vy, omega = sol
    return vx, vy, omega

def wheel_longitudinal_speed(vx, vy, omega, xw, yw, delta):
    """
    Compute wheel longitudinal speed along its rolling direction:
      v_i = [vx - omega*yw, vy + omega*xw]
      v_long = [cos d, sin d] · v_i
    """
    vix = vx - omega*yw
    viy = vy + omega*xw
    return np.cos(delta)*vix + np.sin(delta)*viy

# ---------------- Simulate ----------------
N = int(T/dt)
tt = np.linspace(0, T, N)

x = np.zeros(N)
y = np.zeros(N)
th = np.zeros(N)

vx_body = np.zeros(N)
vy_body = np.zeros(N)
omega_body = np.zeros(N)

v_t = np.zeros((N, 4))
omega_w = np.zeros((N, 4))

P_FL = np.zeros((N, 2)); P_FR = np.zeros((N, 2))
P_RL = np.zeros((N, 2)); P_RR = np.zeros((N, 2))

delta_FL = np.zeros(N); delta_FR = np.zeros(N)
delta_hist = np.zeros(N)

for k in range(N-1):
    delta = steering_square(tt[k])
    delta_hist[k] = delta

    dFL, dFR = ackermann_split(delta, L, t)
    delta_FL[k], delta_FR[k] = dFL, dFR

    # --- Double-track kinematics solve (vx, vy, omega) ---
    vxk, vyk, omegak = solve_body_twist_from_wheels(v_cmd, dFL, dFR, L, t)
    vx_body[k], vy_body[k], omega_body[k] = vxk, vyk, omegak

    # --- Integrate pose in world ---
    # world velocity = R(th) * [vx, vy]
    x[k+1]  = x[k] + ( np.cos(th[k])*vxk - np.sin(th[k])*vyk ) * dt
    y[k+1]  = y[k] + ( np.sin(th[k])*vxk + np.cos(th[k])*vyk ) * dt
    th[k+1] = wrap_pi(th[k] + omegak * dt)

    # --- Wheel longitudinal speeds and angular speeds ---
    # wheel deltas: FL, FR, RL=0, RR=0
    deltas = [dFL, dFR, 0.0, 0.0]
    pts = [(L, t/2), (L, -t/2), (0.0, t/2), (0.0, -t/2)]
    for i, ((xw, yw), d) in enumerate(zip(pts, deltas)):
        vlong = wheel_longitudinal_speed(vxk, vyk, omegak, xw, yw, d)
        v_t[k, i] = vlong
        omega_w[k, i] = vlong / r_wheel

    # --- Wheel centers in world (for plotting) ---
    Rg = R2(th[k])
    pos = np.array([x[k], y[k]])
    P_FL[k,:] = pos + (Rg @ p_FL)
    P_FR[k,:] = pos + (Rg @ p_FR)
    P_RL[k,:] = pos + (Rg @ p_RL)
    P_RR[k,:] = pos + (Rg @ p_RR)

# final fill
delta = steering_square(tt[-1])
delta_hist[-1] = delta
delta_FL[-1], delta_FR[-1] = ackermann_split(delta, L, t)

vxk, vyk, omegak = solve_body_twist_from_wheels(v_cmd, delta_FL[-1], delta_FR[-1], L, t)
vx_body[-1], vy_body[-1], omega_body[-1] = vxk, vyk, omegak

Rg = R2(th[-1]); pos = np.array([x[-1], y[-1]])
P_FL[-1,:] = pos + (Rg @ p_FL)
P_FR[-1,:] = pos + (Rg @ p_FR)
P_RL[-1,:] = pos + (Rg @ p_RL)
P_RR[-1,:] = pos + (Rg @ p_RR)

# ---------------- Figure 1: Animation ----------------
fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
ax_anim.set_aspect('equal')
ax_anim.set_title("Ackermann Kinematics (Double-Track Constraints)")
ax_anim.set_xlabel("X [m]")
ax_anim.set_ylabel("Y [m]")
pad = 0.6
ax_anim.set_xlim(x.min()-pad, x.max()+pad)
ax_anim.set_ylim(y.min()-pad, y.max()+pad)

(trace,) = ax_anim.plot([], [], 'b-', lw=2, label='Path')
(body_line,) = ax_anim.plot([], [], 'k-', lw=2)

(FL_wheel_line,) = ax_anim.plot([], [], 'r-', lw=3, label='FL')
(FR_wheel_line,) = ax_anim.plot([], [], 'g-', lw=3, label='FR')
(RL_wheel_line,) = ax_anim.plot([], [], 'b-', lw=3, label='RL')
(RR_wheel_line,) = ax_anim.plot([], [], 'm-', lw=3, label='RR')
ax_anim.legend(loc='upper right')

def update(frame):
    pos = np.array([x[frame], y[frame]])
    heading = th[frame]

    body_poly = body_outline(pos, heading)
    body_line.set_data(body_poly[:,0], body_poly[:,1])

    Rg = R2(heading)
    cFL = pos + (Rg @ p_FL)
    cFR = pos + (Rg @ p_FR)
    cRL = pos + (Rg @ p_RL)
    cRR = pos + (Rg @ p_RR)

    FL_poly = box_corners(cFL, heading + delta_FL[frame], wheel_len, wheel_wid)
    FR_poly = box_corners(cFR, heading + delta_FR[frame], wheel_len, wheel_wid)
    RL_poly = box_corners(cRL, heading, wheel_len, wheel_wid)
    RR_poly = box_corners(cRR, heading, wheel_len, wheel_wid)

    FL_wheel_line.set_data(FL_poly[:,0], FL_poly[:,1])
    FR_wheel_line.set_data(FR_poly[:,0], FR_poly[:,1])
    RL_wheel_line.set_data(RL_poly[:,0], RL_poly[:,1])
    RR_wheel_line.set_data(RR_poly[:,0], RR_poly[:,1])

    trace.set_data(x[:frame+1], y[:frame+1])

    return (trace, body_line, FL_wheel_line, FR_wheel_line, RL_wheel_line, RR_wheel_line)

ani = FuncAnimation(fig_anim, update, frames=N, interval=dt*1000,
                    blit=False, repeat=False)

# ---------------- Figure 2: Individual wheel positions ----------------
P_all = [P_FL, P_FR, P_RL, P_RR]
fig_pos, axs_pos = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
fig_pos.suptitle("Wheel Positions (X and Y)")
for i, ax in enumerate(axs_pos):
    ax.plot(tt, P_all[i][:, 0], label='x')
    ax.plot(tt, P_all[i][:, 1], '--', label='y')
    ax.set_ylabel(wheel_names[i])
    ax.grid(True)
    ax.legend()
axs_pos[-1].set_xlabel("Time [s]")
fig_pos.tight_layout(rect=[0, 0, 1, 0.96])

# ---------------- Figure 3: Individual wheel longitudinal speeds v_t ----------------
fig_vt, axs_vt = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
fig_vt.suptitle("Wheel Longitudinal Speeds $v_t$")
for i, ax in enumerate(axs_vt):
    ax.plot(tt, v_t[:, i])
    ax.set_ylabel(f"{wheel_names[i]} [m/s]")
    ax.grid(True)
axs_vt[-1].set_xlabel("Time [s]")
fig_vt.tight_layout(rect=[0, 0, 1, 0.96])

# ---------------- Figure 4: Individual wheel angular speeds omega_w ----------------
fig_om, axs_om = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
fig_om.suptitle("Wheel Angular Speeds $\\omega_w$")
for i, ax in enumerate(axs_om):
    ax.plot(tt, omega_w[:, i])
    ax.set_ylabel(f"{wheel_names[i]} [rad/s]")
    ax.grid(True)
axs_om[-1].set_xlabel("Time [s]")
fig_om.tight_layout(rect=[0, 0, 1, 0.96])

if __name__ == "__main__":
    plt.show()
    # To save animation (needs ffmpeg):
    # ani.save("kinematics_double_track.mp4", dpi=120, writer="ffmpeg")
