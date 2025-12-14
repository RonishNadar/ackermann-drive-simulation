import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# USER OPTIONS
# ============================================================
SHOW_B_MATRICES   = False      # plot selected Bx/By entries
ADD_DISTURBANCE   = False      # matched wrench disturbance tau_d added to body wrench
ADD_SENSOR_NOISE  = False     # noise on measured lookahead point z
LOAD_SWITCH       = True      # true plant mass/Iz changes (truck loaded/unloaded)

np.random.seed(0)

# ============================================================
# Vehicle geometry & nominal params (controller model)
# ============================================================
L = 0.35
t = 0.28
lf = 0.18
lr = L - lf

m_nom  = 8.0
Iz_nom = 0.45
g = 9.81

# Tire model params (linear + saturation)
Cf = 250.0
Cr = 280.0
mu = 0.9
vx_min = 0.05

# crude drag-like term
Cdrag = 0.6

# ============================================================
# Simulation
# ============================================================
dt, T = 0.01, 16.0
N = int(T/dt)
tt = np.linspace(0, T, N)

# ============================================================
# Feasible square reference: straight + rounded 90deg turns
# ============================================================
side_len   = 1.6
v_ref      = 0.8
delta_turn = np.deg2rad(20.0)
delta_max  = np.deg2rad(25.0)
delta_rate = np.deg2rad(90.0)  # steering slew [rad/s]

r_turn = v_ref * np.tan(delta_turn) / L
T_turn = (np.pi/2) / max(abs(r_turn), 1e-6)
T_straight = side_len / max(v_ref, 1e-9)

T_ramp = abs(delta_turn) / max(delta_rate, 1e-9)
T_plateau = max(T_turn - 2*T_ramp, 0.0)
T_side = T_straight + (2*T_ramp + T_plateau)
T_total_ref = 4 * T_side

# ============================================================
# Tracking output: lookahead point z = [x,y] + d*[cos psi, sin psi]
# ============================================================
d_look = 0.25

# SMC outer-loop on (z, z_dot)
Lambda = 1.5 * np.eye(2)
Ksm    = 1.2 * np.eye(2)
phi    = 0.05

# choose how to pick nu_dot from z_ddot (underdetermined -> regularize rdot)
w_rdot = 2.0  # larger => smoother yaw accel

# ============================================================
# Truck load switch (true plant parameters)
# ============================================================
m_loaded  = 12.0
Iz_loaded = 0.70
t_load_on  = 6.0
t_load_off = 12.0

# ============================================================
# Helpers
# ============================================================
def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def R2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

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

def clamp_friction(Fx, Fy, Fz, mu):
    Fmax = mu * Fz
    mag = np.hypot(Fx, Fy)
    if mag <= Fmax or mag < 1e-12:
        return Fx, Fy
    s = Fmax / mag
    return Fx*s, Fy*s

# ============================================================
# Wheel positions in body frame (origin at CG)
# ============================================================
p_FL = np.array([+lf, +t/2])
p_FR = np.array([+lf, -t/2])
p_RL = np.array([-lr, +t/2])
p_RR = np.array([-lr, -t/2])
wheel_pos = np.stack([p_FL, p_FR, p_RL, p_RR], axis=0)
wheel_names = ["FL","FR","RL","RR"]

# Drawing geometry
body_overhang_rear  = 0.05
body_overhang_front = 0.05
wheel_len, wheel_wid = 0.16, 0.06

def box_corners(center_xy, heading, length, width):
    l, w = length/2.0, width/2.0
    local = np.array([[+l,+w],[+l,-w],[-l,-w],[-l,+w],[+l,+w]])
    return (R2(heading) @ local.T).T + center_xy

def body_outline(center_xy, heading):
    Lb = lf + lr + body_overhang_front + body_overhang_rear
    x_center = (lf + body_overhang_front - (lr + body_overhang_rear))/2.0
    body_center_R = np.array([x_center, 0.0])
    return box_corners(center_xy + (R2(heading) @ body_center_R), heading, Lb, t)

# ============================================================
# Dynamics matrices (BODY coords)
# ============================================================
def M_matrix(m, Iz):
    return np.diag([m, m, Iz])

def C_matrix(m, nu):
    vx, vy, r = nu
    return np.array([
        [0.0,   m*r, 0.0],
        [-m*r,  0.0, 0.0],
        [0.0,   0.0, 0.0]
    ])

def H_i(xi, yi):
    return np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-yi, xi]
    ])

def Bx_By(wheel_heading):
    """
    tau = Bx Fx_vec + By Fy_vec
    Fx_vec, Fy_vec are wheel-frame forces (4,)
    """
    Bx = np.zeros((3,4))
    By = np.zeros((3,4))
    for i in range(4):
        xi, yi = wheel_pos[i]
        Hi = H_i(xi, yi)
        di = wheel_heading[i]
        a = np.array([np.cos(di), np.sin(di)])     # wheel +x (longitudinal) in body
        b = np.array([-np.sin(di), np.cos(di)])    # wheel +y (lateral) in body
        Bx[:, i] = Hi @ a
        By[:, i] = Hi @ b
    return Bx, By

def static_Fz(m):
    Fz_front_axle = m * g * (lr / L)
    Fz_rear_axle  = m * g * (lf / L)
    return np.array([
        0.5 * Fz_front_axle,
        0.5 * Fz_front_axle,
        0.5 * Fz_rear_axle,
        0.5 * Fz_rear_axle,
    ])

# ============================================================
# Reference generator: feasible square (rate-limited steering)
# ============================================================
def delta_profile_one_side(t_side):
    if t_side < T_straight:
        return 0.0
    tau = t_side - T_straight

    # ramp up
    if tau < T_ramp:
        return np.sign(delta_turn) * min(abs(delta_turn), delta_rate * tau)

    tau -= T_ramp
    # plateau
    if tau < T_plateau:
        return delta_turn

    tau -= T_plateau
    # ramp down
    if tau < T_ramp:
        return np.sign(delta_turn) * max(0.0, abs(delta_turn) - delta_rate * tau)

    return 0.0

def reference_square(tt):
    xr = np.zeros_like(tt)
    yr = np.zeros_like(tt)
    psir = np.zeros_like(tt)
    deltar = np.zeros_like(tt)
    rref = np.zeros_like(tt)

    for k in range(len(tt)-1):
        tnow = tt[k]
        t_mod = tnow % T_total_ref
        t_side = t_mod - int(t_mod // T_side) * T_side

        delta = np.clip(delta_profile_one_side(t_side), -delta_max, +delta_max)
        deltar[k] = delta
        r = v_ref * np.tan(delta) / L
        rref[k] = r

        xr[k+1] = xr[k] + v_ref*np.cos(psir[k]) * dt
        yr[k+1] = yr[k] + v_ref*np.sin(psir[k]) * dt
        psir[k+1] = wrap_pi(psir[k] + r * dt)

    deltar[-1] = deltar[-2]
    rref[-1] = rref[-2]
    return xr, yr, psir, deltar, rref

xr, yr, psir, deltar, rref = reference_square(tt)

# Reference lookahead point and its derivative (finite diff)
zr = np.stack([xr + d_look*np.cos(psir), yr + d_look*np.sin(psir)], axis=1)
zr_d = np.gradient(zr, dt, axis=0)
zr_dd = np.gradient(zr_d, dt, axis=0)

# ============================================================
# Disturbances / noise
# ============================================================
def wrench_disturbance(tnow):
    # matched disturbance in body wrench tau = [Fx, Fy, Mz]
    Fx_d = 2.0*np.sin(0.7*tnow)
    Fy_d = 1.5*np.cos(0.4*tnow)
    Mz_d = 0.4*np.sin(0.3*tnow)
    return np.array([Fx_d, Fy_d, Mz_d])

def sensor_noise():
    return 0.01*np.random.randn(2)

# ============================================================
# Initialize states
# ============================================================
x = 0.0
y = 0.0
psi = 0.0
vx = v_ref
vy = 0.0
r_yaw = 0.0

# Logs
x_log = np.zeros(N); y_log = np.zeros(N); psi_log = np.zeros(N)
vx_log = np.zeros(N); vy_log = np.zeros(N); r_log = np.zeros(N)
delta_log = np.zeros(N)
Fx_w_log = np.zeros((N,4))
Fy_w_log = np.zeros((N,4))
e_log = np.zeros((N,2))
s_log = np.zeros((N,2))

# selected B entries for visualization
Bx00 = np.zeros(N); By00 = np.zeros(N)   # (row0,col0) = mapping wheel FL
Bx20 = np.zeros(N); By20 = np.zeros(N)   # yaw moment contribution of FL

# ============================================================
# Simulation loop
# ============================================================
for k, tnow in enumerate(tt):
    # True plant parameters (load/unload)
    if LOAD_SWITCH and (t_load_on <= tnow <= t_load_off):
        m_true, Iz_true = m_loaded, Iz_loaded
    else:
        m_true, Iz_true = m_nom, Iz_nom

    M_true = M_matrix(m_true, Iz_true)
    Fz_true = static_Fz(m_true)

    # --- reference at time k ---
    z_r   = zr[k].copy()
    z_r_d = zr_d[k].copy()
    z_r_dd = zr_dd[k].copy()
    delta_ref = deltar[k]

    # --- measured output (lookahead point) ---
    z = np.array([x + d_look*np.cos(psi), y + d_look*np.sin(psi)])
    if ADD_SENSOR_NOISE:
        z = z + sensor_noise()

    # z_dot = J(psi) * nu, where nu=[vx,vy,r]
    # z = [x;y] + d*[cos psi; sin psi]
    # x_dot,y_dot = R(psi)[vx;vy]
    # z_dot = [x_dot;y_dot] + d*r*[-sin psi; cos psi]
    Rwb = R2(psi)
    p_dot = Rwb @ np.array([vx, vy])
    z_dot = p_dot + d_look * r_yaw * np.array([-np.sin(psi), np.cos(psi)])

    # errors
    e   = z - z_r
    e_d = z_dot - z_r_d
    e_log[k] = e

    # sliding surface
    s = e_d + Lambda @ e
    s_log[k] = s

    # desired z_ddot (SMC)
    sat_arg = np.clip(s/phi, -1.0, 1.0)
    z_dd_des = z_r_dd - Lambda @ e_d - (Ksm @ sat_arg)

    # --- Map desired z_ddot to desired nu_dot = [vx_dot, vy_dot, r_dot] ---
    # We use an affine relation:
    # z_dd = A(psi,nu)*nu_dot + b(psi,nu)
    # Solve least squares for nu_dot with regularization on r_dot.
    #
    # Exact:
    # p_dot = Rwb [vx;vy]
    # p_dd  = Rwb [vx_dot;vy_dot] + r * Rwb * [[0,-1],[1,0]] [vx;vy]
    # z_dd  = p_dd + d*( r_dot*[-sin,cos] + r*(-cos,-sin)*r )
    #
    # Build A so that z_dd = A * nu_dot + b
    J = np.array([[0.0, -1.0],
                  [1.0,  0.0]])

    # A_v maps [vx_dot, vy_dot] -> Rwb*[vx_dot; vy_dot]
    A_v = Rwb

    # A_r maps r_dot -> d*[ -sin; cos ]
    A_r = d_look * np.array([[-np.sin(psi)],
                             [ np.cos(psi)]])

    A = np.hstack([A_v, A_r])  # 2x3

    # b = r*Rwb*J*[vx;vy] + d*( -r^2*[cos;sin] )
    b = (r_yaw * (Rwb @ (J @ np.array([vx, vy])))) + d_look * (-(r_yaw**2) * np.array([np.cos(psi), np.sin(psi)]))

    # solve: A nu_dot ≈ z_dd_des - b, with regularization on r_dot
    y_rhs = z_dd_des - b
    W = np.diag([1.0, 1.0, w_rdot])  # penalize rdot
    Aw = A @ np.linalg.inv(W)        # 2x3
    nu_dot_tilde = np.linalg.lstsq(Aw, y_rhs, rcond=None)[0]
    nu_dot_des = np.linalg.inv(W) @ nu_dot_tilde
    vx_dot_des, vy_dot_des, r_dot_des = nu_dot_des

    # --- Steering command: use reference steering, but rate-limit towards it ---
    if k == 0:
        delta_cmd = 0.0
    else:
        delta_cmd = delta_log[k-1]

    # slew to delta_ref
    max_step = delta_rate * dt
    delta_cmd = np.clip(delta_cmd + np.clip(delta_ref - delta_cmd, -max_step, +max_step),
                        -delta_max, +delta_max)
    delta_log[k] = delta_cmd

    # wheel headings
    dFL, dFR = ackermann_split(delta_cmd, L, t)
    wheel_heading = np.array([dFL, dFR, 0.0, 0.0])

    # --- Build B matrices ---
    Bx, By = Bx_By(wheel_heading)
    if SHOW_B_MATRICES:
        Bx00[k] = Bx[0,0]; By00[k] = By[0,0]
        Bx20[k] = Bx[2,0]; By20[k] = By[2,0]

    # --- Compute wheel contact velocities in BODY frame ---
    # v_i = [vx - r*y_i, vy + r*x_i]
    v_i = np.zeros((4,2))
    for i in range(4):
        xi, yi = wheel_pos[i]
        v_i[i,0] = vx - r_yaw*yi
        v_i[i,1] = vy + r_yaw*xi

    # --- Tire forces in wheel frame: Fx chosen to match desired vx_dot roughly (rear drive),
    #     Fy from slip angle model + friction circle
    #
    # We compute a desired body wrench tau_des = M*nu_dot_des + C*nu
    # then solve least squares for Fx_w,Fy_w that produce tau_des using [Bx By].
    nu = np.array([vx, vy, r_yaw])
    C = C_matrix(m_true, nu)
    tau_des = (M_true @ nu_dot_des) + (C @ nu)

    # Add matched disturbance to plant (not to controller)
    tau_dist = wrench_disturbance(tnow) if ADD_DISTURBANCE else np.zeros(3)

    # Solve for wheel-frame forces that produce tau_des:
    # tau = Bx Fx + By Fy  =>  tau = [Bx By] * [Fx;Fy]
    G = np.hstack([Bx, By])         # 3x8
    u = np.linalg.lstsq(G, tau_des, rcond=None)[0]  # 8,
    Fx_w = u[:4]
    Fy_w = u[4:]

    # Now apply tire saturation with friction circle and re-project simply:
    Fx_i = Fx_w.copy()
    Fy_i = Fy_w.copy()

    # (optional) improve realism: recompute Fy from slip angle and keep Fx from allocation
    for i in range(4):
        vxi = np.sign(v_i[i,0]) * max(abs(v_i[i,0]), vx_min)
        vyi = v_i[i,1]
        alpha = wrap_pi(wheel_heading[i] - np.arctan2(vyi, vxi))
        Cc = Cf if i < 2 else Cr
        Fy_model = Cc * alpha

        # blend: mostly allocation, but keep some model structure
        Fy_i[i] = 0.5*Fy_i[i] + 0.5*Fy_model

        Fx_i[i], Fy_i[i] = clamp_friction(Fx_i[i], Fy_i[i], Fz_true[i], mu)

    Fx_w_log[k,:] = Fx_i
    Fy_w_log[k,:] = Fy_i

    # --- Plant dynamics ---
    tau = (Bx @ Fx_i) + (By @ Fy_i) + tau_dist

    # drag (body)
    tau_drag = np.array([Cdrag*vx, Cdrag*vy, 0.0])
    tau = tau - tau_drag

    # nu_dot = M^{-1}(tau - C nu)
    nu_dot = np.linalg.solve(M_true, (tau - C @ nu))

    # integrate body velocities
    vx += nu_dot[0] * dt
    vy += nu_dot[1] * dt
    r_yaw += nu_dot[2] * dt

    # integrate world pose
    x_dot = vx*np.cos(psi) - vy*np.sin(psi)
    y_dot = vx*np.sin(psi) + vy*np.cos(psi)
    psi_dot = r_yaw

    x += x_dot * dt
    y += y_dot * dt
    psi = wrap_pi(psi + psi_dot * dt)

    # logs
    x_log[k], y_log[k], psi_log[k] = x, y, psi
    vx_log[k], vy_log[k], r_log[k] = vx, vy, r_yaw

# ============================================================
# Animation + Plots
# ============================================================
# Build reference COM path from reference lookahead:
# zr = [xr + d cos psi, yr + d sin psi]  => COM = zr - d*[cos psi, sin psi]
x_ref_com = zr[:,0] - d_look*np.cos(psir)
y_ref_com = zr[:,1] - d_look*np.sin(psir)

# --- Figure 1: Animation ---
fig_anim, ax = plt.subplots(figsize=(7,7))
ax.set_aspect('equal')
ax.set_title("Dynamic Ackermann Tracking (Feasible Square Reference)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

pad = 0.8
ax.set_xlim(min(x_log.min(), x_ref_com.min())-pad, max(x_log.max(), x_ref_com.max())+pad)
ax.set_ylim(min(y_log.min(), y_ref_com.min())-pad, max(y_log.max(), y_ref_com.max())+pad)

ax.plot(x_ref_com, y_ref_com, 'k--', lw=2, label="reference (COM)")
(trace,) = ax.plot([], [], 'b-', lw=2, label="actual (COM)")
(body_line,) = ax.plot([], [], 'k-', lw=2)

# wheel centers (rear axle frame used in your kinematics; here CG frame)
pFL_draw = p_FL
pFR_draw = p_FR
pRL_draw = p_RL
pRR_draw = p_RR

(FL_line,) = ax.plot([], [], 'r-', lw=3, label='FL')
(FR_line,) = ax.plot([], [], 'g-', lw=3, label='FR')
(RL_line,) = ax.plot([], [], 'b-', lw=3, label='RL')
(RR_line,) = ax.plot([], [], 'm-', lw=3, label='RR')
ax.legend(loc="upper right")

def update_anim(i):
    pos = np.array([x_log[i], y_log[i]])
    heading = psi_log[i]

    body_poly = body_outline(pos, heading)
    body_line.set_data(body_poly[:,0], body_poly[:,1])

    Rg = R2(heading)
    cFL = pos + (Rg @ pFL_draw)
    cFR = pos + (Rg @ pFR_draw)
    cRL = pos + (Rg @ pRL_draw)
    cRR = pos + (Rg @ pRR_draw)

    dFL, dFR = ackermann_split(delta_log[i], L, t)

    FL_poly = box_corners(cFL, heading + dFL, wheel_len, wheel_wid)
    FR_poly = box_corners(cFR, heading + dFR, wheel_len, wheel_wid)
    RL_poly = box_corners(cRL, heading,       wheel_len, wheel_wid)
    RR_poly = box_corners(cRR, heading,       wheel_len, wheel_wid)

    FL_line.set_data(FL_poly[:,0], FL_poly[:,1])
    FR_line.set_data(FR_poly[:,0], FR_poly[:,1])
    RL_line.set_data(RL_poly[:,0], RL_poly[:,1])
    RR_line.set_data(RR_poly[:,0], RR_poly[:,1])

    trace.set_data(x_log[:i+1], y_log[:i+1])
    return trace, body_line, FL_line, FR_line, RL_line, RR_line

ani = FuncAnimation(fig_anim, update_anim, frames=N, interval=dt*1000, blit=False, repeat=False)

# --- Figure 2: Tracking / states ---
fig2, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
fig2.suptitle("States and Tracking")

axs[0].plot(tt, vx_log, label="vx")
axs[0].axhline(v_ref, linestyle="--", label="v_ref")
axs[0].set_ylabel("vx [m/s]"); axs[0].grid(True); axs[0].legend()

axs[1].plot(tt, r_log, label="r")
axs[1].plot(tt, rref, "--", label="r_ref (bicycle)")
axs[1].set_ylabel("r [rad/s]"); axs[1].grid(True); axs[1].legend()

axs[2].plot(tt, delta_log, label="delta_cmd")
axs[2].plot(tt, deltar, "--", label="delta_ref")
axs[2].set_ylabel("delta [rad]"); axs[2].grid(True); axs[2].legend()

e_norm = np.linalg.norm(e_log, axis=1)
axs[3].plot(tt, e_norm, label="||e|| (lookahead)")
axs[3].set_ylabel("error [m]"); axs[3].set_xlabel("Time [s]")
axs[3].grid(True); axs[3].legend()

fig2.tight_layout(rect=[0,0,1,0.96])

# --- Figure 3: Wheel forces ---
fig3, axs3 = plt.subplots(4, 2, figsize=(12, 9), sharex=True)
fig3.suptitle("Wheel Forces (wheel frame)")

for i in range(4):
    axs3[i,0].plot(tt, Fx_w_log[:,i])
    axs3[i,0].set_ylabel(f"{wheel_names[i]} Fx [N]")
    axs3[i,0].grid(True)

    axs3[i,1].plot(tt, Fy_w_log[:,i])
    axs3[i,1].set_ylabel(f"{wheel_names[i]} Fy [N]")
    axs3[i,1].grid(True)

axs3[-1,0].set_xlabel("Time [s]")
axs3[-1,1].set_xlabel("Time [s]")
fig3.tight_layout(rect=[0,0,1,0.96])

# --- Figure 4 (optional): B matrix entries ---
if SHOW_B_MATRICES:
    fig4, axs4 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig4.suptitle("Example B-matrix Entries (FL column)")

    axs4[0].plot(tt, Bx00, label=r"$B_x(0,FL)$")
    axs4[0].plot(tt, By00, label=r"$B_y(0,FL)$")
    axs4[0].set_ylabel("force mapping"); axs4[0].grid(True); axs4[0].legend()

    axs4[1].plot(tt, Bx20, label=r"$B_x(2,FL)$")
    axs4[1].plot(tt, By20, label=r"$B_y(2,FL)$")
    axs4[1].set_ylabel("yaw moment map"); axs4[1].set_xlabel("Time [s]")
    axs4[1].grid(True); axs4[1].legend()

    fig4.tight_layout(rect=[0,0,1,0.93])

if __name__ == "__main__":
    plt.show()
    # To save animation (needs ffmpeg):
    # ani.save("ackermann_feasible_square_tracking.mp4", dpi=120, writer="ffmpeg")
