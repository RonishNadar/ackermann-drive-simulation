"""
Ackermann S-path tracking using DYNAMICS with either:
  - Sliding Mode Control (SMC) on z̈, or
  - ADAPTIVE control estimating a matched disturbance in z̈.

Actuation: wheel torques via allocation from [Fx, Mz].

Right panel:
  - SMC     : sliding condition s^T sdot vs -eta*||s||_1
  - ADAPTIVE: ||s|| and ||d_hat|| (norm of estimated disturbance)

Run: python dynamics_ackermann_smc_adaptive.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# =================== USER OPTION ===================
mode = "SMC"     # "SMC" or "ADAPTIVE"

# =================== Geometry & plant ===================
L = 0.35          # wheelbase [m]
t = 0.28          # track width [m]
r = 0.06          # wheel radius [m]

m  = 9.0          # mass [kg]
Iz = 0.62         # yaw inertia [kg m^2]

# simple drag/damping (tune)
c_v_lin, c_v_quad = 2.0, 0.8
c_omega = 0.6

# look-ahead distance
d = 0.25

# =================== Simulation ===================
dt, T = 0.01, 12.0
v_ref_nom = 0.8
delta_max = np.deg2rad(18.0)
S_period  = 6.0

# Common PD shaping (on z̈ command)
Kp = np.diag([4.0, 4.0])
Kd = np.diag([2.2, 2.2])

# Sliding-mode terms (SMC)
Lambda = np.diag([1.4, 1.4])   # s = e_dot + Lambda e
Ksm    = np.diag([1.2, 1.2])
phi    = 0.04
eta    = 0.4                   # for the inequality plot
rho0, rho1 = 0.0, 0.0          # set >0 if you know a bound

# Adaptive terms (ADAPTIVE)
delta_adapt = 0.8              # δ
S_mat = Lambda                 # choose S = Λ as in your slide
dhat_init = np.zeros(2)        # start at zero

# Saturations
v_min = 0.02; v_max = 1.5
omega_max = 3.0
delta_hw  = np.deg2rad(28.0)

# Drawing geometry
body_overhang_rear  = 0.05
body_overhang_front = 0.05
wheel_len, wheel_wid = 0.16, 0.06

np.random.seed(0)

# =================== Helpers ===================
def wrap_pi(a): return (a + np.pi) % (2*np.pi) - np.pi
def sat(v): return np.clip(v, -1.0, 1.0)

def R2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,-s],[s,c]])

def H(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c, -d*s],
                     [ s,  d*c]])

def H_inv(th):
    c, s = np.cos(th), np.sin(th)
    return (1.0/d)*np.array([[ d*c, d*s],
                             [ -s,   c]])

def Hdot(th, omega):
    c, s = np.cos(th), np.sin(th)
    dHdth = np.array([[-s, -d*c],
                      [ c, -d*s]])
    return dHdth * omega

def ackermann_split(delta, L, t):
    td = np.tan(delta)
    if abs(td) < 1e-8: return delta, delta
    Rmid = L/td
    if abs(Rmid) <= t/2 + 1e-6: Rmid = np.sign(Rmid)*(t/2 + 1e-6)
    return np.arctan2(L, Rmid - t/2), np.arctan2(L, Rmid + t/2)

def box_corners(center_xy, heading, length, width):
    l, w = length/2.0, width/2.0
    local = np.array([[+l,+w],[+l,-w],[-l,-w],[-l,+w],[+l,+w]])
    return (R2(heading) @ local.T).T + center_xy

def body_outline(center_xy, heading):
    Lb = L + body_overhang_front + body_overhang_rear
    x_center = (L + body_overhang_front - body_overhang_rear)/2.0
    body_center_R = np.array([x_center, 0.0])
    return box_corners(center_xy + (R2(heading) @ body_center_R), heading, Lb, t)

# wheel centers in body frame
p_FL = np.array([L,  t/2]); p_FR = np.array([L, -t/2])
p_RL = np.array([0., t/2]); p_RR = np.array([0., -t/2])

# reference (bicycle) one-step with z_r, ż_r, z̈_r
def ref_step(prev_pose, tnow, dt):
    if prev_pose is None:
        x, y, th = 0.0, 0.0, 0.0
    else:
        x, y, th = prev_pose
    delta = delta_max*np.sin(2*np.pi*tnow/S_period)
    thdot = v_ref_nom*np.tan(delta)/L
    xdot, ydot = v_ref_nom*np.cos(th), v_ref_nom*np.sin(th)
    x += xdot*dt; y += ydot*dt; th = wrap_pi(th + thdot*dt)

    zr   = np.array([x + d*np.cos(th), y + d*np.sin(th)])
    zr_d = np.array([v_ref_nom*np.cos(th) - d*thdot*np.sin(th),
                     v_ref_nom*np.sin(th) + d*thdot*np.cos(th)])
    thdd = (v_ref_nom/L)*(1/np.cos(delta)**2)*((delta_max*2*np.pi/S_period)*np.cos(2*np.pi*tnow/S_period))
    zr_dd = np.array([
        -v_ref_nom*thdot*np.sin(th) - d*thdd*np.sin(th) - d*(thdot**2)*np.cos(th),
         v_ref_nom*thdot*np.cos(th) + d*thdd*np.cos(th) - d*(thdot**2)*np.sin(th)
    ])
    return (x, y, th), zr, zr_d, zr_dd, delta

# allocation: [Fx, Mz] -> wheel forces f (min-norm)
def stack_allocation(delta_FL, delta_FR):
    cL, sL = np.cos(delta_FL), np.sin(delta_FL)
    cR, sR = np.cos(delta_FR), np.sin(delta_FR)
    S = np.array([
        [ cL,   cR, 1.0,  1.0],
        [ L*sL - (t/2)*cL,  L*sR + (t/2)*cR,  -(t/2),   +(t/2) ]
    ])
    pinv = S.T @ np.linalg.inv(S @ S.T)
    return S, pinv

# (optional) plant wrench disturbance (set to zeros if undesired)
def wrench_disturbance(tnow):
    Fx_d = 0.8*np.sin(0.5*tnow)
    Mz_d = 0.2*np.cos(0.3*tnow)
    return np.array([Fx_d, Mz_d])

# =================== Simulate dynamics ===================
N = int(T/dt)
tt = np.linspace(0, T, N)

# states
x = y = th = 0.0
v = v_ref_nom
omega = 0.0

pose_ref = None
dhat = dhat_init.copy()   # adaptive estimate of disturbance in z̈ channel (2D)

# logs
x_log = np.zeros(N); y_log = np.zeros(N); th_log = np.zeros(N)
v_log = np.zeros(N); omega_log = np.zeros(N)
Fx_cmd_log = np.zeros(N); Mz_cmd_log = np.zeros(N)
tau_log = np.zeros((N,4))
e_log = np.zeros((N,2)); s_log = np.zeros((N,2))
sd_log = np.zeros(N); rhs_log = np.zeros(N); dhat_norm = np.zeros(N)
delta_FL_log = np.zeros(N); delta_FR_log = np.zeros(N)

s_prev = np.zeros(2)

for k, tnow in enumerate(tt):

    # reference
    pose_ref, zr, zr_d, zr_dd, _ = ref_step(pose_ref, tnow, dt)

    # output & derivatives
    z   = np.array([x + d*np.cos(th), y + d*np.sin(th)])
    z_d = H(th) @ np.array([v, omega])

    e   = z - zr
    e_d = z_d - zr_d
    e_log[k] = e

    # sliding variable
    s = e_d + Lambda @ e
    s_log[k] = s

    # desired z̈ (mode-dependent)
    if mode.upper() == "SMC":
        rho = rho0 + rho1*np.linalg.norm(e)   # can be 0 if unknown
        z_dd_des = zr_dd - Lambda @ e_d - Ksm @ sat(s/phi) - rho * s / (np.linalg.norm(s)+1e-9)
    else:  # ADAPTIVE
        z_dd_des = zr_dd - Kd @ e_d - Kp @ e - dhat
        dhat += (delta_adapt * (S_mat @ s)) * dt  #  \dot{dhat} = δ S s
        dhat_norm[k] = np.linalg.norm(dhat)

    # solve for [v̇, ω̇] from z̈ = H [v̇, ω̇]^T + Ḣ [v, ω]^T
    vdot_omega = H_inv(th) @ ( z_dd_des - Hdot(th, omega) @ np.array([v, omega]) )
    v_dot, omega_dot = float(vdot_omega[0]), float(vdot_omega[1])

    # computed wrench
    Dv = c_v_lin*v + c_v_quad*v*abs(v)
    Domega = c_omega*omega
    Fx_cmd = m*v_dot + Dv
    Mz_cmd = Iz*omega_dot + Domega

    # steering for allocation geometry
    delta_cmd = np.arctan2(L*omega, max(v, v_min))
    delta_cmd = np.clip(delta_cmd, -delta_hw, +delta_hw)
    dFL, dFR = ackermann_split(delta_cmd, L, t)
    delta_FL_log[k], delta_FR_log[k] = dFL, dFR

    # allocate wheel forces and torques
    _, pinv = stack_allocation(dFL, dFR)
    f = pinv @ np.array([Fx_cmd, Mz_cmd])
    tau = r * f

    # plant dynamics with wrench disturbance
    Fx_real, Mz_real = (Fx_cmd, Mz_cmd) + wrench_disturbance(tnow)
    v_dot_real     = (Fx_real - Dv)/m
    omega_dot_real = (Mz_real - Domega)/Iz

    v     = np.clip(v + v_dot_real*dt, 0.0, v_max)
    omega = np.clip(omega + omega_dot_real*dt, -omega_max, omega_max)

    x += v*np.cos(th)*dt
    y += v*np.sin(th)*dt
    th = wrap_pi(th + omega*dt)

    # logs
    x_log[k], y_log[k], th_log[k] = x, y, th
    v_log[k], omega_log[k] = v, omega
    Fx_cmd_log[k], Mz_cmd_log[k] = Fx_cmd, Mz_cmd
    tau_log[k,:] = tau

    # sliding inequality terms (always computed; only meaningful for SMC)
    sdot = (s - s_prev)/dt
    sd_log[k]  = float(s @ sdot)
    rhs_log[k] = -eta * (np.abs(s).sum())
    s_prev = s.copy()

# =================== Visualization ===================
fig = plt.figure(figsize=(12.8, 5.8), constrained_layout=True)
gs  = GridSpec(1, 2, figure=fig, width_ratios=[2.0, 1.0], wspace=0.28)

# Left (animation)
axA = fig.add_subplot(gs[0,0]); axA.set_aspect('equal')
axA.set_title(f"Dynamics-level {mode} tracking — S path")
axA.set_xlabel("X [m]"); axA.set_ylabel("Y [m]")
pad = 0.8
axA.set_xlim(x_log.min()-pad, x_log.max()+pad)
axA.set_ylim(y_log.min()-pad, y_log.max()+pad)

# draw ref COM curve (recompute quickly)
pose_ref = None; xr, yr = [], []
for tnow in tt:
    pose_ref, zr, zr_d, zr_dd, _ = ref_step(pose_ref, tnow, dt)
    xr.append(zr[0]-d*np.cos(pose_ref[2]))
    yr.append(zr[1]-d*np.sin(pose_ref[2]))
axA.plot(xr, yr, 'k--', lw=1.2, label='reference (COM)')
(trace,)     = axA.plot([], [], 'b-', lw=2, label='actual (COM)')

(body_line,) = axA.plot([], [], 'k-', lw=2)
(FL_line,)   = axA.plot([], [], 'r-', lw=3, label='FL')
(FR_line,)   = axA.plot([], [], 'g-', lw=3, label='FR')
(RL_line,)   = axA.plot([], [], 'b-', lw=3, label='RL')
(RR_line,)   = axA.plot([], [], 'm-', lw=3, label='RR')
axA.legend(loc='upper right')

# Right: mode-dependent panel
axR = fig.add_subplot(gs[0,1])

if mode.upper() == "SMC":
    axR.set_title("Sliding condition: $s^T\\dot{s}$ vs $-\\eta\\,\\|s\\|_1$")
    axR.set_xlabel("Time [s]"); axR.set_ylabel("value")
    (line_sd,)  = axR.plot([], [], label=r"$s^T\dot{s}$")
    (line_rhs,) = axR.plot([], [], label=r"$-\eta\,\|s\|_1$")
    axR.legend(loc="upper left")
else:  # ADAPTIVE
    axR.set_title("Adaptive: error norm and estimate magnitude")
    axR.set_xlabel("Time [s]"); axR.set_ylabel(r"$\|e\|$,  $\|\hat d_z\|$")
    (line_enorm,) = axR.plot([], [], label=r"$\|e\|$")
    (line_dhat,)  = axR.plot([], [], label=r"$\|\hat d_z\|$")
    axR.legend(loc="upper left")


# drawing helpers
def draw_car(i):
    pos = np.array([x_log[i], y_log[i]]); heading = th_log[i]
    body_poly = body_outline(pos, heading)
    body_line.set_data(body_poly[:,0], body_poly[:,1])
    Rg = R2(heading)
    cFL = pos + (Rg @ p_FL); cFR = pos + (Rg @ p_FR)
    cRL = pos + (Rg @ p_RL); cRR = pos + (Rg @ p_RR)
    dFL, dFR = delta_FL_log[i], delta_FR_log[i]
    FL_poly = box_corners(cFL, heading + dFL, wheel_len, wheel_wid)
    FR_poly = box_corners(cFR, heading + dFR, wheel_len, wheel_wid)
    RL_poly = box_corners(cRL, heading,        wheel_len, wheel_wid)
    RR_poly = box_corners(cRR, heading,        wheel_len, wheel_wid)
    FL_line.set_data(FL_poly[:,0], FL_poly[:,1])
    FR_line.set_data(FR_poly[:,0], FR_poly[:,1])
    RL_line.set_data(RL_poly[:,0], RL_poly[:,1])
    RR_line.set_data(RR_poly[:,0], RR_poly[:,1])

def update(i):
    draw_car(i)
    trace.set_data(x_log[:i+1], y_log[:i+1])
    if mode.upper() == "SMC":
        line_sd.set_data(tt[:i+1], sd_log[:i+1])
        line_rhs.set_data(tt[:i+1], rhs_log[:i+1])
        return (trace, body_line, FL_line, FR_line, RL_line, RR_line,
                line_sd, line_rhs)
    else:  # ADAPTIVE
        # Plot error norm and ||d_hat|| (not sliding norm)
        line_enorm.set_data(tt[:i+1], np.linalg.norm(e_log[:i+1], axis=1))
        line_dhat.set_data(tt[:i+1], dhat_norm[:i+1])
        return (trace, body_line, FL_line, FR_line, RL_line, RR_line,
                line_enorm, line_dhat)

ani = FuncAnimation(fig, update, frames=N, interval=dt*1000, blit=False, repeat=False)

if __name__ == "__main__":
    plt.show()
    # ani.save("sliding_mode_controller.mp4", dpi=120, writer="ffmpeg")
    # ani.save("adaptive_controller.mp4", dpi=120, writer="ffmpeg")
