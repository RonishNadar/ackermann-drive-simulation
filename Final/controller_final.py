"""
Ackermann Double-Track Dynamics + Spline Path Following (Sliding Mode Control)
+ Stability analysis plots (Lyapunov + reaching condition evidence)
+ Testing & validation metrics (RMS / max / boundary-layer occupancy)

Run:
  python smc_ackermann_path_following_stability_validation.py

Dependencies:
  numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

# ---- SciPy spline ----
try:
    from scipy.interpolate import CubicSpline
except Exception as e:
    raise ImportError(
        "This script needs scipy for CubicSpline.\n"
        "Install with: pip install scipy\n"
        f"Original import error: {e}"
    )

# ============================================================
# Helpers
# ============================================================
def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], float)

def clamp(x, lo, hi):
    return float(np.minimum(np.maximum(x, lo), hi))

def sat(x):
    return np.clip(x, -1.0, 1.0)

def smc_smooth(z, eps):
    # eps > 0 : larger eps => smoother, less jitter
    return np.tanh(z / eps)

# ============================================================
# Params
# ============================================================
@dataclass
class AckermannParams:
    # geometry
    L: float = 0.35      # wheelbase [m]
    T: float = 0.28      # track width [m]
    r_w: float = 0.06    # wheel radius [m]
    Lf: float = 0.175    # CG -> front axle [m]
    Lr: float = 0.175    # CG -> rear axle [m]

    # rigid body
    m: float = 12.0      # kg
    Iz: float = 0.8      # kg*m^2

    # linear damping (body)
    du: float = 4.0      # N/(m/s)
    dv: float = 8.0      # N/(m/s)
    dr: float = 2.0      # N*m/(rad/s)

    # linear tire cornering stiffness (N/rad)
    Cf: float = 150.0
    Cr: float = 180.0

    # traction saturation (very simple)
    mu: float = 0.9
    g: float = 9.81

    # drawing
    body_length: float = 0.45
    body_width:  float = 0.30
    wheel_length: float = 0.10
    wheel_width:  float = 0.04

@dataclass
class SimParams:
    dt: float = 0.01
    steer_limit_deg: float = 45.0

@dataclass
class SMCParams:
    # max desired speed used by scheduler [m/s]
    u_max: float = 4.0
    # lateral acceleration limit for speed schedule [m/s^2]
    ay_max: float = 3.0

    # lateral sliding surface s_lat = epsi + lam*ey
    lam: float = 1.2

    # steering SMC
    k_delta: float = 0.8     # steering correction gain [rad]
    phi: float = 0.25        # boundary layer (bigger -> less chatter)

    # speed SMC
    k_u: float = 250.0       # maps speed error to Fx (N)
    phi_u: float = 0.4       # boundary layer in speed [m/s]

    # actuators
    tau_max: float = 8.0
    steer_limit_deg: float = 45.0

    # projection settings
    proj_window: float = 3.0     # search window around s_prev [m]
    proj_samples: int = 41       # how many candidates in that window

# ============================================================
# Ackermann front steering from curvature (double-track)
# ============================================================
def ackermann_front_steering_from_curvature(kappa, p: AckermannParams, steer_limit_deg=45.0):
    lim = np.deg2rad(steer_limit_deg)
    halfT = 0.5 * p.T

    if abs(kappa) < 1e-12:
        return 0.0, 0.0

    R = 1.0 / kappa
    Rabs = abs(R)
    if Rabs <= halfT + 1e-6:
        Rabs = halfT + 1e-6

    delta_in  = np.arctan(p.L / (Rabs - halfT))
    delta_out = np.arctan(p.L / (Rabs + halfT))

    sgn = np.sign(kappa)
    if kappa > 0:  # left: FL inner
        dFL = sgn * delta_in
        dFR = sgn * delta_out
    else:          # right: FR inner
        dFL = sgn * delta_out
        dFR = sgn * delta_in

    dFL = clamp(dFL, -lim, lim)
    dFR = clamp(dFR, -lim, lim)
    return dFL, dFR

# ============================================================
# Dynamics matrices: M, C, G, D (body frame at CG)
# ============================================================
def M_matrix(p: AckermannParams):
    return np.diag([p.m, p.m, p.Iz])

def C_matrix(nu, p: AckermannParams):
    u, v, r = nu
    m = p.m
    return np.array([[0.0, -m*r, 0.0],
                     [m*r,  0.0, 0.0],
                     [0.0,  0.0, 0.0]], float)

def G_vector(_eta, _p: AckermannParams):
    return np.zeros(3)

def D_matrix(p: AckermannParams):
    return np.diag([p.du, p.dv, p.dr])

# ============================================================
# Wheel torque -> wrench via B(delta)
# ============================================================
def B_matrix(delta, p: AckermannParams):
    halfT = 0.5 * p.T
    pos = {
        "RL": np.array([-p.Lr, +halfT]),
        "RR": np.array([-p.Lr, -halfT]),
        "FL": np.array([+p.Lf, +halfT]),
        "FR": np.array([+p.Lf, -halfT]),
    }
    keys = ["RL","RR","FL","FR"]
    B = np.zeros((3,4), float)

    for j,k in enumerate(keys):
        x_i, y_i = pos[k]
        d = float(delta[k])
        c = np.cos(d); s = np.sin(d)
        B[:, j] = np.array([c, s, x_i*s - y_i*c], float) / p.r_w
    return B

def traction_saturate(Fx, Fy, p: AckermannParams):
    Fmax = p.mu * p.m * p.g
    F = np.hypot(Fx, Fy)
    if F > Fmax and F > 1e-12:
        s = Fmax / F
        return Fx*s, Fy*s
    return Fx, Fy

# ============================================================
# Tire wrench: linear bicycle model
# ============================================================
def tire_wrench_linear(nu, delta_axle, p: AckermannParams, eps_u=0.2):
    u, v, r = nu
    u_eff = u if abs(u) > eps_u else (eps_u * np.sign(u) if abs(u) > 1e-6 else eps_u)

    alpha_f = np.arctan2(v + p.Lf*r, u_eff) - delta_axle
    alpha_r = np.arctan2(v - p.Lr*r, u_eff)

    Fy_f = -p.Cf * alpha_f
    Fy_r = -p.Cr * alpha_r

    # rotate front lateral force into body frame
    Fx_f = -Fy_f * np.sin(delta_axle)
    Fy_fB =  Fy_f * np.cos(delta_axle)

    Fx_r = 0.0
    Fy_rB = Fy_r

    Fx = Fx_f + Fx_r
    Fy = Fy_fB + Fy_rB
    Mz = p.Lf * Fy_fB - p.Lr * Fy_rB

    Fx, Fy = traction_saturate(Fx, Fy, p)
    return np.array([Fx, Fy, Mz], float), (alpha_f, alpha_r)

# ============================================================
# Spline path: p(s) with curvature kappa(s)
# ============================================================
class SplinePath:
    def __init__(self, waypoints, closed=False):
        pts = np.array(waypoints, float)
        assert pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 3

        if closed:
            if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
                pts = np.vstack([pts, pts[0]])

        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.hstack(([0.0], np.cumsum(ds)))
        self.s = s
        self.s_max = float(s[-1])
        self.closed = closed

        bc = "periodic" if closed else "natural"
        self.xs = CubicSpline(s, pts[:,0], bc_type=bc)
        self.ys = CubicSpline(s, pts[:,1], bc_type=bc)

    def wrap_s(self, s):
        if not self.closed:
            return float(np.clip(s, 0.0, self.s_max))
        return float(s % self.s_max)

    def eval(self, s):
        s = self.wrap_s(s)

        x = float(self.xs(s))
        y = float(self.ys(s))

        dx  = float(self.xs(s, 1))
        dy  = float(self.ys(s, 1))
        ddx = float(self.xs(s, 2))
        ddy = float(self.ys(s, 2))

        psi = float(np.arctan2(dy, dx))
        denom = (dx*dx + dy*dy)**1.5 + 1e-12
        kappa = float((dx*ddy - dy*ddx) / denom)

        return x, y, psi, kappa

def project_to_path(path: SplinePath, x, y, s_prev, cp: SMCParams):
    w = cp.proj_window

    if not path.closed:
        s0 = max(0.0, s_prev - w)
        s1 = min(path.s_max, s_prev + w)
        ss = np.linspace(s0, s1, cp.proj_samples)
    else:
        ss = np.linspace(s_prev - w, s_prev + w, cp.proj_samples)
        ss = np.array([path.wrap_s(si) for si in ss], float)

    best_s = float(path.wrap_s(s_prev) if path.closed else s_prev)
    best_d2 = 1e18

    for s in ss:
        px, py, _, _ = path.eval(s)
        d2 = (x - px)**2 + (y - py)**2
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(s)

    return best_s

def speed_schedule(kappa, cp: SMCParams):
    return min(cp.u_max, float(np.sqrt(cp.ay_max / (abs(kappa) + 1e-3))))

# ============================================================
# Sliding Mode Controller for spline path following
# ============================================================
def smc_path_controller_spline(path: SplinePath, p: AckermannParams, sp: SimParams, cp: SMCParams):
    steer_lim = np.deg2rad(cp.steer_limit_deg)

    def cb(t, state, s_prev):
        x, y, psi, nu = state
        u, v, r = nu

        # 1) project to get s*
        s = project_to_path(path, x, y, s_prev, cp)
        xd, yd, psi_d, kappa = path.eval(s)

        # 2) errors in path-tangent frame
        ex = x - xd
        ey = y - yd
        e_y = -np.sin(psi_d)*ex + np.cos(psi_d)*ey
        e_psi = wrap_to_pi(psi - psi_d)

        # 3) speed reference from curvature
        u_ref = speed_schedule(kappa, cp)

        # 4) sliding surface (lateral)
        s_lat = e_psi + cp.lam * e_y

        # 5) steering: feedforward + SMC correction
        delta_ff = np.arctan(p.L * kappa)
        delta_axle = delta_ff - cp.k_delta * smc_smooth(s_lat, cp.phi)
        delta_axle = clamp(delta_axle, -steer_lim, steer_lim)

        # convert axle steer -> double-track FL/FR
        kappa_cmd = np.tan(delta_axle) / p.L
        dFL, dFR = ackermann_front_steering_from_curvature(
            kappa_cmd, p, steer_limit_deg=cp.steer_limit_deg
        )

        # 6) speed SMC: rear wheel torques to track u_ref
        Fx_des = (p.du*u) - cp.k_u * smc_smooth((u - u_ref), cp.phi_u)

        tau_each = 0.5 * p.r_w * Fx_des
        tau_each = clamp(tau_each, -cp.tau_max, cp.tau_max)

        delta_cmd = {"RL": 0.0, "RR": 0.0, "FL": dFL, "FR": dFR}
        tau_cmd   = {"RL": tau_each, "RR": tau_each, "FL": 0.0, "FR": 0.0}

        dbg = dict(
            s=s, xd=xd, yd=yd, psi_d=psi_d, kappa=kappa,
            e_y=e_y, e_psi=e_psi, s_lat=s_lat,
            u_ref=u_ref, delta_axle=delta_axle, Fx_des=Fx_des
        )
        return delta_cmd, tau_cmd, dbg, s

    return cb

# ============================================================
# Simulator (Option C dynamics) for a fixed time horizon
# ============================================================
def simulate_dynamics(T_total, p: AckermannParams, sp: SimParams, controller_cb):
    dt = sp.dt
    N = int(np.ceil(T_total/dt))
    t = np.arange(N) * dt

    # states
    x = np.zeros(N); y = np.zeros(N); th = np.zeros(N)
    u = np.zeros(N); v = np.zeros(N); r = np.zeros(N)

    # logs
    delta = {k: np.zeros(N) for k in ["RL","RR","FL","FR"]}
    tau_w = {k: np.zeros(N) for k in ["RL","RR","FL","FR"]}

    tau_long  = np.zeros((N,3))
    tau_tire  = np.zeros((N,3))
    tau_total = np.zeros((N,3))

    alpha_f_log = np.zeros(N)
    alpha_r_log = np.zeros(N)

    # controller debug
    s_path = np.zeros(N)
    e_y_log = np.zeros(N)
    e_psi_log = np.zeros(N)
    s_lat_log = np.zeros(N)
    u_ref_log = np.zeros(N)
    kappa_log = np.zeros(N)

    # ---- stability/validation logs ----
    s_u_log = np.zeros(N)
    V_lat_log = np.zeros(N)
    V_u_log = np.zeros(N)
    sdot_lat_log = np.zeros(N)
    sdot_u_log = np.zeros(N)
    reach_lat_log = np.zeros(N)
    reach_u_log = np.zeros(N)

    M = M_matrix(p)
    D = D_matrix(p)

    s_prev = 0.0

    for k in range(1, N):
        nu = np.array([u[k-1], v[k-1], r[k-1]], float)
        state = (x[k-1], y[k-1], th[k-1], nu)

        delta_cmd, tau_cmd, dbg, s_prev = controller_cb(t[k-1], state, s_prev)

        s_path[k-1] = dbg["s"]
        e_y_log[k-1] = dbg["e_y"]
        e_psi_log[k-1] = dbg["e_psi"]
        s_lat_log[k-1] = dbg["s_lat"]
        u_ref_log[k-1] = dbg["u_ref"]
        kappa_log[k-1] = dbg["kappa"]

        # stability surfaces
        s_u = u[k-1] - dbg["u_ref"]
        s_u_log[k-1] = s_u
        V_lat_log[k-1] = 0.5 * (dbg["s_lat"]**2)
        V_u_log[k-1]   = 0.5 * (s_u**2)

        for wn in ["RL","RR","FL","FR"]:
            delta[wn][k-1] = float(delta_cmd[wn])
            tau_w[wn][k-1] = float(tau_cmd[wn])

        delta_axle = 0.5 * (delta["FL"][k-1] + delta["FR"][k-1])

        # wheel torque wrench
        B = B_matrix({wn: delta[wn][k-1] for wn in ["RL","RR","FL","FR"]}, p)
        tau_vec = np.array([tau_w["RL"][k-1], tau_w["RR"][k-1],
                            tau_w["FL"][k-1], tau_w["FR"][k-1]], float)
        tauB_long = B @ tau_vec
        tau_long[k-1,:] = tauB_long

        # tire wrench
        tauB_tire, (af, ar_) = tire_wrench_linear(nu, delta_axle, p)
        tau_tire[k-1,:] = tauB_tire
        alpha_f_log[k-1] = af
        alpha_r_log[k-1] = ar_

        # total wrench
        tauB = tauB_long + tauB_tire
        tau_total[k-1,:] = tauB

        # dynamics: M nu_dot = tauB - C nu - D nu - G
        C = C_matrix(nu, p)
        G = G_vector(np.array([x[k-1], y[k-1], th[k-1]]), p)
        rhs = tauB - (C @ nu) - (D @ nu) - G
        nu_dot = np.linalg.solve(M, rhs)

        u[k] = u[k-1] + nu_dot[0]*dt
        v[k] = v[k-1] + nu_dot[1]*dt
        r[k] = r[k-1] + nu_dot[2]*dt

        # kinematics: eta_dot = R(psi) * [u v]^T, psi_dot = r
        vI = rot2(th[k-1]) @ np.array([u[k-1], v[k-1]], float)
        x[k]  = x[k-1] + vI[0]*dt
        y[k]  = y[k-1] + vI[1]*dt
        th[k] = wrap_to_pi(th[k-1] + r[k-1]*dt)

    # copy last debug
    s_path[-1] = s_path[-2]
    e_y_log[-1] = e_y_log[-2]
    e_psi_log[-1] = e_psi_log[-2]
    s_lat_log[-1] = s_lat_log[-2]
    u_ref_log[-1] = u_ref_log[-2]
    kappa_log[-1] = kappa_log[-2]
    alpha_f_log[-1] = alpha_f_log[-2]
    alpha_r_log[-1] = alpha_r_log[-2]
    s_u_log[-1] = s_u_log[-2]
    V_lat_log[-1] = V_lat_log[-2]
    V_u_log[-1] = V_u_log[-2]
    for wn in ["RL","RR","FL","FR"]:
        delta[wn][-1] = delta[wn][-2]
        tau_w[wn][-1] = tau_w[wn][-2]
    tau_long[-1,:]  = tau_long[-2,:]
    tau_tire[-1,:]  = tau_tire[-2,:]
    tau_total[-1,:] = tau_total[-2,:]

    # derivatives + reaching evidence (finite differences)
    sdot_lat_log[:-1] = np.diff(s_lat_log) / dt
    sdot_lat_log[-1]  = sdot_lat_log[-2]
    sdot_u_log[:-1]   = np.diff(s_u_log) / dt
    sdot_u_log[-1]    = sdot_u_log[-2]

    reach_lat_log[:] = s_lat_log * sdot_lat_log
    reach_u_log[:]   = s_u_log   * sdot_u_log

    return dict(
        t=t, x=x, y=y, th=th, u=u, v=v, r=r,
        delta=delta, tau_w=tau_w,
        tau_long=tau_long, tau_tire=tau_tire, tau_total=tau_total,
        alpha_f=alpha_f_log, alpha_r=alpha_r_log,
        s_path=s_path, e_y=e_y_log, e_psi=e_psi_log, s_lat=s_lat_log,
        u_ref=u_ref_log, kappa=kappa_log,
        # stability/validation logs
        s_u=s_u_log, V_lat=V_lat_log, V_u=V_u_log,
        sdot_lat=sdot_lat_log, sdot_u=sdot_u_log,
        reach_lat=reach_lat_log, reach_u=reach_u_log,
        params=p, simparams=sp
    )

# ============================================================
# Plotting
# ============================================================
def make_plots(sim, path: SplinePath, show_path=True):
    t = sim["t"]
    x,y = sim["x"], sim["y"]
    u,v,r = sim["u"], sim["v"], sim["r"]
    delta = sim["delta"]
    tau_w = sim["tau_w"]

    e_y = sim["e_y"]
    e_psi = sim["e_psi"]
    s_lat = sim["s_lat"]
    u_ref = sim["u_ref"]
    kappa = sim["kappa"]

    fig, axs = plt.subplots(5, 1, figsize=(11, 15), sharex=False)

    axs[0].plot(x, y, label="car")
    if show_path:
        ss = np.linspace(0.0, path.s_max, 400)
        px = np.array([path.eval(s)[0] for s in ss])
        py = np.array([path.eval(s)[1] for s in ss])
        axs[0].plot(px, py, "--", label="path")
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_title("Trajectory (Dynamics + SMC Path Following)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, u, label="u")
    axs[1].plot(t, v, label="v")
    axs[1].plot(t, r, label="r")
    axs[1].plot(t, u_ref, "--", label="u_ref (scheduled)")
    axs[1].set_title("Body velocities + scheduled speed")
    axs[1].grid(True)
    axs[1].legend(ncols=4)

    axs[2].plot(t, e_y, label="e_y (cross-track)")
    axs[2].plot(t, np.rad2deg(e_psi), label="e_psi [deg]")
    axs[2].plot(t, s_lat, label="s_lat")
    axs[2].set_title("Path tracking errors / sliding surface")
    axs[2].grid(True)
    axs[2].legend(ncols=3)

    axs[3].plot(t, kappa, label="kappa(s*)")
    axs[3].set_title("Path curvature seen by controller")
    axs[3].grid(True)
    axs[3].legend()

    axs[4].plot(t, tau_w["RL"], label="tau_RL")
    axs[4].plot(t, tau_w["RR"], label="tau_RR")
    axs[4].plot(t, np.rad2deg(delta["FL"]), "--", label="delta_FL [deg]")
    axs[4].plot(t, np.rad2deg(delta["FR"]), "--", label="delta_FR [deg]")
    axs[4].set_title("Actuation")
    axs[4].grid(True)
    axs[4].legend(ncols=4)

    plt.tight_layout()
    plt.show()

def make_stability_plots(sim, eta_lat=0.2, eta_u=0.2):
    t = sim["t"]

    s_lat = sim["s_lat"]
    reach_lat = sim["reach_lat"]
    V_lat = sim["V_lat"]

    s_u = sim["s_u"]
    reach_u = sim["reach_u"]
    V_u = sim["V_u"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0,0].plot(t, V_lat, label="V_lat = 0.5*s_lat^2")
    axs[0,0].set_title("Lateral Lyapunov candidate")
    axs[0,0].grid(True)
    axs[0,0].legend()

    axs[0,1].plot(t, reach_lat, label=r"$s_{lat}\dot{s}_{lat}$")
    axs[0,1].plot(t, -eta_lat*np.abs(s_lat), "--", label=r"$-\eta|s_{lat}|$")
    axs[0,1].axhline(0.0, linewidth=1)
    axs[0,1].set_title("Lateral reaching condition evidence")
    axs[0,1].grid(True)
    axs[0,1].legend()

    axs[1,0].plot(t, V_u, label="V_u = 0.5*s_u^2")
    axs[1,0].set_title("Speed Lyapunov candidate")
    axs[1,0].grid(True)
    axs[1,0].legend()

    axs[1,1].plot(t, reach_u, label=r"$s_u\dot{s}_u$")
    axs[1,1].plot(t, -eta_u*np.abs(s_u), "--", label=r"$-\eta|s_u|$")
    axs[1,1].axhline(0.0, linewidth=1)
    axs[1,1].set_title("Speed reaching condition evidence")
    axs[1,1].grid(True)
    axs[1,1].legend()

    plt.tight_layout()
    plt.show()

def compute_metrics(sim, cp: SMCParams):
    ey = sim["e_y"]
    epsi = sim["e_psi"]
    s_lat = sim["s_lat"]
    su = sim["s_u"]

    out = {}
    out["RMS_e_y_m"] = float(np.sqrt(np.mean(ey**2)))
    out["MAX_abs_e_y_m"] = float(np.max(np.abs(ey)))

    epsi_deg = np.rad2deg(epsi)
    out["RMS_e_psi_deg"] = float(np.sqrt(np.mean(epsi_deg**2)))
    out["MAX_abs_e_psi_deg"] = float(np.max(np.abs(epsi_deg)))

    out["frac_in_lat_boundary_(|s_lat|<=phi)"] = float(np.mean(np.abs(s_lat) <= cp.phi))
    out["frac_in_u_boundary_(|u-u_ref|<=phi_u)"] = float(np.mean(np.abs(su) <= cp.phi_u))
    return out

# ============================================================
# Animation
# ============================================================
def animate(sim, path: SplinePath = None, interval_ms=20, trail=True, show_path=True):
    p = sim["params"]
    t = sim["t"]
    x,y,th = sim["x"], sim["y"], sim["th"]
    delta = sim["delta"]

    halfT = 0.5*p.T
    wheel_body = {
        "RL": np.array([-p.Lr, +halfT]),
        "RR": np.array([-p.Lr, -halfT]),
        "FL": np.array([+p.Lf, +halfT]),
        "FR": np.array([+p.Lf, -halfT]),
    }

    fig, ax = plt.subplots(figsize=(9,7))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title("Ackermann Double-Track Dynamics + SMC Path Following")

    pad = 1.0
    ax.set_xlim(np.min(x)-pad, np.max(x)+pad)
    ax.set_ylim(np.min(y)-pad, np.max(y)+pad)

    if show_path and (path is not None):
        ss = np.linspace(0.0, path.s_max, 500)
        px = np.array([path.eval(s)[0] for s in ss])
        py = np.array([path.eval(s)[1] for s in ss])
        ax.plot(px, py, "--", lw=1)

    body_line, = ax.plot([], [], lw=2)
    wheel_lines = {k: ax.plot([], [], lw=2)[0] for k in wheel_body}
    heading_line, = ax.plot([], [], lw=2)
    trail_line = ax.plot([], [], lw=1)[0] if trail else None

    def rect(center, length, width, ang):
        Lh, Wh = length/2, width/2
        pts = np.array([[ Lh,  Wh],
                        [ Lh, -Wh],
                        [-Lh, -Wh],
                        [-Lh,  Wh],
                        [ Lh,  Wh]], float)
        Rm = rot2(ang)
        ptsW = (Rm @ pts.T).T + center
        return ptsW

    def update(i):
        xi, yi, thi = x[i], y[i], th[i]
        Rm = rot2(thi)

        body_center = np.array([xi, yi])
        bp = rect(body_center, p.body_length, p.body_width, thi)
        body_line.set_data(bp[:,0], bp[:,1])

        for kname in wheel_body:
            wp = np.array([xi, yi]) + Rm @ wheel_body[kname]
            ang = thi + delta[kname][i] if kname in ["FL","FR"] else thi
            wpts = rect(wp, p.wheel_length, p.wheel_width, ang)
            wheel_lines[kname].set_data(wpts[:,0], wpts[:,1])

        head = np.array([xi, yi])
        tip = head + rot2(thi) @ np.array([0.25, 0.0])
        heading_line.set_data([head[0], tip[0]], [head[1], tip[1]])

        if trail_line is not None:
            trail_line.set_data(x[:i+1], y[:i+1])

        artists = [body_line, heading_line] + list(wheel_lines.values())
        if trail_line is not None:
            artists.append(trail_line)
        return artists

    ani = FuncAnimation(fig, update, frames=len(t), interval=interval_ms, blit=True)
    plt.show()
    return ani

# ============================================================
# Example waypoint sets
# ============================================================
def make_circle_waypoints(R=6.0, N=12, center=(0.0, 0.0)):
    cx, cy = center
    ang = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = [(cx + R*np.cos(a), cy + R*np.sin(a)) for a in ang]
    pts.append(pts[0])
    return pts

def make_figure8_waypoints(a=6.0, b=3.0, N=30):
    tt = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = []
    for t in tt:
        x = a*np.sin(t)
        y = b*np.sin(t)*np.cos(t)
        pts.append((x, y))
    pts.append(pts[0])
    return pts

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # --- plant params ---
    p = AckermannParams(
        L=0.35, T=0.28, r_w=0.06,
        Lf=0.175, Lr=0.175,
        m=12.0, Iz=0.8,
        du=4.0, dv=8.0, dr=2.0,
        Cf=150.0, Cr=180.0,
        mu=0.9
    )
    sp = SimParams(dt=0.01, steer_limit_deg=60.0)

    # --- controller params ---
    cp = SMCParams(
        u_max=4.0, ay_max=3.0,
        lam=1.2,
        k_delta=0.8, phi=0.25,
        k_u=250.0, phi_u=0.25,
        tau_max=8.0,
        steer_limit_deg=45.0,
        proj_window=3.0,
        proj_samples=41
    )

    # ===== choose your path =====
    # waypoints = make_circle_waypoints(R=6.0, N=14, center=(0.0, 0.0)); closed = True
    waypoints = make_figure8_waypoints(a=6.0, b=3.0, N=40); closed = True
    # waypoints = [(0,0), (10,0), (12,4), (10,8), (0,8), (0,0)]; closed = True

    path = SplinePath(waypoints, closed=closed)
    controller = smc_path_controller_spline(path, p, sp, cp)

    T_total = 25.0
    sim = simulate_dynamics(T_total, p, sp, controller)

    # ---- animation + your original plots ----
    animate(sim, path=path, interval_ms=20, trail=True, show_path=True)
    make_plots(sim, path=path, show_path=True)

    # ---- stability analysis evidence ----
    make_stability_plots(sim, eta_lat=0.2, eta_u=0.2)

    # ---- testing & validation metrics ----
    metrics = compute_metrics(sim, cp)
    print("\n=== Testing & Validation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
