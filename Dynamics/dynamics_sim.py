import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

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

# ============================================================
# Ackermann steering (double-track) from curvature kappa = 1/R
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
    if kappa > 0:     # left: FL inner
        dFL = sgn * delta_in
        dFR = sgn * delta_out
    else:             # right: FR inner
        dFL = sgn * delta_out
        dFR = sgn * delta_in

    dFL = clamp(dFL, -lim, lim)
    dFR = clamp(dFR, -lim, lim)
    return dFL, dFR

# ============================================================
# Rigid body matrices: M, C, G, D (BODY frame at CG)
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
# Wheel torque -> longitudinal torques via B(delta)
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
        c = np.cos(d)
        s = np.sin(d)
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
# Tire lateral torques
# ============================================================
def tire_wrench_linear(nu, delta_axle, p: AckermannParams, eps_u=0.2):
    u, v, r = nu
    u_eff = u if abs(u) > eps_u else (eps_u * np.sign(u) if abs(u) > 1e-6 else eps_u)

    alpha_f = np.arctan2(v + p.Lf*r, u_eff) - delta_axle
    alpha_r = np.arctan2(v - p.Lr*r, u_eff)

    Fy_f = -p.Cf * alpha_f
    Fy_r = -p.Cr * alpha_r

    # front force into body frame
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
# Circle command: constant curvature + speed tracking via rear torques
# ============================================================
def circle_command(p: AckermannParams,
                   R=6.0,
                   v_des=4.0,
                   k_u=60.0,
                   tau_max=6.0,
                   steer_limit_deg=45.0):

    kappa = 1.0 / R
    dFL, dFR = ackermann_front_steering_from_curvature(kappa, p, steer_limit_deg=steer_limit_deg)

    def cb(_t, nu):
        u = float(nu[0])
        Fx_des = k_u * (v_des - u)
        tau_each = 0.5 * p.r_w * Fx_des
        tau_each = clamp(tau_each, -tau_max, tau_max)

        delta_cmd = {"RL": 0.0, "RR": 0.0, "FL": dFL, "FR": dFR}
        tau_cmd   = {"RL": tau_each, "RR": tau_each, "FL": 0.0, "FR": 0.0}
        return delta_cmd, tau_cmd

    return cb

# ============================================================
# Simulator: run until yaw change reaches 2π
# ============================================================
def simulate_dynamics_optionC_until_circle(p: AckermannParams,
                                           sp: SimParams,
                                           command_callback,
                                           yaw_target=2*np.pi,
                                           max_time=60.0):
    dt = sp.dt
    max_steps = int(np.ceil(max_time / dt))

    # preallocate (max)
    t  = np.zeros(max_steps)
    x  = np.zeros(max_steps)
    y  = np.zeros(max_steps)
    th = np.zeros(max_steps)

    u = np.zeros(max_steps)
    v = np.zeros(max_steps)
    r = np.zeros(max_steps)

    delta = {k: np.zeros(max_steps) for k in ["RL","RR","FL","FR"]}
    tau_w = {k: np.zeros(max_steps) for k in ["RL","RR","FL","FR"]}

    tau_long  = np.zeros((max_steps,3))
    tau_tire  = np.zeros((max_steps,3))
    tau_total = np.zeros((max_steps,3))

    alpha_f_log = np.zeros(max_steps)
    alpha_r_log = np.zeros(max_steps)

    M = M_matrix(p)
    D = D_matrix(p)

    # track accumulated yaw in an "unwrapped" way
    th_unwrap = 0.0
    th0 = th[0]

    n = 1
    for k in range(1, max_steps):
        t[k] = t[k-1] + dt
        nu = np.array([u[k-1], v[k-1], r[k-1]], float)

        # commands
        delta_cmd, tau_cmd = command_callback(t[k-1], nu)
        for wn in ["RL","RR","FL","FR"]:
            delta[wn][k-1] = float(delta_cmd[wn])
            tau_w[wn][k-1] = float(tau_cmd[wn])

        delta_axle = 0.5 * (delta["FL"][k-1] + delta["FR"][k-1])

        # wheel torque wrench
        del_now = {wn: delta[wn][k-1] for wn in ["RL","RR","FL","FR"]}
        B = B_matrix(del_now, p)
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

        # dynamics
        C = C_matrix(nu, p)
        G = G_vector(np.array([x[k-1], y[k-1], th[k-1]]), p)
        rhs = tauB - (C @ nu) - (D @ nu) - G
        nu_dot = np.linalg.solve(M, rhs)

        u[k] = u[k-1] + nu_dot[0]*dt
        v[k] = v[k-1] + nu_dot[1]*dt
        r[k] = r[k-1] + nu_dot[2]*dt

        # kinematics
        vI = rot2(th[k-1]) @ np.array([u[k-1], v[k-1]], float)
        x[k]  = x[k-1] + vI[0]*dt
        y[k]  = y[k-1] + vI[1]*dt
        th[k] = wrap_to_pi(th[k-1] + r[k-1]*dt)

        # unwrap yaw increment robustly
        dth = wrap_to_pi(th[k] - th[k-1])
        th_unwrap += dth

        n = k + 1

        if abs(th_unwrap - th0) >= yaw_target:
            break

    # finalize last sample copies
    last = n - 1
    for wn in ["RL","RR","FL","FR"]:
        delta[wn][last] = delta[wn][last-1]
        tau_w[wn][last] = tau_w[wn][last-1]
    tau_long[last,:]  = tau_long[last-1,:]
    tau_tire[last,:]  = tau_tire[last-1,:]
    tau_total[last,:] = tau_total[last-1,:]
    alpha_f_log[last] = alpha_f_log[last-1]
    alpha_r_log[last] = alpha_r_log[last-1]

    # trim
    t = t[:n]; x = x[:n]; y = y[:n]; th = th[:n]
    u = u[:n]; v = v[:n]; r = r[:n]
    tau_long = tau_long[:n,:]
    tau_tire = tau_tire[:n,:]
    tau_total = tau_total[:n,:]
    alpha_f_log = alpha_f_log[:n]
    alpha_r_log = alpha_r_log[:n]
    for wn in ["RL","RR","FL","FR"]:
        delta[wn] = delta[wn][:n]
        tau_w[wn] = tau_w[wn][:n]

    return dict(
        t=t, x=x, y=y, th=th,
        u=u, v=v, r=r,
        delta=delta, tau_w=tau_w,
        tau_long=tau_long, tau_tire=tau_tire, tau_total=tau_total,
        alpha_f=alpha_f_log, alpha_r=alpha_r_log,
        params=p, simparams=sp
    )

# ============================================================
# Plotting
# ============================================================
def make_plots(sim):
    t = sim["t"]
    x,y = sim["x"], sim["y"]
    u,v,r = sim["u"], sim["v"], sim["r"]
    delta = sim["delta"]
    tau_w = sim["tau_w"]
    tau_long = sim["tau_long"]
    tau_tire = sim["tau_tire"]
    tau_total = sim["tau_total"]
    alpha_f = sim["alpha_f"]
    alpha_r = sim["alpha_r"]

    fig, axs = plt.subplots(5,1, figsize=(11,14), sharex=True)

    axs[0].plot(x, y)
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_title("Trajectory (Dynamics + Linear Tire Model) — Circle (stop at 2π yaw)")
    axs[0].set_ylabel("y [m]")
    axs[0].grid(True)

    axs[1].plot(t, u, label="u (forward)")
    axs[1].plot(t, v, label="v (lateral)")
    axs[1].plot(t, r, label="r (yaw rate)")
    axs[1].set_ylabel("Body velocities")
    axs[1].grid(True)
    axs[1].legend(ncols=3)

    axs[2].plot(t, tau_long[:,0], label="Fx long")
    axs[2].plot(t, tau_long[:,1], label="Fy long")
    axs[2].plot(t, tau_long[:,2], label="Mz long")
    axs[2].plot(t, tau_tire[:,0], "--", label="Fx tire")
    axs[2].plot(t, tau_tire[:,1], "--", label="Fy tire")
    axs[2].plot(t, tau_tire[:,2], "--", label="Mz tire")
    axs[2].set_ylabel("Wrench parts")
    axs[2].grid(True)
    axs[2].legend(ncols=3)

    axs[3].plot(t, tau_total[:,0], label="Fx total")
    axs[3].plot(t, tau_total[:,1], label="Fy total")
    axs[3].plot(t, tau_total[:,2], label="Mz total")
    axs[3].set_ylabel("Total wrench")
    axs[3].grid(True)
    axs[3].legend(ncols=3)

    axs[4].plot(t, tau_w["RL"], label="tau_RL")
    axs[4].plot(t, tau_w["RR"], label="tau_RR")
    axs[4].plot(t, np.rad2deg(delta["FL"]), "--", label="delta_FL [deg]")
    axs[4].plot(t, np.rad2deg(delta["FR"]), "--", label="delta_FR [deg]")
    axs[4].plot(t, np.rad2deg(alpha_f), ":", label="alpha_f [deg]")
    axs[4].plot(t, np.rad2deg(alpha_r), ":", label="alpha_r [deg]")
    axs[4].set_ylabel("Torques / angles")
    axs[4].set_xlabel("time [s]")
    axs[4].grid(True)
    axs[4].legend(ncols=3)

    plt.tight_layout()
    plt.show()

# ============================================================
# Animation
# ============================================================
def animate(sim, interval_ms=20, trail=True):
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
    ax.set_title("Ackermann Double-Track Dynamics (Circle, stop at 2π yaw)")

    pad = 1.0
    ax.set_xlim(np.min(x)-pad, np.max(x)+pad)
    ax.set_ylim(np.min(y)-pad, np.max(y)+pad)

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
        R = rot2(ang)
        ptsW = (R @ pts.T).T + center
        return ptsW

    def update(i):
        xi, yi, thi = x[i], y[i], th[i]
        R = rot2(thi)

        body_center = np.array([xi, yi])
        bp = rect(body_center, p.body_length, p.body_width, thi)
        body_line.set_data(bp[:,0], bp[:,1])

        for kname in wheel_body:
            wp = np.array([xi, yi]) + R @ wheel_body[kname]
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
# Main
# ============================================================
if __name__ == "__main__":
    p = AckermannParams(
        L=0.35, T=0.28, r_w=0.06,
        Lf=0.175, Lr=0.175,
        m=12.0, Iz=0.8,
        du=4.0, dv=8.0, dr=2.0,
        Cf=150.0, Cr=180.0,
        mu=0.9
    )
    sp = SimParams(dt=0.01, steer_limit_deg=45.0)

    # Your values
    R_circle = 3.0
    v_circle = 4.0

    # Command (you can tune k_u and tau_max)
    cmd = circle_command(
        p,
        R=R_circle,
        v_des=v_circle,
        k_u=60.0,
        tau_max=6.0,
        steer_limit_deg=sp.steer_limit_deg
    )

    # Stop when yaw accumulates 2π (not when "ideal time" elapses)
    sim = simulate_dynamics_optionC_until_circle(
        p, sp, cmd,
        yaw_target=2*np.pi,
        max_time=60.0
    )

    ani = animate(sim, interval_ms=20, trail=True)

    print("Saving video...")
    ani.save("dynamics_sim_animation.mp4", writer="ffmpeg", fps=60)
    print("Video saved.")

    make_plots(sim)
