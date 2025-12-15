import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

# =========================
# Helpers
# =========================
def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], float)

def clamp(x, lo, hi):
    return float(np.minimum(np.maximum(x, lo), hi))

# =========================
# Params
# =========================
@dataclass
class AckermannParams:
    L: float = 0.35   # wheelbase [m]
    T: float = 0.28   # track width [m]
    r: float = 0.06   # wheel radius [m]

    # drawing
    body_length: float = 0.45
    body_width:  float = 0.30
    wheel_length: float = 0.10
    wheel_width:  float = 0.04

@dataclass
class SimParams:
    dt: float = 0.02
    steer_limit_deg: float = 45.0
    # heading tracking gain for mapping inertial velocity -> omega
    k_heading: float = 3.0
    # keep omega within reasonable range (optional)
    omega_max: float = 4.0

# =========================
# Double-track Ackermann wheel mapping from (v, omega)
# =========================
def ackermann_wheels_from_vw(v, w, p: AckermannParams, steer_limit_deg=45.0):
    halfT = 0.5 * p.T
    lim = np.deg2rad(steer_limit_deg)

    # --- 1. ACKERMANN STEERING GEOMETRY (Angles) ---
    if abs(w) < 1e-9:
        delta_FL = 0.0
        delta_FR = 0.0
    else:
        R = v / w
        Rabs = abs(R)
        
        # Guard against impossible geometry
        if Rabs <= halfT + 1e-6:
            Rabs = halfT + 1e-6

        # Standard Ackermann formulas
        delta_in_mag  = np.arctan(p.L / (Rabs - halfT))
        delta_out_mag = np.arctan(p.L / (Rabs + halfT))
        sgnw = np.sign(w)

        delta_in  = sgnw * delta_in_mag
        delta_out = sgnw * delta_out_mag

        if w > 0:   # left turn
            delta_FL, delta_FR = delta_in, delta_out
        else:       # right turn
            delta_FL, delta_FR = delta_out, delta_in

        delta_FL = clamp(delta_FL, -lim, lim)
        delta_FR = clamp(delta_FR, -lim, lim)

    # --- 2. WHEEL VELOCITIES via INVERSE KINEMATICS MATRIX ---
    # Implements the standard matrix for rolling-without-slipping
    # [w_fl, w_fr, w_rl, w_rr]^T = (1/r) * M * [vx, vy, w]^T
    
    # Precompute trig terms
    cFL, sFL = np.cos(delta_FL), np.sin(delta_FL)
    cFR, sFR = np.cos(delta_FR), np.sin(delta_FR)

    # Construct the Jacobian Matrix (M) from the image
    # Note: L is p.L, t/2 is halfT
    # Row 1 (FL): projects body motion onto steered wheel vector
    row_FL = np.array([cFL, sFL, p.L*sFL - halfT*cFL])
    
    # Row 2 (FR): projects body motion onto steered wheel vector
    row_FR = np.array([cFR, sFR, p.L*sFR + halfT*cFR])
    
    # Row 3 (RL): Standard differential drive (left side)
    row_RL = np.array([1.0, 0.0, -halfT])
    
    # Row 4 (RR): Standard differential drive (right side)
    row_RR = np.array([1.0, 0.0, +halfT])

    # Stack into Matrix M (4x3)
    M = np.vstack([row_FL, row_FR, row_RL, row_RR])

    # Input Vector (Body Velocities)
    # v_y is 0.0 because the controller enforces the non-holonomic constraint
    U = np.array([v, 0.0, w])

    # Compute Angular Velocities of Wheels (omega = 1/r * M * U)
    omegas = (1.0 / p.r) * (M @ U)

    omega_wheel = {
        "FL": omegas[0],
        "FR": omegas[1],
        "RL": omegas[2],
        "RR": omegas[3]
    }

    # Convert back to linear speeds (v = omega * r) for plotting/logging
    v_wheel = {k: omega_wheel[k] * p.r for k in omega_wheel}
    
    delta = {"RL": 0.0, "RR": 0.0, "FL": delta_FL, "FR": delta_FR}
    
    return v_wheel, omega_wheel, delta

# =========================
# Inertial command segments
# =========================
@dataclass
class SegmentI:
    duration: float
    vx_I: float
    vy_I: float
    wz_I: float  # desired yaw rate (can be inconsistent with vx,vy)

def build_testcase_I(name="straight"):
    if name == "straight":
        return [SegmentI(8.0, 0.35, 0.0, 0.0)]

    if name == "circle_like":
        return [SegmentI(18.0, 0.0, 0.0, 0.0)]

    if name == "figure8_like":
        w = 0.35
        Tloop = 2*np.pi/abs(w)
        return [SegmentI(2*Tloop + 0.5, 0.0, 0.0, 0.0)]

    if name == "infeasible":
        return [SegmentI(10.0, 0.3, 0.0, 0.35)]

    return [SegmentI(8.0, 0.3, 0.0, 0.25)]

# =========================
# Core: inertial desired -> feasible Ackermann (v, omega)
# =========================
def inertial_to_ackermann(vx_I, vy_I, wz_des, theta, sp: SimParams):
    v_I = np.array([vx_I, vy_I], float)
    speed = float(np.linalg.norm(v_I))

    if speed < 1e-9:
        theta_des = theta
    else:
        theta_des = float(np.arctan2(vy_I, vx_I))

    e = wrap_to_pi(theta_des - theta)

    omega = float(wz_des + sp.k_heading * e)
    omega = clamp(omega, -sp.omega_max, sp.omega_max)

    # forward speed (you can allow reverse if you want; keeping it simple/robust here)
    v = speed

    # infeasibility measure (what sideways velocity would be required)
    v_B = rot2(theta).T @ v_I
    vy_B_required = float(v_B[1])

    return v, omega, theta_des, e, vy_B_required

# =========================
# Simulation (consistent integration!)
# =========================
def simulate_inertial(segments, p: AckermannParams, sp: SimParams,
                      inertial_callback=None):
    dt = sp.dt

    # build timeline
    cmds = []
    t = 0.0
    for seg in segments:
        n = int(np.ceil(seg.duration / dt))
        for _ in range(n):
            cmds.append((t, seg.vx_I, seg.vy_I, seg.wz_I))
            t += dt

    N = len(cmds)
    T = np.array([c[0] for c in cmds])

    x = np.zeros(N)
    y = np.zeros(N)
    th = np.zeros(N)

    vxI = np.zeros(N)
    vyI = np.zeros(N)
    wz_des = np.zeros(N)

    v_used = np.zeros(N)
    w_used = np.zeros(N)
    theta_des = np.zeros(N)
    e_heading = np.zeros(N)
    vyB_required = np.zeros(N)

    v_w = {k: np.zeros(N) for k in ["RL","RR","FL","FR"]}
    om_w = {k: np.zeros(N) for k in ["RL","RR","FL","FR"]}
    delta = {k: np.zeros(N) for k in ["RL","RR","FL","FR"]}

    for k in range(1, N):
        t = T[k-1]

        # 1) desired inertial command at this time
        if inertial_callback is None:
            _, vx, vy, wz = cmds[k-1]
        else:
            vx, vy, wz = inertial_callback(t)

        vxI[k-1], vyI[k-1], wz_des[k-1] = vx, vy, wz

        # 2) map to feasible Ackermann (v, omega)
        v, w, th_d, e, vyB_req = inertial_to_ackermann(vx, vy, wz, th[k-1], sp)
        v_used[k-1], w_used[k-1] = v, w
        theta_des[k-1], e_heading[k-1], vyB_required[k-1] = th_d, e, vyB_req

        # 3) integrate pose using the SAME (v, omega)
        x[k]  = x[k-1] + (v*np.cos(th[k-1]))*dt
        y[k]  = y[k-1] + (v*np.sin(th[k-1]))*dt
        th[k] = wrap_to_pi(th[k-1] + w*dt)

        # 4) wheels from the SAME (v, omega)
        vw, ow, de = ackermann_wheels_from_vw(v, w, p, sp.steer_limit_deg)
        for name in ["RL","RR","FL","FR"]:
            v_w[name][k]  = vw[name]
            om_w[name][k] = ow[name]
            delta[name][k] = de[name]

    # last command record
    vxI[-1], vyI[-1], wz_des[-1] = vxI[-2], vyI[-2], wz_des[-2]
    v_used[-1], w_used[-1] = v_used[-2], w_used[-2]
    theta_des[-1], e_heading[-1], vyB_required[-1] = theta_des[-2], e_heading[-2], vyB_required[-2]

    return dict(
        t=T, x=x, y=y, th=th,
        vxI=vxI, vyI=vyI, wz_des=wz_des,
        v_used=v_used, w_used=w_used,
        theta_des=theta_des, e_heading=e_heading, vyB_required=vyB_required,
        v_w=v_w, om_w=om_w, delta=delta,
        params=p, simparams=sp
    )

# =========================
# Plots
# =========================
def make_plots(sim):
    t = sim["t"]
    x,y,th = sim["x"], sim["y"], sim["th"]

    vxI, vyI, wz_des = sim["vxI"], sim["vyI"], sim["wz_des"]
    v_used, w_used = sim["v_used"], sim["w_used"]
    vyB_required = sim["vyB_required"]

    v_w, delta = sim["v_w"], sim["delta"]

    fig, axs = plt.subplots(4,1, figsize=(10,12), sharex=True)

    axs[0].plot(x, y)
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_ylabel("y [m]")
    axs[0].set_title("Trajectory (x-y)")
    axs[0].grid(True)

    axs[1].plot(t, vxI, label="vx_I")
    axs[1].plot(t, vyI, label="vy_I")
    axs[1].plot(t, wz_des, label="wz_des")
    axs[1].set_ylabel("Inertial cmd")
    axs[1].grid(True)
    axs[1].legend(ncols=3)

    axs[2].plot(t, v_used, label="v_used (body forward)")
    axs[2].plot(t, w_used, label="omega_used")
    axs[2].plot(t, vyB_required, "--", label="vy_B required (infeasibility)")
    axs[2].set_ylabel("Feasible cmd / infeas.")
    axs[2].grid(True)
    axs[2].legend(ncols=3)

    axs[3].plot(t, v_w["RL"], label="v_RL")
    axs[3].plot(t, v_w["RR"], label="v_RR")
    axs[3].plot(t, v_w["FL"], label="v_FL")
    axs[3].plot(t, v_w["FR"], label="v_FR")
    axs[3].plot(t, np.rad2deg(delta["FL"]), "--", label="delta_FL [deg]")
    axs[3].plot(t, np.rad2deg(delta["FR"]), "--", label="delta_FR [deg]")
    axs[3].set_ylabel("Wheels")
    axs[3].set_xlabel("time [s]")
    axs[3].grid(True)
    axs[3].legend(ncols=3)

    plt.tight_layout()
    plt.show()

# =========================
# Animation
# =========================
def animate(sim, interval_ms=20, trail=True):
    p = sim["params"]
    t = sim["t"]
    x,y,th = sim["x"], sim["y"], sim["th"]
    delta = sim["delta"]

    halfT = 0.5*p.T
    wheel_body = {
        "RL": np.array([0.0, +halfT]),
        "RR": np.array([0.0, -halfT]),
        "FL": np.array([p.L, +halfT]),
        "FR": np.array([p.L, -halfT]),
    }

    fig, ax = plt.subplots(figsize=(9,7))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title("Double-Track Ackermann (Inertial cmd → feasible (v,ω) → wheels)")

    pad = 0.6
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

        body_center = np.array([xi, yi]) + R @ np.array([p.L/2, 0.0])
        bp = rect(body_center, p.body_length, p.body_width, thi)
        body_line.set_data(bp[:,0], bp[:,1])

        for k in wheel_body:
            wp = np.array([xi, yi]) + R @ wheel_body[k]
            ang = thi + delta[k][i] if k in ["FL","FR"] else thi
            wpts = rect(wp, p.wheel_length, p.wheel_width, ang)
            wheel_lines[k].set_data(wpts[:,0], wpts[:,1])

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

# =========================
# Example: a true circle via inertial commands (time-varying vx_I, vy_I)
# =========================
def circle_inertial_callback(v=0.3, w=0.35):
    # Desired inertial velocity follows the vehicle heading you want for a circle:
    # vx_I = v cos(w t), vy_I = v sin(w t), and wz_des = w
    def cb(t):
        return v*np.cos(w*t), v*np.sin(w*t), w
    return cb

def figure8_inertial_callback(v=0.3, w=0.35):
    Tloop = 2*np.pi/abs(w)

    def cb(t):
        if t < Tloop:
            # loop 1: +w
            psi = w*t
            wz  = +w
        else:
            # loop 2: -w, reset phase so direction is continuous
            tau = t - Tloop
            psi = w*Tloop - w*tau   # decreases
            wz  = -w

        vx_I = v*np.cos(psi)
        vy_I = v*np.sin(psi)
        return vx_I, vy_I, wz

    return cb

# =========================
# Main
# =========================
if __name__ == "__main__":
    p = AckermannParams(L=0.35, T=0.28, r=0.06)
    sp = SimParams(dt=0.02, k_heading=3.0, omega_max=4.0, steer_limit_deg=45.0)

    segs = build_testcase_I("figure8_like")
    sim = simulate_inertial(segs, p, sp, inertial_callback=figure8_inertial_callback(v=0.3, w=0.35))

    # 1. Create the animation object
    ani = animate(sim, interval_ms=20, trail=True)

    # 2. SAVE THE VIDEO HERE ----------------------------------
    print("Saving video...")
    ani.save("kinematics_sim_animation.mp4", writer="ffmpeg", fps=60)
    print("Video saved.")
    # ---------------------------------------------------------

    # 3. Show static plots
    make_plots(sim)