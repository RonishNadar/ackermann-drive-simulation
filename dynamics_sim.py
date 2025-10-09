#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
import csv

# Try SciPy QR (pivoting); fall back to NumPy SVD if unavailable
try:
    from scipy.linalg import qr as scipy_qr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -------------------- CONFIG --------------------
LOAD_FROM_CSV = False                 # set True to read CSV (see load_inputs)
CSV_PATH = "ackermann_dynamics_inputs.csv"
DT = 0.01
T_TOTAL = 12.0                        # hard stop at 12 s
SAVE_MP4 = False
MP4_PATH = "ackermann_dynamics.mp4"
DEBUG_LOG_PATH = "ackermann_debug_log.csv"

# -------------------- PARAMS --------------------
@dataclass
class Params:
    Lf: float = 0.16
    Lr: float = 0.16
    track: float = 0.24
    rw: float = 0.05
    body_len: float = 0.38
    body_wid: float = 0.26
    wheel_len: float = 0.08
    wheel_wid: float = 0.04
    m: float = 8.0
    Iz: float = 0.45
    delta_max: float = np.deg2rad(35.0)

@dataclass
class State:
    q:  np.ndarray = field(default_factory=lambda: np.zeros(3))   # [x,y,psi]
    qd: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [xd,yd,psid]

# -------------------- GEOMETRY --------------------
def rot2d(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,-s],[s,c]])

def wheel_positions(p: Params):
    t2 = p.track/2
    r_FL = np.array([+p.Lf, +t2])
    r_FR = np.array([+p.Lf, -t2])
    r_RL = np.array([-p.Lr, +t2])
    r_RR = np.array([-p.Lr, -t2])
    return r_FL, r_FR, r_RL, r_RR

def wheel_headings_world(psi, dFL, dFR):
    th_FL = psi + dFL
    th_FR = psi + dFR
    th_RL = th_RR = psi
    return th_FL, th_FR, th_RL, th_RR

# -------------------- Force map (B) --------------------
def B_matrix(p: Params, psi, dFL, dFR):
    # World-frame lever arms for yaw torque
    Rw = rot2d(psi)
    r_FL_b, r_FR_b, r_RL_b, r_RR_b = wheel_positions(p)
    r_FL = Rw @ r_FL_b; r_FR = Rw @ r_FR_b
    r_RL = Rw @ r_RL_b; r_RR = Rw @ r_RR_b

    th_FL, th_FR, th_RL, th_RR = wheel_headings_world(psi, dFL, dFR)
    cos, sin = np.cos, np.sin
    e_FL = np.array([cos(th_FL), sin(th_FL)])
    e_FR = np.array([cos(th_FR), sin(th_FR)])
    e_RL = np.array([cos(th_RL), sin(th_RL)])
    e_RR = np.array([cos(th_RR), sin(th_RR)])

    B = np.zeros((3,4))
    # Force components
    B[0,0], B[1,0] = e_FL
    B[0,1], B[1,1] = e_FR
    B[0,2], B[1,2] = e_RL
    B[0,3], B[1,3] = e_RR
    # Yaw moment (z) = r × F  (world frame)
    B[2,0] = r_FL[0]*e_FL[1] - r_FL[1]*e_FL[0]
    B[2,1] = r_FR[0]*e_FR[1] - r_FR[1]*e_FR[0]
    B[2,2] = r_RL[0]*e_RL[1] - r_RL[1]*e_RL[0]
    B[2,3] = r_RR[0]*e_RR[1] - r_RR[1]*e_RR[0]
    return B

# -------------------- Constraints A, Adot --------------------
def A_and_Adot(p: Params, psi, dFL, dFR, psidot, dFLdot=0.0, dFRdot=0.0):
    r_FL, r_FR, r_RL, r_RR = wheel_positions(p)
    th_FL, th_FR, th_RL, th_RR = wheel_headings_world(psi, dFL, dFR)
    th     = [th_FL, th_FR, th_RL, th_RR]
    rj     = [r_FL,  r_FR,  r_RL,  r_RR]
    thdot  = [psidot+dFLdot, psidot+dFRdot, psidot, psidot]

    A  = np.zeros((4,3))
    Ad = np.zeros((4,3))
    for j in range(4):
        s, c = np.sin(th[j]), np.cos(th[j])
        rx, ry = rj[j]
        # lateral no-slip: n^T (v + ω×r) = 0; n = [-sin, +cos]
        A[j,0] = -s
        A[j,1] =  c
        A[j,2] =  s*ry + c*rx
        thd = thdot[j]
        Ad[j,0] = -c*thd
        Ad[j,1] = -s*thd
        Ad[j,2] = (c*ry - s*rx)*thd
    return A, Ad

# -------------------- Row selection helpers --------------------
def select_rows_qr(A):
    QT, RT, piv = scipy_qr(A.T, pivoting=True, mode='economic')
    return piv[:2]

def select_rows_svd(A):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    lev = np.sum(U**2, axis=1)
    return np.argsort(-lev)[:2]

# --------------- INPUTS (CSV or scripted) ---------------
def load_inputs(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    t = data['t']
    FFL = data['F_FL']; FFR = data['F_FR']; FRL = data['F_RL']; FRR = data['F_RR']
    # Expect FL/FR columns; fallback to inner/outer names if present
    dFL = data['delta_FL'] if 'delta_FL' in data.dtype.names else data['delta_i']
    dFR = data['delta_FR'] if 'delta_FR' in data.dtype.names else data['delta_o']
    dFLd = data['delta_FL_dot'] if 'delta_FL_dot' in data.dtype.names else (
           data['delta_i_dot'] if 'delta_i_dot' in data.dtype.names else np.zeros_like(t))
    dFRd = data['delta_FR_dot'] if 'delta_FR_dot' in data.dtype.names else (
           data['delta_o_dot'] if 'delta_o_dot' in data.dtype.names else np.zeros_like(t))
    F = np.vstack((FFL, FFR, FRL, FRR)).T
    D = np.vstack((dFL, dFR, dFLd, dFRd)).T
    return t, F, D

def scripted_inputs(T, dt):
    """S curve with C1 (cos-ramp) steering, angles are FL/FR (not inner/outer)."""
    t = np.arange(0.0, T+dt, dt)
    n = len(t)
    F  = np.zeros((n,4))
    dFL = np.zeros(n); dFR = np.zeros(n)

    # phases
    t0,t1,t2,t3,t4,t5 = 0.0,2.0,5.0,7.0,10.0,12.0
    i0,i1,i2,i3,i4,i5 = [int(x/dt) for x in (t0,t1,t2,t3,t4,t5)]

    # forces (tune to taste)
    F[i0:i1,:] = 10.0
    F[i1:i2,:] = 8.0
    F[i2:i3,:] = 5.0
    F[i3:i4,:] = 8.0
    F[i4:i5,:] = 4.0

    steer_angle = 15.0
    d_in  = np.deg2rad(steer_angle + 2.0)   # inner magnitude
    d_out = np.deg2rad(steer_angle - 2.0)   # outer magnitude

    def cos_ramp(arr, i_start, i_end, v0, v1):
        if i_end <= i_start: return
        k = i_end - i_start
        s = np.linspace(0.0, 1.0, k, endpoint=False)
        arr[i_start:i_end] = v0 + (v1 - v0)*0.5*(1.0 - np.cos(np.pi*s))

    tau = 0.4; ir = max(1, int(tau/dt))

    # LEFT (inner=FL): FL=+d_in, FR=+d_out
    cos_ramp(dFL, i1, i1+ir, 0.0, +d_in);  dFL[i1+ir:i2-ir] = +d_in;  cos_ramp(dFL, i2-ir, i2, +d_in, 0.0)
    cos_ramp(dFR, i1, i1+ir, 0.0, +d_out); dFR[i1+ir:i2-ir] = +d_out; cos_ramp(dFR, i2-ir, i2, +d_out, 0.0)
    # RIGHT (inner=FR): FL=-d_out, FR=-d_in
    cos_ramp(dFL, i3, i3+ir, 0.0, -d_out); dFL[i3+ir:i4-ir] = -d_out; cos_ramp(dFL, i4-ir, i4, -d_out, 0.0)
    cos_ramp(dFR, i3, i3+ir, 0.0, -d_in);  dFR[i3+ir:i4-ir] = -d_in;  cos_ramp(dFR, i4-ir, i4, -d_in, 0.0)

    dFLd = np.gradient(dFL, dt)
    dFRd = np.gradient(dFR, dt)
    D = np.vstack((dFL, dFR, dFLd, dFRd)).T
    return t, F, D

# -------------------- SIMULATOR (midpoint KKT + row selection) --------------------
class AckermannDynamicsSim:
    def __init__(self, p=Params()):
        self.p = p
        self.s = State()
        self.M = np.diag([p.m, p.m, p.Iz])
        self.Minv = np.linalg.inv(self.M)
        self.eps_reg = 1e-8   # regularize tiny solves

    def _pick_rows(self, A):
        return select_rows_qr(A) if SCIPY_OK else select_rows_svd(A)

    def step(self, dt, F4, dFL, dFR, dFLd=0.0, dFRd=0.0):
        dFL = float(np.clip(dFL, -self.p.delta_max, self.p.delta_max))
        dFR = float(np.clip(dFR, -self.p.delta_max, self.p.delta_max))

        q, qd = self.s.q, self.s.qd
        psi, r = q[2], qd[2]

        # ---- first frame (dt==0): project velocity onto constraints and return ----
        if dt <= 0.0:
            A_full, _ = A_and_Adot(self.p, psi, dFL, dFR, 0.0, 0.0, 0.0)
            rows = self._pick_rows(A_full)
            Ared = A_full[rows, :]
            AMi  = Ared @ self.Minv
            W    = AMi @ Ared.T
            Wreg = W + self.eps_reg*np.eye(W.shape[0])
            mu   = np.linalg.solve(Wreg, -(Ared @ qd))
            qd_p = qd + (self.Minv @ Ared.T @ mu)
            self.s.qd = qd_p
            # outputs
            r_FL_b, r_FR_b, r_RL_b, r_RR_b = wheel_positions(self.p)
            Rw = rot2d(q[2])
            def wheel_vel_world(r_b):
                omega_cross_r_body = np.array([-qd_p[2] * r_b[1], qd_p[2] * r_b[0]])
                return qd_p[:2] + (Rw @ omega_cross_r_body)
            v_FL = wheel_vel_world(r_FL_b); v_FR = wheel_vel_world(r_FR_b)
            v_RL = wheel_vel_world(r_RL_b); v_RR = wheel_vel_world(r_RR_b)
            th_FL, th_FR, th_RL, th_RR = wheel_headings_world(q[2], dFL, dFR)
            def v_long(v, th): return float(v @ np.array([np.cos(th), np.sin(th)]))
            return dict(q=q.copy(), qd=qd_p.copy(),
                        V_FL=v_long(v_FL, th_FL), V_FR=v_long(v_FR, th_FR),
                        V_RL=v_long(v_RL, th_RL), V_RR=v_long(v_RR, th_RR),
                        dFL=dFL, dFR=dFR)

        # ---------- midpoint states (evaluate constraint here) ----------
        psi_mid = psi + 0.5*dt*r
        dFL_mid = dFL + 0.5*dt*dFLd
        dFR_mid = dFR + 0.5*dt*dFRd

        Bmid = B_matrix(self.p, psi_mid, dFL_mid, dFR_mid)
        A_full, _ = A_and_Adot(self.p, psi_mid, dFL_mid, dFR_mid, 0.0, 0.0, 0.0)  # Ad not needed

        # --- pick best two independent rows each step ---
        rows = self._pick_rows(A_full)
        Ared = A_full[rows, :]

        AMi = Ared @ self.Minv
        W   = AMi @ Ared.T
        Wreg = W + self.eps_reg*np.eye(W.shape[0])

        # Discrete KKT: enforce A_mid * qd_{n+1} = 0
        rhs = -(1.0/dt) * (Ared @ qd) - (AMi @ (Bmid @ F4))
        lam = np.linalg.solve(Wreg, rhs)

        qdd    = self.Minv @ (Bmid @ F4 + Ared.T @ lam)
        qd_new = qd + dt*qdd
        q_new  = q + dt*qd_new
        q_new[2] = (q_new[2] + np.pi) % (2*np.pi) - np.pi

        self.s.q, self.s.qd = q_new, qd_new

        # wheel-center longitudinal speeds for plots (ω×r rotated to world)
        r_FL_b, r_FR_b, r_RL_b, r_RR_b = wheel_positions(self.p)
        Rw = rot2d(q_new[2]); rr = qd_new[2]
        def wheel_vel_world(r_b):
            omega_cross_r_body = np.array([-rr * r_b[1], rr * r_b[0]])
            return qd_new[:2] + (Rw @ omega_cross_r_body)
        v_FL = wheel_vel_world(r_FL_b); v_FR = wheel_vel_world(r_FR_b)
        v_RL = wheel_vel_world(r_RL_b); v_RR = wheel_vel_world(r_RR_b)
        th_FL, th_FR, th_RL, th_RR = wheel_headings_world(q_new[2], dFL, dFR)
        def v_long(v, th): return float(v @ np.array([np.cos(th), np.sin(th)]))
        return dict(q=q_new.copy(), qd=qd_new.copy(),
                    V_FL=v_long(v_FL, th_FL), V_FR=v_long(v_FR, th_FR),
                    V_RL=v_long(v_RL, th_RL), V_RR=v_long(v_RR, th_RR),
                    dFL=dFL, dFR=dFR)

# -------------------- PLOTTING --------------------
def make_rect(w,h,**kw): return Rectangle((-w/2,-h/2), w, h, fill=False, **kw)
def set_patch(ax, rect, theta, center_xy):
    tr = (plt.matplotlib.transforms.Affine2D().rotate(theta).translate(*center_xy))
    rect.set_transform(tr + ax.transData)
def autoscale_y(ax, ys):
    if len(ys)<2: return
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    if ymin==ymax: ymin-=0.1; ymax+=0.1
    pad = 0.05*(ymax-ymin); ax.set_ylim(ymin-pad, ymax+pad)

# -------------------- MAIN / ANIMATION --------------------
def run():
    if LOAD_FROM_CSV:
        t_arr, F_arr, D_arr = load_inputs(CSV_PATH)
    else:
        t_arr, F_arr, D_arr = scripted_inputs(T_TOTAL, DT)

    p = Params()
    sim = AckermannDynamicsSim(p)

    fig = plt.figure(figsize=(14,8))
    gs = GridSpec(nrows=2, ncols=2, width_ratios=[2.4,1.4], height_ratios=[1,1], figure=fig)
    gs_right = gs[:,1].subgridspec(4,1, hspace=0.45)

    ax_anim = fig.add_subplot(gs[:,0])
    ax_anim.set_aspect('equal', adjustable='datalim')  # lighter than 'box'
    ax_anim.set_title("Ackermann Dynamics")
    path_ln, = ax_anim.plot([],[], lw=1.2)

    ax_ws_r = fig.add_subplot(gs_right[0,0]); ax_ws_f = fig.add_subplot(gs_right[1,0])
    ax_state = fig.add_subplot(gs_right[2,0]); ax_in = fig.add_subplot(gs_right[3,0])

    ax_ws_r.set_title("Rear longitudinal speeds"); ax_ws_r.set_xlim(t_arr[0], t_arr[-1])
    ln_rl, = ax_ws_r.plot([],[], label="V_RL"); ln_rr, = ax_ws_r.plot([],[], label="V_RR"); ax_ws_r.legend()
    ax_ws_f.set_title("Front longitudinal speeds"); ax_ws_f.set_xlim(t_arr[0], t_arr[-1])
    ln_fl, = ax_ws_f.plot([],[], label="V_FL"); ln_fr, = ax_ws_f.plot([],[], label="V_FR"); ax_ws_f.legend()

    ax_state.set_title("States: x, y, psi"); ax_state.set_xlim(t_arr[0], t_arr[-1]); ax_state.set_ylim(-2,20)
    ln_x, = ax_state.plot([],[], label="x [m]"); ln_y, = ax_state.plot([],[], label="y [m]")
    ln_psi, = ax_state.plot([],[], label="psi [rad]"); ax_state.legend()

    ax_in.set_title("Inputs: mean |F| & steering (deg)"); ax_in.set_xlim(t_arr[0], t_arr[-1])
    ln_F, = ax_in.plot([],[], label="mean(|F|) [N]")
    ln_dFL, = ax_in.plot([],[], label="delta_FL [deg]")
    ln_dFR, = ax_in.plot([],[], label="delta_FR [deg]"); ax_in.legend()

    # Pre-set axis limits (avoid per-frame panning)
    ax_anim.set_xlim(-2, 24); ax_anim.set_ylim(-2, 20)
    ax_ws_r.set_ylim(-2, 12); ax_ws_f.set_ylim(-2, 12)
    ax_in.set_ylim(-25, 25)

    body = make_rect(p.body_len, p.body_wid, linewidth=2)
    w_RL = make_rect(p.wheel_len, p.wheel_wid, linewidth=1.5)
    w_RR = make_rect(p.wheel_len, p.wheel_wid, linewidth=1.5)
    w_FL = make_rect(p.wheel_len, p.wheel_wid, linewidth=1.5)
    w_FR = make_rect(p.wheel_len, p.wheel_wid, linewidth=1.5)
    for a in (body,w_RL,w_RR,w_FL,w_FR): ax_anim.add_patch(a)

    offs = {
        'RL': np.array([-p.Lr, +p.track/2]),
        'RR': np.array([-p.Lr, -p.track/2]),
        'FL': np.array([+p.Lf, +p.track/2]),
        'FR': np.array([+p.Lf, -p.track/2]),
    }

    # -------------------- DEBUG LOG SETUP --------------------
    debug_fields = [
        "frame", "time", "x", "y", "psi",
        "xd", "yd", "psid",
        "V_FL", "V_FR", "V_RL", "V_RR",
        "dFL", "dFR",
        "F_FL", "F_FR", "F_RL", "F_RR"
    ]
    debug_file = open(DEBUG_LOG_PATH, "w", newline="")
    debug_writer = csv.DictWriter(debug_file, fieldnames=debug_fields)
    debug_writer.writeheader()

    log_t, log_x, log_y, log_psi = [], [], [], []
    log_V_RL, log_V_RR, log_V_FL, log_V_FR = [], [], [], []
    log_meanF, log_dFL, log_dFR = [], [], []

    def update(frame_idx):
        i = frame_idx
        # hard stop at last frame
        if i >= len(t_arr) - 1:
            # draw final frame then stop & close log
            F4 = F_arr[-1,:].astype(float)
            dFL, dFR, dFLd, dFRd = D_arr[-1,0], D_arr[-1,1], D_arr[-1,2], D_arr[-1,3]
            dt_sim = 0.0 if len(t_arr) < 2 else float(t_arr[-1] - t_arr[-2])
            out = sim.step(dt_sim, F4, dFL, dFR, dFLd, dFRd)

            # log + draw final
            log_t.append(t_arr[-1])
            log_x.append(out['q'][0]); log_y.append(out['q'][1]); log_psi.append(out['q'][2])
            log_V_RL.append(out['V_RL']); log_V_RR.append(out['V_RR'])
            log_V_FL.append(out['V_FL']); log_V_FR.append(out['V_FR'])
            log_meanF.append(float(np.mean(np.abs(F4))))
            log_dFL.append(np.rad2deg(out['dFL'])); log_dFR.append(np.rad2deg(out['dFR']))

            x,y,psi = out['q']
            path_ln.set_data(log_x, log_y)
            Rb = rot2d(psi)
            body_center = np.array([x,y]) + Rb @ np.array([p.Lf/2 - p.Lr/2, 0.0])
            set_patch(ax_anim, body, psi, body_center)
            set_patch(ax_anim, w_RL, psi, np.array([x,y]) + Rb @ offs['RL'])
            set_patch(ax_anim, w_RR, psi, np.array([x,y]) + Rb @ offs['RR'])
            set_patch(ax_anim, w_FL, psi + out['dFL'], np.array([x,y]) + Rb @ offs['FL'])
            set_patch(ax_anim, w_FR, psi + out['dFR'], np.array([x,y]) + Rb @ offs['FR'])

            ln_rl.set_data(log_t, log_V_RL); ln_rr.set_data(log_t, log_V_RR)
            ln_fl.set_data(log_t, log_V_FL); ln_fr.set_data(log_t, log_V_FR)
            ln_x.set_data(log_t, log_x); ln_y.set_data(log_t, log_y); ln_psi.set_data(log_t, log_psi)
            ln_F.set_data(log_t, log_meanF); ln_dFL.set_data(log_t, log_dFL); ln_dFR.set_data(log_t, log_dFR)

            # write final row
            debug_writer.writerow({
                "frame": i, "time": t_arr[-1],
                "x": out["q"][0], "y": out["q"][1], "psi": out["q"][2],
                "xd": out["qd"][0], "yd": out["qd"][1], "psid": out["qd"][2],
                "V_FL": out["V_FL"], "V_FR": out["V_FR"], "V_RL": out["V_RL"], "V_RR": out["V_RR"],
                "dFL": out["dFL"], "dFR": out["dFR"],
                "F_FL": F4[0], "F_FR": F4[1], "F_RL": F4[2], "F_RR": F4[3]
            })

            ani.event_source.stop()
            debug_file.close()
            print(f"Simulation completed at t = {t_arr[-1]:.2f}s. Debug log saved to {DEBUG_LOG_PATH}")
            return (path_ln,)

        if i < len(t_arr)-1:
            # keep UI responsive
            ani.event_source.interval = max(20.0, float((t_arr[i+1]-t_arr[i]) * 1000.0))

        F4 = F_arr[i,:].astype(float)
        dFL, dFR, dFLd, dFRd = D_arr[i,0], D_arr[i,1], D_arr[i,2], D_arr[i,3]
        dt_sim = 0.0 if i==0 else float(t_arr[i]-t_arr[i-1])

        out = sim.step(dt_sim, F4, dFL, dFR, dFLd, dFRd)

        log_t.append(t_arr[i])
        log_x.append(out['q'][0]); log_y.append(out['q'][1]); log_psi.append(out['q'][2])
        log_V_RL.append(out['V_RL']); log_V_RR.append(out['V_RR'])
        log_V_FL.append(out['V_FL']); log_V_FR.append(out['V_FR'])
        log_meanF.append(float(np.mean(np.abs(F4))))
        log_dFL.append(np.rad2deg(out['dFL'])); log_dFR.append(np.rad2deg(out['dFR']))

        x,y,psi = out['q']
        path_ln.set_data(log_x, log_y)

        Rb = rot2d(psi)
        body_center = np.array([x,y]) + Rb @ np.array([p.Lf/2 - p.Lr/2, 0.0])
        set_patch(ax_anim, body, psi, body_center)
        set_patch(ax_anim, w_RL, psi, np.array([x,y]) + Rb @ offs['RL'])
        set_patch(ax_anim, w_RR, psi, np.array([x,y]) + Rb @ offs['RR'])
        set_patch(ax_anim, w_FL, psi + out['dFL'], np.array([x,y]) + Rb @ offs['FL'])
        set_patch(ax_anim, w_FR, psi + out['dFR'], np.array([x,y]) + Rb @ offs['FR'])

        ln_rl.set_data(log_t, log_V_RL); ln_rr.set_data(log_t, log_V_RR)
        ln_fl.set_data(log_t, log_V_FL); ln_fr.set_data(log_t, log_V_FR)
        ln_x.set_data(log_t, log_x); ln_y.set_data(log_t, log_y); ln_psi.set_data(log_t, log_psi)
        ln_F.set_data(log_t, log_meanF); ln_dFL.set_data(log_t, log_dFL); ln_dFR.set_data(log_t, log_dFR)

        # write one debug row per frame
        debug_writer.writerow({
            "frame": i, "time": t_arr[i],
            "x": out["q"][0], "y": out["q"][1], "psi": out["q"][2],
            "xd": out["qd"][0], "yd": out["qd"][1], "psid": out["qd"][2],
            "V_FL": out["V_FL"], "V_FR": out["V_FR"], "V_RL": out["V_RL"], "V_RR": out["V_RR"],
            "dFL": out["dFL"], "dFR": out["dFR"],
            "F_FL": F4[0], "F_FR": F4[1], "F_RL": F4[2], "F_RR": F4[3]
        })

        return (path_ln, body, w_RL, w_RR, w_FL, w_FR,
                ln_rl, ln_rr, ln_fl, ln_fr, ln_x, ln_y, ln_psi, ln_F, ln_dFL, ln_dFR)

    ani = animation.FuncAnimation(
        fig, update, frames=len(t_arr), interval=20, blit=False, repeat=False
    )
    plt.tight_layout()
    if SAVE_MP4:
        fps = max(1, int(round(1.0/np.mean(np.diff(t_arr)))))
        writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=2000)
        ani.save(MP4_PATH, writer=writer)
        print(f"Saved: {MP4_PATH}")
    else:
        plt.show()

if __name__ == "__main__":
    run()
