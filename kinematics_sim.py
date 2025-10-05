#!/usr/bin/env python3
# Ackermann-drive kinematics
# LEFT: car animation; RIGHT: 4 live plots (rear/front wheel speeds, states x/y/phi, inputs xdot/ydot/phidot)
# Uses CSV inputs literally as global velocities: x += xdot_g*dt, y += ydot_g*dt, phi += phidot*dt
# Note: This allows lateral slip relative to the Ackermann constraint if (xdot,ydot) is not aligned with heading.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter
from dataclasses import dataclass, field
import csv

# ========================== CONFIG ==========================
LOAD_FROM_CSV = True
CSV_PATH = "ackermann_traj.csv"   # CSV with columns: t, xdot_g, ydot_g, phidot
T_TOTAL = 12.0
DT = 0.02

save_path = "ackermann_sim_s_curve.mp4"
fps = int(1000 / 20)  # same as interval=20 in FuncAnimation

# ==================== CSV LOADER ============================
def load_commands_from_csv(csv_path: str):
    t_list, x_list, y_list, psi_list = [], [], [], []
    with open(csv_path, "r", newline="") as f:
        rdr = csv.reader(f)
        first = next(rdr)
        try:
            float(first[0]); row0 = first   # no header
        except Exception:
            row0 = None                     # header present
        if row0 is not None:
            t_list.append(float(row0[0])); x_list.append(float(row0[1]))
            y_list.append(float(row0[2])); psi_list.append(float(row0[3]))
        for row in rdr:
            t_list.append(float(row[0])); x_list.append(float(row[1]))
            y_list.append(float(row[2])); psi_list.append(float(row[3]))
    return np.array(t_list), np.array(x_list), np.array(y_list), np.array(psi_list)

# ================= Parameters & State =======================
@dataclass
class AckermannParams:
    L: float = 0.32
    track: float = 0.24
    r: float = 0.05
    body_len: float = 0.38
    body_wid: float = 0.26
    wheel_len: float = 0.08
    wheel_wid: float = 0.04
    delta_max: float = np.deg2rad(35.0)  # steer clamp (for display)

@dataclass
class AckermannState:
    x: float = 0.0
    y: float = 0.0
    phi: float = 0.0
    delta: float = 0.0

# ================= Vehicle (strict global) ==================
@dataclass
class AckermannVehicle:
    p: AckermannParams = field(default_factory=AckermannParams)
    s: AckermannState   = field(default_factory=AckermannState)

    @staticmethod
    def rot2d(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def wheel_kin_from_global(self, xdot_g: float, ydot_g: float, phidot: float):
        """
        Given strict global inputs, compute *display* kinematics:
        - v_forward = projection of [xdot_g, ydot_g] onto body x (purely for wheel speeds)
        - omega = phidot
        - delta for front wheels from (v_forward, omega)
        """
        # body-frame linear velocity (for wheel calcs only)
        R_T = self.rot2d(-self.s.phi)
        vx_body, vy_body = (R_T @ np.array([xdot_g, ydot_g])).tolist()
        v_forward = vx_body
        omega = phidot

        # Steering display
        if abs(v_forward) < 1e-8:
            delta_cmd = 0.0
        else:
            delta_cmd = np.arctan2(self.p.L * omega, v_forward)
        delta_cmd = float(np.clip(delta_cmd, -self.p.delta_max, self.p.delta_max))

        # Wheel-center velocities in BODY frame (using v_forward, omega)
        L, t = self.p.L, self.p.track
        v_RL = np.array([v_forward - omega*t/2, 0.0])
        v_RR = np.array([v_forward + omega*t/2, 0.0])
        v_FL = np.array([v_forward - omega*t/2, omega*L])
        v_FR = np.array([v_forward + omega*t/2, omega*L])

        V_RL = float(np.linalg.norm(v_RL)); V_RR = float(np.linalg.norm(v_RR))
        V_FL = float(np.linalg.norm(v_FL)); V_FR = float(np.linalg.norm(v_FR))
        delta_L = float(np.arctan2(v_FL[1], v_FL[0]))
        delta_R = float(np.arctan2(v_FR[1], v_FR[0]))
        return dict(v_forward=v_forward, omega=omega, delta=delta_cmd,
                    V_RL=V_RL, V_RR=V_RR, V_FL=V_FL, V_FR=V_FR,
                    delta_L=delta_L, delta_R=delta_R)

    def step_strict_global(self, dt: float, xdot_g: float, ydot_g: float, phidot: float):
        """
        Strict global integration (no projection):
        x += xdot_g*dt, y += ydot_g*dt, phi += phidot*dt
        """
        self.s.x   += dt * xdot_g
        self.s.y   += dt * ydot_g
        self.s.phi = (self.s.phi + dt * phidot + np.pi) % (2*np.pi) - np.pi

        kin = self.wheel_kin_from_global(xdot_g, ydot_g, phidot)
        self.s.delta = kin['delta']
        return kin

# ================= Helpers (drawing) =======================
def rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def make_rect(w, h, **kw):
    return Rectangle((-w/2, -h/2), w, h, fill=False, **kw)

def set_patch_transform(ax, rect, theta, center_xy):
    tr = (plt.matplotlib.transforms.Affine2D()
          .rotate(theta)
          .translate(*center_xy))
    rect.set_transform(tr + ax.transData)

# ========================= Main ============================
def run():
    # Time base
    if LOAD_FROM_CSV:
        t_cmd, xdot_cmd, ydot_cmd, phidot_cmd = load_commands_from_csv(CSV_PATH)
        # Use CSV's own time base; also make a dense display grid for axis limits
        t = t_cmd.copy()
        xdot_arr, ydot_arr, phidot_arr = xdot_cmd, ydot_cmd, phidot_cmd
        T = float(t[-1]) if len(t) > 0 else T_TOTAL
    else:
        T, dt = T_TOTAL, DT
        t = np.arange(0, T+dt, dt)
        # Example: move purely along +X, spin yaw (will cause lateral slip in body)
        xdot_arr = np.full_like(t, 0.8, dtype=float)
        ydot_arr = np.zeros_like(t)
        phidot_arr = np.piecewise(t, [t<4, (t>=4)&(t<8), t>=8], [0.2, -0.2, 0.0])

    veh = AckermannVehicle()
    p = veh.p

    # Figure layout
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(nrows=2, ncols=2, width_ratios=[2.4, 1.4], height_ratios=[1, 1], figure=fig)
    gs_right = gs[:, 1].subgridspec(4, 1, hspace=0.45)

    # Left: animation
    ax_anim = fig.add_subplot(gs[:, 0])
    ax_anim.set_aspect('equal', 'box')
    ax_anim.set_title("Ackermann (Global inputs)")
    path_ln, = ax_anim.plot([], [], lw=1.2)

    # Right: live plots
    ax_rw = fig.add_subplot(gs_right[0, 0])   # rear wheel speeds
    ax_fw = fig.add_subplot(gs_right[1, 0])   # front wheel speeds
    ax_state = fig.add_subplot(gs_right[2, 0])# x, y, phi
    ax_in = fig.add_subplot(gs_right[3, 0])   # xdot, ydot, phidot

    # Rear
    ax_rw.set_title("Rear wheel speeds")
    ax_rw.set_xlim(t[0], t[-1])
    ax_rw.set_xlabel("t [s]"); ax_rw.set_ylabel("speed [m/s]")
    ln_rl, = ax_rw.plot([], [], label="V_RL")
    ln_rr, = ax_rw.plot([], [], label="V_RR")
    ax_rw.legend(loc="upper right")

    # Front
    ax_fw.set_title("Front wheel speeds")
    ax_fw.set_xlim(t[0], t[-1])
    ax_fw.set_xlabel("t [s]"); ax_fw.set_ylabel("speed [m/s]")
    ln_fl, = ax_fw.plot([], [], label="V_FL")
    ln_fr, = ax_fw.plot([], [], label="V_FR")
    ax_fw.legend(loc="upper right")

    # States
    ax_state.set_title("States: x, y, phi")
    ax_state.set_xlim(t[0], t[-1])
    ax_state.set_xlabel("t [s]"); ax_state.set_ylabel("value")
    ln_x,   = ax_state.plot([], [], label="x [m]")
    ln_y,   = ax_state.plot([], [], label="y [m]")
    ln_phi, = ax_state.plot([], [], label="phi [rad]")
    ax_state.legend(loc="upper right")

    # Inputs
    ax_in.set_title("Inputs: xdot, ydot, phidot")
    ax_in.set_xlim(t[0], t[-1])
    ax_in.set_xlabel("t [s]"); ax_in.set_ylabel("value")
    ln_xdot,   = ax_in.plot([], [], label="xdot [m/s]")
    ln_ydot,   = ax_in.plot([], [], label="ydot [m/s]")
    ln_phidot, = ax_in.plot([], [], label="phidot [rad/s]")
    ax_in.legend(loc="upper right")

    # Patches
    body = make_rect(p.body_len, p.body_wid, linewidth=2)
    w_RL = make_rect(p.wheel_len, p.wheel_wid, linewidth=1.5)
    w_RR = make_rect(p.wheel_len, p.wheel_wid, linewidth=1.5)
    w_FL = make_rect(p.wheel_len, p.wheel_wid, linewidth=1.5)
    w_FR = make_rect(p.wheel_len, p.wheel_wid, linewidth=1.5)
    for artist in (body, w_RL, w_RR, w_FL, w_FR):
        ax_anim.add_patch(artist)

    offs = {
        'RL': np.array([0.0,  +p.track/2]),
        'RR': np.array([0.0,  -p.track/2]),
        'FL': np.array([p.L,  +p.track/2]),
        'FR': np.array([p.L,  -p.track/2]),
    }

    # Logs
    log_t = []
    log_x, log_y, log_phi = [], [], []
    log_V_RL, log_V_RR, log_V_FL, log_V_FR = [], [], [], []
    log_xdot, log_ydot, log_phidot = [], [], []

    # Initial view
    ax_anim.set_xlim(-1.5, 1.5)
    ax_anim.set_ylim(-1.5, 1.5)

    state = dict(i=0)

    def autoscale_y(ax, ys):
        if len(ys) < 2:
            return
        ymin = min(ys); ymax = max(ys)
        if ymin == ymax:
            ymin -= 0.1; ymax += 0.1
        pad = 0.05 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    def update(_frame):
        i = state['i']
        if i >= len(t):
            return (path_ln, body, w_RL, w_RR, w_FL, w_FR,
                    ln_rl, ln_rr, ln_fl, ln_fr, ln_x, ln_y, ln_phi,
                    ln_xdot, ln_ydot, ln_phidot)

        xdot = float(xdot_arr[i])
        ydot = float(ydot_arr[i])
        psi_dot = float(phidot_arr[i])

        kin = veh.step_strict_global(DT if not LOAD_FROM_CSV else (t[i]-t[i-1] if i>0 else 0.0),
                                     xdot, ydot, psi_dot)

        # logs
        log_t.append(t[i])
        log_x.append(veh.s.x); log_y.append(veh.s.y); log_phi.append(veh.s.phi)
        log_V_RL.append(kin['V_RL']); log_V_RR.append(kin['V_RR'])
        log_V_FL.append(kin['V_FL']); log_V_FR.append(kin['V_FR'])
        log_xdot.append(xdot); log_ydot.append(ydot); log_phidot.append(psi_dot)

        # keep car in view
        xmin, xmax = ax_anim.get_xlim(); ymin, ymax = ax_anim.get_ylim()
        pad = 0.5
        if veh.s.x < xmin+pad or veh.s.x > xmax-pad or veh.s.y < ymin+pad or veh.s.y > ymax-pad:
            ax_anim.set_xlim(min(xmin, veh.s.x-1.0), max(xmax, veh.s.x+1.0))
            ax_anim.set_ylim(min(ymin, veh.s.y-1.0), max(ymax, veh.s.y+1.0))

        # path & vehicle
        path_ln.set_data(log_x, log_y)
        Rb = rot2d(veh.s.phi)
        body_center = np.array([veh.s.x, veh.s.y]) + Rb @ np.array([p.L/2, 0.0])
        set_patch_transform(ax_anim, body, veh.s.phi, body_center)
        c_RL = np.array([veh.s.x, veh.s.y]) + Rb @ offs['RL']
        c_RR = np.array([veh.s.x, veh.s.y]) + Rb @ offs['RR']
        c_FL = np.array([veh.s.x, veh.s.y]) + Rb @ offs['FL']
        c_FR = np.array([veh.s.x, veh.s.y]) + Rb @ offs['FR']
        set_patch_transform(ax_anim, w_RL, veh.s.phi, c_RL)
        set_patch_transform(ax_anim, w_RR, veh.s.phi, c_RR)
        set_patch_transform(ax_anim, w_FL, veh.s.phi + veh.s.delta, c_FL)
        set_patch_transform(ax_anim, w_FR, veh.s.phi + veh.s.delta, c_FR)

        # live plots
        ln_rl.set_data(log_t, log_V_RL); ln_rr.set_data(log_t, log_V_RR)
        ln_fl.set_data(log_t, log_V_FL); ln_fr.set_data(log_t, log_V_FR)
        autoscale_y(ax_rw, log_V_RL + log_V_RR)
        autoscale_y(ax_fw, log_V_FL + log_V_FR)

        ln_x.set_data(log_t, log_x); ln_y.set_data(log_t, log_y); ln_phi.set_data(log_t, log_phi)
        autoscale_y(ax_state, log_x + log_y + log_phi)

        ln_xdot.set_data(log_t, log_xdot); ln_ydot.set_data(log_t, log_ydot); ln_phidot.set_data(log_t, log_phidot)
        autoscale_y(ax_in, log_xdot + log_ydot + log_phidot)

        state['i'] = i + 1
        return (path_ln, body, w_RL, w_RR, w_FL, w_FR,
                ln_rl, ln_rr, ln_fl, ln_fr, ln_x, ln_y, ln_phi,
                ln_xdot, ln_ydot, ln_phidot)

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=False)
    plt.tight_layout()

    # print(f"Saving animation to {save_path} ...")
    # writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    # ani.save(save_path, writer=writer)
    # print("✅ MP4 saved successfully!")

    plt.show()

if __name__ == "__main__":
    run()
