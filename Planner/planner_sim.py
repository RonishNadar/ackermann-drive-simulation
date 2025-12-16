import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons, RadioButtons
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
import heapq
import random

try:
    from scipy.interpolate import splprep, splev
except ImportError:
    raise ImportError("Import error: scipy is required")

# ============================================================
# 1. MATH & OBSTACLES
# ============================================================
def wrap_to_pi(a): return (a + np.pi) % (2*np.pi) - np.pi
def rot2(t): return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
def clamp(x, lo, hi): return float(np.minimum(np.maximum(x, lo), hi))
def smc_smooth(z, eps): return np.tanh(z / eps)

def get_rect_verts(cx, cy, w, h, angle):
    pts = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    return (rot2(angle) @ pts.T).T + np.array([cx, cy])

class Obstacle:
    def is_collision(self, x, y, margin): pass
    def draw(self, ax): pass
    def ray_hit(self, ox, oy, dx, dy, max_range): return None

class RectObs(Obstacle):
    def __init__(self, x, y, w, h, angle):
        self.x, self.y, self.w, self.h, self.angle = x, y, w, h, angle
        self.R_inv = rot2(-angle)
    def is_collision(self, x, y, margin):
        local = self.R_inv @ np.array([x-self.x, y-self.y])
        return abs(local[0]) <= (self.w/2+margin) and abs(local[1]) <= (self.h/2+margin)
    def draw(self, ax):
        v = get_rect_verts(self.x, self.y, self.w, self.h, self.angle)
        p = patches.Polygon(v, closed=True, fc='#505050', ec='#1a1a1a', lw=2, zorder=5)
        ax.add_patch(p); return p
    def ray_hit(self, ox, oy, dx, dy, max_range):
        lx, ly = self.R_inv @ np.array([ox-self.x, oy-self.y])
        ldx, ldy = self.R_inv @ np.array([dx, dy])
        t_min, t_max = 0, max_range
        for pos, dir_, size in [(lx, ldx, self.w/2), (ly, ldy, self.h/2)]:
            if abs(dir_) < 1e-6:
                if abs(pos) > size: return None
            else:
                t1, t2 = (-size - pos) / dir_, (size - pos) / dir_
                t_min = max(t_min, min(t1, t2)); t_max = min(t_max, max(t1, t2))
        if t_max >= t_min and t_min < max_range: return t_min
        return None

class CircleObs(Obstacle):
    def __init__(self, x, y, r): self.x, self.y, self.r = x, y, r
    def is_collision(self, x, y, margin): return (x-self.x)**2 + (y-self.y)**2 <= (self.r+margin)**2
    def draw(self, ax): return patches.Circle((self.x, self.y), self.r, fc='#505050', ec='k', zorder=5)

class OilPatch(Obstacle):
    def __init__(self, x, y, r, mu=0.2): self.x, self.y, self.r, self.mu = x, y, r, mu
    def is_collision(self, x, y, margin): return False
    def contains(self, x, y): return (x-self.x)**2 + (y-self.y)**2 <= self.r**2
    def draw(self, ax):
        p = patches.Circle((self.x, self.y), self.r, fc='#e67e22', alpha=0.3, ls='--', zorder=2)
        ax.add_patch(p); return p

class DynamicObs(Obstacle):
    def __init__(self, x, y, r, vx, vy):
        self.x, self.y, self.r, self.vx, self.vy = x, y, r, vx, vy
        self.patch = None
    def update(self, dt, xlim, ylim):
        self.x += self.vx * dt; self.y += self.vy * dt
        if self.x < 0 or self.x > xlim: self.vx *= -1
        if self.y < 0 or self.y > ylim: self.vy *= -1
        if self.patch: self.patch.center = (self.x, self.y)
    def is_collision(self, x, y, margin): return (x - self.x)**2 + (y - self.y)**2 <= (self.r + margin)**2
    def draw(self, ax):
        self.patch = patches.Circle((self.x, self.y), self.r, fc='#00d2d3', ec='k', zorder=6, alpha=0.8)
        ax.add_patch(self.patch); return self.patch
    def ray_hit(self, ox, oy, dx, dy, max_range):
        fx, fy = ox - self.x, oy - self.y
        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = (fx*fx + fy*fy) - self.r**2
        disc = b*b - 4*a*c
        if disc < 0: return None
        t = (-b - np.sqrt(disc))/(2*a)
        return t if 0 <= t <= max_range else None

# ============================================================
# 2. VEHICLE & PLANNING
# ============================================================
@dataclass
class AckermannParams:
    L: float = 0.35; T: float = 0.28; r_w: float = 0.06
    Lf: float = 0.175; Lr: float = 0.175
    body_length: float = 0.50; body_width: float = 0.32
    wheel_length: float = 0.12; wheel_width: float = 0.05
    m: float = 12.0; Iz: float = 0.8
    du: float = 4.0; dv: float = 8.0; dr: float = 2.0
    Cf: float = 150.0; Cr: float = 180.0; mu_nominal: float = 0.9

@dataclass
class SMCParams:
    u_max: float = 2.5
    ay_max: float = 2.0
    lam: float = 1.2
    k_delta: float = 0.8
    phi: float = 0.25
    k_u: float = 250.0; phi_u: float = 0.4; tau_max: float = 8.0
    steer_limit_deg: float = 45.0

class AckermannCar:
    def __init__(self, p): self.p = p; self.reset(0,0,0)
    def reset(self, x, y, th):
        self.state = np.array([x, y, th, 0.0, 0.0, 0.0])
        self.s_prev = 0.0; self.delta_viz = 0.0
    def step(self, delta, tau, dt, ext_force=(0,0,0), mu=None):
        x, y, th, u, v, r = self.state
        p = self.p
        mu = mu if mu else p.mu_nominal
        d_axle = 0.5*(delta['FL']+delta['FR']); self.delta_viz = d_axle
        u_eff = u if abs(u)>0.5 else 0.5
        af = np.arctan2(v+p.Lf*r, u_eff) - d_axle
        ar = np.arctan2(v-p.Lr*r, u_eff)
        Fmax_f = mu*p.m*9.8*p.Lr/p.L; Fmax_r = mu*p.m*9.8*p.Lf/p.L
        Fyf = np.clip(-p.Cf*af, -Fmax_f, Fmax_f)
        Fyr = np.clip(-p.Cr*ar, -Fmax_r, Fmax_r)
        Fx = (tau['RL']+tau['RR'])/p.r_w - p.du*u + ext_force[0]
        Fy = Fyf*np.cos(d_axle) + Fyr - p.dv*v + ext_force[1]
        Mz = p.Lf*Fyf*np.cos(d_axle) - p.Lr*Fyr - p.dr*r + ext_force[2]
        u += (Fx/p.m + v*r)*dt; v += (Fy/p.m - u*r)*dt; r += (Mz/p.Iz)*dt
        x += (u*np.cos(th) - v*np.sin(th))*dt; y += (u*np.sin(th) + v*np.cos(th))*dt; th += r*dt
        self.state = np.array([x, y, th, u, v, r])

class SmoothPath:
    def __init__(self, pts):
        pts = np.array(pts)
        if len(pts) < 2: 
            self.tck = None; self.s_max = 0.1; self.pts = np.array([[0,0],[0.1,0.1]])
            return
        keep = [True] * len(pts)
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i]-pts[i-1]) < 0.1: keep[i] = False
        pts = pts[keep]
        if len(pts) < 2:
            self.tck = None; self.s_max = 0.1; self.pts = np.array([[0,0],[0.1,0.1]])
            return
        try:
            tck, u = splprep(pts.T, s=1.5, k=min(3, len(pts)-1)) 
            self.tck = tck
            fine_u = np.linspace(0, 1, 100)
            x_fine, y_fine = splev(fine_u, self.tck)
            self.fine_pts = np.vstack([x_fine, y_fine]).T
            ds = np.linalg.norm(np.diff(self.fine_pts, axis=0), axis=1)
            self.s_max = np.sum(ds)
        except Exception:
            self.tck = None; self.s_max = 1.0; self.pts = pts

    def eval(self, s):
        if self.tck is None: return 0,0,0,0
        u_val = np.clip(s / self.s_max, 0, 1)
        (x, y) = splev(u_val, self.tck)
        (dx, dy) = splev(u_val, self.tck, der=1)
        (ddx, ddy) = splev(u_val, self.tck, der=2)
        psi = np.arctan2(dy, dx)
        k = (dx*ddy - dy*ddx) / ((dx**2 + dy**2)**1.5 + 1e-6)
        return float(x), float(y), float(psi), float(k)

def run_smc(car, path, cp):
    x, y, psi, u, v, r = car.state
    s_best, d_best = car.s_prev, 1e9
    ss = np.linspace(max(0, car.s_prev-2), min(path.s_max, car.s_prev+4), 60)
    for si in ss:
        px, py, _, _ = path.eval(si)
        d = (x-px)**2 + (y-py)**2
        if d < d_best: d_best=d; s_best=float(si)
    
    xd, yd, pd, k = path.eval(s_best)
    ey = -np.sin(pd)*(x-xd) + np.cos(pd)*(y-yd)
    ep = wrap_to_pi(psi - pd)
    
    u_ref = min(cp.u_max, np.sqrt(cp.ay_max/(abs(k)+1e-3)))
    rem = path.s_max - s_best
    if rem < 5.0: u_ref = min(u_ref, cp.u_max*(rem/5.0))
    if rem < 0.2: u_ref = 0.0
    
    df = np.arctan(car.p.L*k) - cp.k_delta*smc_smooth(ep + cp.lam*ey, cp.phi)
    df = clamp(df, -np.deg2rad(45), np.deg2rad(45))
    fx = (car.p.du*u) - cp.k_u*smc_smooth(u-u_ref, cp.phi_u) if u_ref>0 else 0
    tau = clamp(0.5*car.p.r_w*fx, -cp.tau_max, 8.0)
    return {'FL':df, 'FR':df}, {'RL':tau, 'RR':tau}, s_best

# --- HYBRID A* PLANNER ---
class HybridAStar:
    def __init__(self, xlim, ylim):
        self.xlim, self.ylim = xlim, ylim
        self.primitives = [-35, -20, 0, 20, 35]
    class Node:
        def __init__(self, x, y, yaw, cost, parent, steer):
            self.x, self.y, self.yaw, self.cost, self.parent, self.steer = x, y, yaw, cost, parent, steer
        def __lt__(self, o): return self.cost < o.cost
    def plan(self, start, goal, obstacles):
        openset = []; heapq.heappush(openset, (0, self.Node(*start, 0, None, 0)))
        closed = set()
        while openset:
            _, curr = heapq.heappop(openset)
            if np.hypot(curr.x-goal[0], curr.y-goal[1]) < 1.0:
                path = []
                while curr: path.append([curr.x, curr.y]); curr=curr.parent
                return path[::-1], curr
            idx = (int(curr.x), int(curr.y), int(curr.yaw*5))
            if idx in closed: continue
            closed.add(idx)
            if len(closed) > 5000: return None
            for deg in self.primitives:
                steer = np.deg2rad(deg)
                beta = np.arctan(0.5*np.tan(steer))
                nx = curr.x + 1.5*np.cos(curr.yaw + beta)
                ny = curr.y + 1.5*np.sin(curr.yaw + beta)
                nyaw = wrap_to_pi(curr.yaw + (1.5/0.35)*np.sin(beta))
                if not (0<=nx<=self.xlim and 0<=ny<=self.ylim): continue
                if any(o.is_collision(nx, ny, 0.7) for o in obstacles): continue
                new_cost = curr.cost + 1.5 + abs(deg)*0.01
                h = np.hypot(nx-goal[0], ny-goal[1])
                heapq.heappush(openset, (new_cost+h, self.Node(nx, ny, nyaw, new_cost, curr, deg)))
        return None

# ============================================================
# 3. INTERACTIVE SIMULATION
# ============================================================
class InteractiveSim:
    def __init__(self):
        self.p = AckermannParams()
        self.cp = SMCParams()
        self.car = AckermannCar(self.p)
        self.planner = HybridAStar(40, 30)
        
        self.static_obs, self.dynamic_obs, self.oil_patches = [], [], []
        self.waypoints = []
        self.current_wp_idx = 0
        self.plan_end_idx = 0 
        
        self.path = None
        self.running = False
        self.replan_cooldown = 0
        self.wind_force = (0, 0)
        
        self.cfg = {'wind': True, 'oil': True, 'moving': True, 'lidar': True}
        self.c = {'bg': '#212529', 'text': '#f8f9fa', 'accent': '#48dbfb', 'danger': '#ff6b6b', 'success': '#1dd1a1'}

        self.fig = plt.figure(figsize=(14, 9), facecolor=self.c['bg'])
        gs = GridSpec(1, 2, width_ratios=[3.5, 1])
        
        self.ax = self.fig.add_subplot(gs[0])
        self.ax.set_facecolor('#2f3640')
        self.ax.set_xlim(0, 40); self.ax.set_ylim(0, 30); self.ax.set_aspect('equal')
        self.ax.tick_params(colors=self.c['text']); self.ax.grid(True, linestyle=':', alpha=0.3)
        for s in self.ax.spines.values(): s.set_edgecolor('#495057')

        self.car_body = patches.Polygon([[0,0]], fc=self.c['danger'], ec='k', zorder=10)
        self.ax.add_patch(self.car_body)
        self.wheels = [patches.Polygon([[0,0]], fc='k', zorder=11) for _ in range(4)]
        for w in self.wheels: self.ax.add_patch(w)
        
        self.line, = self.ax.plot([],[], '--', color=self.c['success'], lw=2.5, alpha=0.8)
        self.marks, = self.ax.plot([],[], 'o', color=self.c['accent'], ms=8, markeredgecolor='k')
        self.obs_patches = []
        self.lidar_dots, = self.ax.plot([], [], 'r.', markersize=4, zorder=15)
        
        self.wind_lines = []
        for _ in range(15):
            l, = self.ax.plot([], [], color='white', alpha=0.0, lw=1.5)
            self.wind_lines.append(l)
        
        self.setup_ui(gs)
        self.gen_obstacles(None)
        self.ani = FuncAnimation(self.fig, self.update, interval=30, blit=False, cache_frame_data=False)

    def setup_ui(self, gs):
        ax_ui = self.fig.add_subplot(gs[1]); ax_ui.axis('off')
        
        self.txt_stat = ax_ui.text(0.05, 0.95, "READY", transform=ax_ui.transAxes, fontsize=14, fontweight='bold', color=self.c['success'])
        self.txt_vel = ax_ui.text(0.05, 0.80, "Vel: 0.00", transform=ax_ui.transAxes, color=self.c['text'])
        self.txt_env = ax_ui.text(0.05, 0.72, "Wind: OFF", transform=ax_ui.transAxes, color=self.c['text'])
        
        def btn(rect, txt, col):
            b = Button(self.fig.add_axes(rect), txt, color=col, hovercolor='#6c757d')
            b.label.set_color('white'); b.label.set_fontweight('bold')
            return b

        self.btn_plan = btn([0.74, 0.60, 0.1, 0.05], 'PLAN', self.c['accent']); self.btn_plan.on_clicked(self.plan_callback)
        self.btn_go = btn([0.85, 0.60, 0.1, 0.05], 'GO', self.c['success']); self.btn_go.on_clicked(self.start)
        self.btn_rst = btn([0.74, 0.53, 0.21, 0.05], 'RESET', self.c['danger']); self.btn_rst.on_clicked(self.reset)
        self.btn_rnd = btn([0.74, 0.40, 0.21, 0.05], 'Random Map', '#495057'); self.btn_rnd.on_clicked(self.gen_obstacles)
        
        ax_chk = self.fig.add_axes([0.74, 0.15, 0.2, 0.15], frameon=False)
        self.chk = CheckButtons(ax_chk, ['Wind', 'Oil', 'Moving', 'Lidar'], [True, True, True, True])
        for p in ax_chk.patches: p.set_facecolor(self.c['bg']); p.set_edgecolor(self.c['text'])
        for l in self.chk.labels: l.set_color(self.c['text'])
        self.chk.on_clicked(self.toggle_cfg)
        self.fig.canvas.mpl_connect('button_press_event', self.click)

    def toggle_cfg(self, label):
        m = {'Wind':'wind', 'Oil':'oil', 'Moving':'moving', 'Lidar':'lidar'}
        self.cfg[m[label]] = not self.cfg[m[label]]
        if label=='Wind': self.txt_env.set_text(f"Wind: {'ON' if self.cfg['wind'] else 'OFF'}")

    def gen_obstacles(self, e):
        self.running = False
        for p in self.obs_patches: p.remove()
        self.obs_patches = []
        self.static_obs, self.dynamic_obs, self.oil_patches = [], [], []
        
        all_placed = []
        def check_overlap(new_o, existing, buf=1.5):
            for ex in existing:
                dist = np.hypot(new_o.x - ex.x, new_o.y - ex.y)
                r1 = new_o.r if hasattr(new_o, 'r') else np.hypot(new_o.w/2, new_o.h/2)
                r2 = ex.r if hasattr(ex, 'r') else np.hypot(ex.w/2, ex.h/2)
                if dist < (r1 + r2 + buf): return True
            return False

        for _ in range(8):
            for _ in range(20):
                o = RectObs(np.random.uniform(5,35), np.random.uniform(5,25), np.random.uniform(2,4), np.random.uniform(2,4), np.random.uniform(0,3))
                if not check_overlap(o, all_placed):
                    self.static_obs.append(o); self.obs_patches.append(o.draw(self.ax)); all_placed.append(o); break
        for _ in range(5):
            for _ in range(20):
                o = OilPatch(np.random.uniform(5,35), np.random.uniform(5,25), 2.0)
                if not check_overlap(o, all_placed, 0.5):
                    self.oil_patches.append(o); self.obs_patches.append(o.draw(self.ax)); all_placed.append(o); break
        for _ in range(4):
            for _ in range(20):
                o = DynamicObs(np.random.uniform(5,35), np.random.uniform(5,25), 1.2, np.random.uniform(-0.5,0.5), np.random.uniform(-0.5,0.5))
                if not check_overlap(o, all_placed):
                    self.dynamic_obs.append(o); self.obs_patches.append(o.draw(self.ax)); all_placed.append(o); break
        self.reset(None)

    def reset(self, e):
        self.car.reset(3,5,0); self.waypoints = []; self.path = None
        self.current_wp_idx = 0; self.plan_end_idx = 0
        self.running = False # FIX: Stop simulation loop
        self.line.set_data([],[]); self.marks.set_data([],[])
        self.lidar_dots.set_data([],[])
        for l in self.wind_lines: l.set_data([],[])
        self.txt_stat.set_text("READY"); self.txt_stat.set_color(self.c['success'])
        plt.draw()

    def click(self, e):
        if e.inaxes==self.ax and not self.running:
            if not self.waypoints or self.plan_end_idx > 0:
                if self.plan_end_idx > 0:
                    self.waypoints = []; self.plan_end_idx = 0; self.current_wp_idx = 0
                self.line.set_data([], []); self.marks.set_data([], [])
                self.path = None
                self.txt_stat.set_text("SETTING POINTS..."); self.txt_stat.set_color(self.c['text'])
                plt.draw()
            self.waypoints.append((e.xdata, e.ydata))
            self.marks.set_data([x for x,y in self.waypoints], [y for x,y in self.waypoints])
            plt.draw()

    def plan_callback(self, e):
        if self.plan_end_idx > 0:
            new_pts = self.waypoints[self.plan_end_idx:]
            self.waypoints = new_pts
            self.current_wp_idx = 0; self.plan_end_idx = 0
        
        if not self.waypoints: 
            self.txt_stat.set_text("NO POINTS"); self.txt_stat.set_color(self.c['danger'])
            return
            
        self.marks.set_data([x for x,y in self.waypoints], [y for x,y in self.waypoints])
        self.line.set_data([], []); plt.draw()

        self.txt_stat.set_text("PLANNING..."); self.txt_stat.set_color(self.c['accent']); plt.pause(0.01)
        
        full_pts = []; curr = (self.car.state[0], self.car.state[1], self.car.state[2])
        start_idx = 0
        if len(self.waypoints) > 0:
            if np.hypot(curr[0]-self.waypoints[0][0], curr[1]-self.waypoints[0][1]) < 2.0: start_idx = 1
            
        for i in range(start_idx, len(self.waypoints)):
            res = self.planner.plan(curr, self.waypoints[i], self.static_obs)
            if not res: self.txt_stat.set_text("FAIL"); self.txt_stat.set_color(self.c['danger']); return
            full_pts.extend(res[0]); curr = (res[0][-1][0], res[0][-1][1], 0)

        self.construct_path(full_pts)
        self.current_wp_idx = start_idx
        self.plan_end_idx = len(self.waypoints)
        self.txt_stat.set_text("READY"); self.txt_stat.set_color(self.c['success'])

    def construct_path(self, pts):
        self.path = SmoothPath(pts)
        if self.path.tck is None: return 
        s = np.linspace(0, self.path.s_max, 200)
        self.line.set_data([self.path.eval(si)[0] for si in s], [self.path.eval(si)[1] for si in s])

    def start(self, e): 
        if self.path: self.running=True

    def trigger_replan(self):
        self.txt_stat.set_text("AVOIDING!"); self.txt_stat.set_color(self.c['accent'])
        self.car.s_prev = 0.0
        all_obs = self.static_obs + self.dynamic_obs
        
        if self.current_wp_idx >= len(self.waypoints): targets = [(38, 28)]
        else: targets = self.waypoints[self.current_wp_idx:]
            
        curr = (self.car.state[0], self.car.state[1], self.car.state[2])
        full_pts = []
        for i, target in enumerate(targets):
            res = self.planner.plan(curr, target, all_obs)
            if not res:
                self.path = None 
                self.txt_stat.set_text("BLOCKED! WAITING..."); self.txt_stat.set_color(self.c['danger'])
                return
            full_pts.extend(res[0])
            curr = (res[0][-1][0], res[0][-1][1], 0)
        self.construct_path(full_pts)

    def update(self, frame):
        self.txt_vel.set_text(f"Vel: {self.car.state[3]:.2f}")
        
        if self.cfg['moving']:
            for o in self.dynamic_obs:
                o.patch.set_visible(True)
                if self.running: o.update(0.05, 40, 30)
        else:
            for o in self.dynamic_obs: o.patch.set_visible(False)

        wind_active = False
        if self.cfg['wind'] and frame % 120 == 0:
            angle = np.random.uniform(0, 2*np.pi)
            strength = np.random.uniform(20, 50)
            self.wind_force = (strength * np.cos(angle), strength * np.sin(angle))
            
        if self.cfg['wind'] and frame % 120 < 15:
            wind_active = True
            self.txt_env.set_text(f"Wind: {int(np.hypot(*self.wind_force))}N")
            wx, wy = self.wind_force
            norm = np.hypot(wx, wy)
            vx, vy = (wx/norm)*2.0, (wy/norm)*2.0 
            for l in self.wind_lines:
                sx = np.random.uniform(0, 40); sy = np.random.uniform(0, 30)
                l.set_data([sx, sx+vx], [sy, sy+vy]); l.set_alpha(0.6)
        else:
            self.txt_env.set_text("Wind: ON" if self.cfg['wind'] else "Wind: OFF")
            for l in self.wind_lines: l.set_alpha(0.0)

        if self.cfg['lidar']:
            cx, cy, th = self.car.state[:3]; hx, hy = [], []
            all_obs = self.static_obs + (self.dynamic_obs if self.cfg['moving'] else [])
            for i in range(-12, 13):
                angle = th + np.deg2rad(i*5); dx, dy = np.cos(angle), np.sin(angle)
                best_t = 10.0; hit = False
                for o in all_obs:
                    t = o.ray_hit(cx, cy, dx, dy, best_t)
                    if t and t<best_t: best_t=t; hit=True
                if hit: hx.append(cx+dx*best_t); hy.append(cy+dy*best_t)
            self.lidar_dots.set_data(hx, hy)

        if self.running:
            if self.cfg['moving']:
                min_d = 100.0
                for o in self.dynamic_obs:
                    d = np.hypot(self.car.state[0]-o.x, self.car.state[1]-o.y)
                    min_d = min(min_d, d)
                if min_d < 2.0: 
                    self.txt_stat.set_text("EMERGENCY BRAKE!"); self.txt_stat.set_color(self.c['danger'])
                    self.car.state[3] = 0.0 
                    if self.replan_cooldown == 0: self.trigger_replan(); self.replan_cooldown = 50
                    if self.replan_cooldown > 0: self.replan_cooldown -= 1
                    return self.car_body,

            if self.path is None and frame % 20 == 0:
                self.trigger_replan(); return self.car_body,

            if self.path:
                if self.cfg['moving'] and self.replan_cooldown == 0:
                    for o in self.dynamic_obs:
                        if np.hypot(self.car.state[0]-o.x, self.car.state[1]-o.y) < 4.0:
                            self.trigger_replan(); self.replan_cooldown=50; break
                
                if self.path is None: return self.car_body,
                if self.replan_cooldown>0: self.replan_cooldown-=1
                
                n_state = self.car.state.copy()
                n_state[0]+=np.random.normal(0,0.03); n_state[1]+=np.random.normal(0,0.03)
                noisy_car = AckermannCar(self.p); noisy_car.state = n_state; noisy_car.s_prev = self.car.s_prev
                d, t, s_new = run_smc(noisy_car, self.path, self.cp)
                
                if self.path.s_max - s_new > 1.0: pass
                self.car.s_prev = s_new
                
                if self.current_wp_idx < len(self.waypoints):
                    if np.hypot(self.car.state[0]-self.waypoints[self.current_wp_idx][0], self.car.state[1]-self.waypoints[self.current_wp_idx][1]) < 3.0:
                        self.current_wp_idx += 1
                
                mu = 0.9
                if self.cfg['oil']:
                    for o in self.oil_patches:
                        if o.contains(self.car.state[0], self.car.state[1]): mu=0.2; self.txt_stat.set_text("SLIPPING!"); break
                ext = (*self.wind_force, 0) if wind_active else (0,0,0)
                self.car.step(d, t, 0.05, ext, mu)
                
                if self.path.s_max - s_new < 0.5: 
                    self.running=False; 
                    self.txt_stat.set_text("FINISHED")
                    self.current_wp_idx = self.plan_end_idx

        x, y, th = self.car.state[:3]
        steer = self.car.delta_viz
        self.car_body.set_xy(get_rect_verts(x, y, 0.5, 0.32, th))
        offs = [[0.175, 0.14], [0.175, -0.14], [-0.175, 0.14], [-0.175, -0.14]]
        for i, off in enumerate(offs):
            wc = np.array([x,y]) + rot2(th) @ np.array(off)
            wa = th + (steer if i<2 else 0)
            self.wheels[i].set_xy(get_rect_verts(wc[0], wc[1], 0.12, 0.05, wa))
        
        return self.car_body,

if __name__ == "__main__":
    app = InteractiveSim()
    plt.show()