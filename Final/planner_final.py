"""
Ackermann Simulation - Final Robustness Suite
- Planner: Multi-Point Hybrid A* (Replanning enabled)
- Controller: SMC Path Following
- Disturbances: Sensor Noise, Wind Gusts, Oil Patches
- Environment: Static Walls + Bouncing Dynamic Obstacles

Usage:
1. Checkboxes control active disturbances.
2. 'Rand Obs' generates map.
3. Left Click adds waypoints -> 'PLAN' -> 'EXECUTE'.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons
from dataclasses import dataclass
import heapq
import time

try:
    from scipy.interpolate import CubicSpline
except ImportError:
    raise ImportError("Install scipy: pip install scipy")

# ============================================================
# 1. MATH HELPERS
# ============================================================
def wrap_to_pi(a): return (a + np.pi) % (2*np.pi) - np.pi
def rot2(t): return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
def clamp(x, lo, hi): return float(np.minimum(np.maximum(x, lo), hi))
def smc_smooth(z, eps): return np.tanh(z / eps)

def get_rect_verts(cx, cy, w, h, angle):
    pts = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    return (rot2(angle) @ pts.T).T + np.array([cx, cy])

# ============================================================
# 2. OBSTACLE CLASSES
# ============================================================
class Obstacle:
    def is_collision(self, x, y, margin): pass
    def draw(self, ax): pass

class RectObs(Obstacle):
    def __init__(self, x, y, w, h, angle):
        self.x, self.y, self.w, self.h, self.angle = x, y, w, h, angle
        self.R_inv = rot2(-angle)
    def is_collision(self, x, y, margin):
        local = self.R_inv @ np.array([x-self.x, y-self.y])
        return abs(local[0]) <= (self.w/2+margin) and abs(local[1]) <= (self.h/2+margin)
    def draw(self, ax):
        v = get_rect_verts(self.x, self.y, self.w, self.h, self.angle)
        p = patches.Polygon(v, closed=True, fc='#444444', ec='k', zorder=5)
        ax.add_patch(p); return p

class CircleObs(Obstacle): # Static
    def __init__(self, x, y, r): self.x, self.y, self.r = x, y, r
    def is_collision(self, x, y, margin): return (x-self.x)**2 + (y-self.y)**2 <= (self.r+margin)**2
    def draw(self, ax):
        p = patches.Circle((self.x, self.y), self.r, fc='#444444', ec='k', zorder=5)
        ax.add_patch(p); return p

class OilPatch(Obstacle):
    def __init__(self, x, y, r, mu=0.2): self.x, self.y, self.r, self.mu = x, y, r, mu
    def is_collision(self, x, y, margin): return False # Ignored by planner
    def contains(self, x, y): return (x-self.x)**2 + (y-self.y)**2 <= self.r**2
    def draw(self, ax):
        p = patches.Circle((self.x, self.y), self.r, fc='orange', alpha=0.4, ls='--', zorder=2)
        ax.add_patch(p); return p

class DynamicObs(Obstacle):
    def __init__(self, x, y, r, vx, vy):
        self.x, self.y, self.r = x, y, r
        self.vx, self.vy = vx, vy
        self.patch = None
    def update(self, dt, xlim, ylim):
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x < 0 or self.x > xlim: self.vx *= -1
        if self.y < 0 or self.y > ylim: self.vy *= -1
        if self.patch: self.patch.center = (self.x, self.y)
    def is_collision(self, x, y, margin):
        return (x - self.x)**2 + (y - self.y)**2 <= (self.r + margin)**2
    def draw(self, ax):
        self.patch = patches.Circle((self.x, self.y), self.r, fc='tab:blue', ec='k', zorder=6)
        ax.add_patch(self.patch); return self.patch

# ============================================================
# 3. PARAMETERS
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
    u_max: float = 3.5; ay_max: float = 2.5
    lam: float = 1.2; k_delta: float = 0.8; phi: float = 0.25
    k_u: float = 250.0; phi_u: float = 0.4; tau_max: float = 8.0
    steer_limit_deg: float = 45.0
    proj_window: float = 2.0; proj_samples: int = 30

# ============================================================
# 4. SIMULATION CORE
# ============================================================
class AckermannCar:
    def __init__(self, p):
        self.p = p; self.reset(0,0,0)
    def reset(self, x, y, th):
        self.state = np.array([x, y, th, 0.0, 0.0, 0.0])
        self.s_prev = 0.0; self.delta_viz = 0.0
        
    def step(self, delta, tau, dt, ext_force=(0,0,0), mu=None):
        x, y, th, u, v, r = self.state
        p = self.p
        mu = mu if mu else p.mu_nominal
        
        d_axle = 0.5*(delta['FL']+delta['FR'])
        self.delta_viz = d_axle
        
        u_eff = u if abs(u)>0.5 else 0.5
        af = np.arctan2(v+p.Lf*r, u_eff) - d_axle
        ar = np.arctan2(v-p.Lr*r, u_eff)
        
        # Tire forces with friction limit
        Fmax_f = mu*p.m*9.8*p.Lr/p.L
        Fmax_r = mu*p.m*9.8*p.Lf/p.L
        Fyf = np.clip(-p.Cf*af, -Fmax_f, Fmax_f)
        Fyr = np.clip(-p.Cr*ar, -Fmax_r, Fmax_r)
        
        fx_ext, fy_ext, mz_ext = ext_force
        Fx = (tau['RL']+tau['RR'])/p.r_w - p.du*u + fx_ext
        Fy = Fyf*np.cos(d_axle) + Fyr - p.dv*v + fy_ext
        Mz = p.Lf*Fyf*np.cos(d_axle) - p.Lr*Fyr - p.dr*r + mz_ext
        
        u += (Fx/p.m + v*r)*dt
        v += (Fy/p.m - u*r)*dt
        r += (Mz/p.Iz)*dt
        x += (u*np.cos(th) - v*np.sin(th))*dt
        y += (u*np.sin(th) + v*np.cos(th))*dt
        th += r*dt
        
        self.state = np.array([x, y, th, u, v, r])

class SplinePath:
    def __init__(self, pts):
        pts = np.array(pts)
        if len(pts)>1:
            keep=[True]*len(pts)
            for i in range(1,len(pts)):
                if np.linalg.norm(pts[i]-pts[i-1])<0.1: keep[i]=False
            pts=pts[keep]
        
        if len(pts)<2: raise ValueError
        elif len(pts)==2:
            s = np.array([0, np.linalg.norm(pts[1]-pts[0])])
            self.xs = CubicSpline(s, pts[:,0], bc_type='natural')
            self.ys = CubicSpline(s, pts[:,1], bc_type='natural')
            self.s_max = s[-1]
        else:
            ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            s = np.hstack(([0.0], np.cumsum(ds)))
            self.s_max = s[-1]
            self.xs = CubicSpline(s, pts[:,0]); self.ys = CubicSpline(s, pts[:,1])

    def eval(self, s):
        s = np.clip(s, 0, self.s_max)
        x, y = float(self.xs(s)), float(self.ys(s))
        dx, dy = float(self.xs(s,1)), float(self.ys(s,1))
        ddx, ddy = float(self.xs(s,2)), float(self.ys(s,2))
        psi = np.arctan2(dy, dx)
        k = (dx*ddy - dy*ddx)/((dx**2+dy**2)**1.5 + 1e-6)
        return x, y, psi, k

def run_smc(car, path, cp):
    x, y, psi, u, v, r = car.state
    s_best, d_best = car.s_prev, 1e9
    ss = np.linspace(max(0, car.s_prev-3), min(path.s_max, car.s_prev+3), 40)
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
    tau = clamp(0.5*car.p.r_w*fx, -cp.tau_max, cp.tau_max)
    return {'FL':df, 'FR':df}, {'RL':tau, 'RR':tau}, s_best

# ============================================================
# 5. HYBRID A* PLANNER
# ============================================================
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
            
            for deg in self.primitives:
                steer = np.deg2rad(deg)
                beta = np.arctan(0.5*np.tan(steer))
                nx = curr.x + 1.5*np.cos(curr.yaw + beta)
                ny = curr.y + 1.5*np.sin(curr.yaw + beta)
                nyaw = wrap_to_pi(curr.yaw + (1.5/0.35)*np.sin(beta))
                
                if not (0<=nx<=self.xlim and 0<=ny<=self.ylim): continue
                if any(o.is_collision(nx, ny, 0.4) for o in obstacles): continue
                
                new_cost = curr.cost + 1.5 + abs(deg)*0.01
                h = np.hypot(nx-goal[0], ny-goal[1])
                heapq.heappush(openset, (new_cost+h, self.Node(nx, ny, nyaw, new_cost, curr, deg)))
        return None

# ============================================================
# 6. INTERACTIVE SIMULATION
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
        self.path = None
        self.running = False
        self.replan_cooldown = 0
        
        self.cfg = {'wind': True, 'oil': True, 'moving': True}
        
        self.fig, self.ax = plt.subplots(figsize=(10,9))
        plt.subplots_adjust(bottom=0.2)
        self.ax.set_xlim(0,40); self.ax.set_ylim(0,30); self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f4f4f4')
        
        self.car_body = patches.Polygon([[0,0]], fc='red', ec='k', zorder=10)
        self.ax.add_patch(self.car_body)
        self.wheels = [patches.Polygon([[0,0]], fc='k', zorder=11) for _ in range(4)]
        for w in self.wheels: self.ax.add_patch(w)
        
        self.line, = self.ax.plot([],[], 'g--', lw=2)
        self.marks, = self.ax.plot([],[], 'r.', ms=10)
        self.obs_patches = []
        
        self.setup_ui()
        self.gen_obstacles(None)
        
        self.ani = FuncAnimation(self.fig, self.update, interval=30, blit=False, cache_frame_data=False)

    def setup_ui(self):
        ax_obs = plt.axes([0.1, 0.02, 0.15, 0.06])
        ax_plan = plt.axes([0.3, 0.02, 0.15, 0.06])
        ax_go = plt.axes([0.5, 0.02, 0.15, 0.06])
        ax_rst = plt.axes([0.7, 0.02, 0.15, 0.06])
        self.btn_obs = Button(ax_obs, 'Rand Obs'); self.btn_obs.on_clicked(self.gen_obstacles)
        self.btn_plan = Button(ax_plan, 'PLAN'); self.btn_plan.on_clicked(self.plan_callback)
        self.btn_go = Button(ax_go, 'EXECUTE'); self.btn_go.on_clicked(self.start)
        self.btn_rst = Button(ax_rst, 'Reset'); self.btn_rst.on_clicked(self.reset)
        
        ax_chk = plt.axes([0.02, 0.4, 0.15, 0.15], frameon=False)
        self.chk = CheckButtons(ax_chk, ['Wind', 'Oil', 'Moving Obs'], [True, True, True])
        self.chk.on_clicked(self.toggle_cfg)
        self.fig.canvas.mpl_connect('button_press_event', self.click)

    def toggle_cfg(self, label):
        map_ = {'Wind': 'wind', 'Oil': 'oil', 'Moving Obs': 'moving'}
        self.cfg[map_[label]] = not self.cfg[map_[label]]

    def gen_obstacles(self, e):
        self.running = False
        for p in self.obs_patches: p.remove()
        self.obs_patches = []
        self.static_obs, self.dynamic_obs, self.oil_patches = [], [], []
        
        for _ in range(8):
            o = RectObs(np.random.uniform(5,35), np.random.uniform(5,25), np.random.uniform(2,4), np.random.uniform(2,4), np.random.uniform(0,3))
            self.static_obs.append(o); self.obs_patches.append(o.draw(self.ax))
        
        for _ in range(5):
            o = OilPatch(np.random.uniform(5,35), np.random.uniform(5,25), 2.0, 0.2)
            self.oil_patches.append(o); self.obs_patches.append(o.draw(self.ax))
            
        for _ in range(4):
            # Slowed down speed: (-0.5, 0.5)
            vx, vy = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
            o = DynamicObs(np.random.uniform(5,35), np.random.uniform(5,25), 1.2, vx, vy)
            self.dynamic_obs.append(o); self.obs_patches.append(o.draw(self.ax))
        self.reset(None)

    def reset(self, e):
        self.car.reset(3,5,0)
        self.waypoints = []; self.path = None
        self.line.set_data([],[]); self.marks.set_data([],[])
        self.ax.set_title("Ready")
        plt.draw()

    def click(self, e):
        if e.inaxes==self.ax and not self.running:
            self.waypoints.append((e.xdata, e.ydata))
            self.marks.set_data([x for x,y in self.waypoints], [y for x,y in self.waypoints])
            plt.draw()

    def plan_callback(self, e):
        self.current_wp_idx = 0
        self.plan_path(self.static_obs) # Initial plan ignores dynamic

    def plan_path(self, obstacles):
        if not self.waypoints: return
        self.ax.set_title("Planning...")
        plt.pause(0.01)
        
        full = []
        curr_state = (self.car.state[0], self.car.state[1], self.car.state[2])
        
        start_idx = self.current_wp_idx
        if start_idx >= len(self.waypoints): start_idx = len(self.waypoints)-1

        for i in range(start_idx, len(self.waypoints)):
            w = self.waypoints[i]
            res = self.planner.plan(curr_state, w, obstacles)
            if res:
                seg, _ = res
                full.extend(seg)
                curr_state = (seg[-1][0], seg[-1][1], 0)
            else:
                self.ax.set_title("Plan Failed")
                self.running=False; return

        if len(full) > 2:
            self.construct_spline(full)
            self.ax.set_title("Path Ready")
        else:
            self.ax.set_title("Path too short")

    def trigger_replan(self):
        print("!!! REPLANNING !!!")
        self.ax.set_title("!!! REPLANNING !!!")
        plt.pause(0.1)
        # Plan considering static + CURRENT dynamic positions as static
        all_obs = self.static_obs + self.dynamic_obs
        self.plan_path(all_obs)
        if self.path:
            self.car.s_prev = 0.0 # Reset follower
            self.ax.set_title("Replanned!")
        else:
            self.ax.set_title("Blocked!")

    def construct_spline(self, pts):
        f = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(np.array(p)-np.array(f[-1]))>0.5: f.append(p)
        self.path = SplinePath(f)
        s = np.linspace(0, self.path.s_max, 200)
        self.line.set_data([self.path.eval(si)[0] for si in s], [self.path.eval(si)[1] for si in s])

    def start(self, e):
        if self.path: self.running=True

    def update(self, frame):
        # 1. Update Dynamic Obstacles
        if self.cfg['moving']:
            for o in self.dynamic_obs:
                o.update(0.05, 40, 30)
                o.patch.set_visible(True)
        else:
            for o in self.dynamic_obs: o.patch.set_visible(False)

        if self.running and self.path:
            # 2. Collision Check & Replan
            if self.cfg['moving'] and self.replan_cooldown == 0:
                car_pos = self.car.state[0:2]
                for o in self.dynamic_obs:
                    if np.linalg.norm(car_pos - np.array([o.x, o.y])) < 4.0: 
                        self.trigger_replan()
                        self.replan_cooldown = 40
                        break
            if self.replan_cooldown > 0: self.replan_cooldown -= 1

            # 3. Control
            noisy = AckermannCar(self.p)
            noisy.state = self.car.state.copy()
            noisy.state[0]+=np.random.normal(0,0.05); noisy.state[1]+=np.random.normal(0,0.05); noisy.state[2]+=np.random.normal(0,0.02)
            noisy.s_prev = self.car.s_prev
            
            delta, tau, s_curr = run_smc(noisy, self.path, self.cp)
            self.car.s_prev = s_curr
            
            # Check Waypoint Progress
            if self.current_wp_idx < len(self.waypoints):
                tgt = self.waypoints[self.current_wp_idx]
                if np.hypot(self.car.state[0]-tgt[0], self.car.state[1]-tgt[1]) < 3.0:
                    self.current_wp_idx += 1

            # 4. Physics
            mu = 0.9
            if self.cfg['oil']:
                for o in self.oil_patches:
                    if o.contains(self.car.state[0], self.car.state[1]):
                        mu = 0.2; self.ax.set_title("Oil Slide!"); break
            
            ext = (0,0,0)
            if self.cfg['wind'] and frame%120 < 5: ext=(0,100,0)
            
            self.car.step(delta, tau, 0.05, ext, mu)
            
            if self.path.s_max - s_curr < 0.5:
                self.running=False; self.ax.set_title("Finished")

        # Draw Car
        x, y, th = self.car.state[0:3]
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