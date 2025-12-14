"""
Advanced Ackerman Simulation - Robustness Test Suite
- Planner: Multi-Point Hybrid A*
- Controller: SMC Path Following
- Viz: Filled Polygons, Rotating Wheels
- INTERACTIVE: Toggle Switches for Wind & Oil

Usage:
1. 'Rand Obs': Generate Walls (Grey) and Oil (Orange).
2. Use Checkbox to Enable/Disable 'Wind' or 'Oil' effects.
3. Click Points -> 'PLAN' -> 'EXECUTE'.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons
from dataclasses import dataclass
import heapq

# ---- SciPy spline ----
try:
    from scipy.interpolate import CubicSpline
except ImportError:
    raise ImportError("Install scipy: pip install scipy")

# ============================================================
# 1. MATH & GEOMETRY HELPERS
# ============================================================
def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s,  c]], float)

def clamp(x, lo, hi):
    return float(np.minimum(np.maximum(x, lo), hi))

def smc_smooth(z, eps):
    return np.tanh(z / eps)

def get_rect_verts(cx, cy, w, h, angle):
    """Generates vertices for a rotated rectangle."""
    pts = np.array([
        [-w/2, -h/2], [ w/2, -h/2],
        [ w/2,  h/2], [-w/2,  h/2]
    ])
    R = rot2(angle)
    return (R @ pts.T).T + np.array([cx, cy])

# ============================================================
# 2. OBSTACLE DEFINITIONS
# ============================================================
class Obstacle:
    def is_collision(self, x, y, margin): pass
    def draw(self, ax): pass

class CircleObs(Obstacle):
    def __init__(self, x, y, r):
        self.x, self.y, self.r = x, y, r
    def is_collision(self, x, y, margin):
        return (x - self.x)**2 + (y - self.y)**2 <= (self.r + margin)**2
    def draw(self, ax):
        patch = patches.Circle((self.x, self.y), self.r, fc='#555555', ec='k', zorder=5)
        ax.add_patch(patch)
        return patch

class RectObs(Obstacle):
    def __init__(self, x, y, w, h, angle_rad):
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.angle = angle_rad
        self.R_inv = rot2(-angle_rad)
    
    def is_collision(self, point_x, point_y, margin):
        dx = point_x - self.x
        dy = point_y - self.y
        local_pt = self.R_inv @ np.array([dx, dy])
        px, py = local_pt[0], local_pt[1]
        safe_w = (self.w / 2.0) + margin
        safe_h = (self.h / 2.0) + margin
        return abs(px) <= safe_w and abs(py) <= safe_h

    def draw(self, ax):
        verts = get_rect_verts(self.x, self.y, self.w, self.h, self.angle)
        patch = patches.Polygon(verts, closed=True, fc='#333333', ec='k', zorder=5)
        ax.add_patch(patch)
        return patch

class OilPatch(Obstacle):
    """Low friction zone. Planner ignores it, Physics respects it."""
    def __init__(self, x, y, r, mu_val=0.2):
        self.x, self.y, self.r = x, y, r
        self.mu = mu_val
    def is_collision(self, x, y, margin): return False # Invisible to planner
    def contains(self, x, y):
        return (x - self.x)**2 + (y - self.y)**2 <= self.r**2
    def draw(self, ax):
        # Orange dashed circle
        p = patches.Circle((self.x, self.y), self.r, fc='orange', alpha=0.4, ls='--', ec='orange', zorder=2)
        ax.add_patch(p)
        return p

# ============================================================
# 3. PARAMETERS
# ============================================================
@dataclass
class AckermannParams:
    L: float = 0.35; T: float = 0.28; r_w: float = 0.06
    Lf: float = 0.175; Lr: float = 0.175
    # Visualization
    body_length: float = 0.50; body_width:  float = 0.32
    wheel_length: float = 0.12; wheel_width:  float = 0.05
    # Physics
    m: float = 12.0; Iz: float = 0.8
    du: float = 4.0; dv: float = 8.0; dr: float = 2.0
    Cf: float = 150.0; Cr: float = 180.0
    mu_nominal: float = 0.9 # Normal road friction
    g: float = 9.81

@dataclass
class SMCParams:
    u_max: float = 3.5; ay_max: float = 2.5
    lam: float = 1.2; k_delta: float = 0.8; phi: float = 0.25
    k_u: float = 250.0; phi_u: float = 0.4; tau_max: float = 8.0
    steer_limit_deg: float = 45.0
    proj_window: float = 2.0; proj_samples: int = 30

# ============================================================
# 4. DYNAMICS & CONTROLLER
# ============================================================
class AckermannCar:
    def __init__(self, p: AckermannParams, x=0, y=0, th=0):
        self.p = p
        self.reset(x, y, th)

    def reset(self, x, y, th):
        self.state = np.array([x, y, th, 0.0, 0.0, 0.0])
        self.s_prev = 0.0
        self.delta_f_actual = 0.0 

    def step(self, delta_cmd, tau_cmd, dt, ext_force=(0,0,0), current_mu=None):
        x, y, th, u, v, r = self.state
        p = self.p

        # Determine effective friction
        mu = current_mu if current_mu is not None else p.mu_nominal

        # --- Standard Dynamics ---
        delta_axle = 0.5 * (delta_cmd['FL'] + delta_cmd['FR'])
        self.delta_f_actual = delta_axle

        u_eff = u if abs(u) > 0.5 else 0.5
        alpha_f = np.arctan2(v + p.Lf*r, u_eff) - delta_axle
        alpha_r = np.arctan2(v - p.Lr*r, u_eff)
        
        # Linear tire model
        Fy_f_lin = -p.Cf * alpha_f
        Fy_r_lin = -p.Cr * alpha_r

        # --- Friction Saturation (Oil Logic) ---
        # F_max = mu * Normal_Force
        F_max_f = mu * (p.m * p.g * (p.Lr / p.L)) 
        F_max_r = mu * (p.m * p.g * (p.Lf / p.L))

        Fy_f = np.clip(Fy_f_lin, -F_max_f, F_max_f)
        Fy_r = np.clip(Fy_r_lin, -F_max_r, F_max_r)
        
        # --- Forces ---
        fx_ext, fy_ext, mz_ext = ext_force
        
        Fx_tot = (tau_cmd['RL'] + tau_cmd['RR'])/p.r_w + fx_ext
        Fy_tot = (Fy_f * np.cos(delta_axle)) + Fy_r + fy_ext
        Mz_tot = (p.Lf * Fy_f * np.cos(delta_axle)) - (p.Lr * Fy_r) + mz_ext

        # Drag / Damping
        Fx_tot -= p.du * u
        Fy_tot -= p.dv * v
        Mz_tot -= p.dr * r

        # Newton-Euler
        du_dt = (Fx_tot / p.m) + v*r
        dv_dt = (Fy_tot / p.m) - u*r
        dr_dt = Mz_tot / p.Iz

        # Integration
        u += du_dt * dt
        v += dv_dt * dt
        r += dr_dt * dt

        dx = u*np.cos(th) - v*np.sin(th)
        dy = u*np.sin(th) + v*np.cos(th)
        dth = r

        x += dx * dt
        y += dy * dt
        th += dth * dt

        self.state = np.array([x, y, th, u, v, r])
        return self.state

class SplinePath:
    def __init__(self, waypoints):
        pts = np.array(waypoints, float)
        if len(pts) > 1:
            keep = [True] * len(pts)
            for i in range(1, len(pts)):
                if np.linalg.norm(pts[i] - pts[i-1]) < 0.1: keep[i] = False
            pts = pts[keep]
        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.hstack(([0.0], np.cumsum(ds)))
        self.s = s
        self.s_max = float(s[-1])
        if len(pts) < 2: raise ValueError("Path too short")
        elif len(pts) == 2:
             self.xs = CubicSpline(s, pts[:,0], bc_type='natural')
             self.ys = CubicSpline(s, pts[:,1], bc_type='natural')
        else:
             self.xs = CubicSpline(s, pts[:,0]); self.ys = CubicSpline(s, pts[:,1])

    def wrap_s(self, s): return float(np.clip(s, 0.0, self.s_max))
    def eval(self, s):
        s = self.wrap_s(s)
        x = float(self.xs(s)); y = float(self.ys(s))
        dx = float(self.xs(s, 1)); dy = float(self.ys(s, 1))
        ddx = float(self.xs(s, 2)); ddy = float(self.ys(s, 2))
        psi = float(np.arctan2(dy, dx))
        denom = (dx*dx + dy*dy)**1.5 + 1e-12
        kappa = float((dx*ddy - dy*ddx) / denom)
        return x, y, psi, kappa

def run_smc_step(car: AckermannCar, path: SplinePath, cp: SMCParams):
    x, y, psi, u, v, r = car.state
    w = cp.proj_window
    s0 = max(0.0, car.s_prev - w); s1 = min(path.s_max, car.s_prev + w)
    ss = np.linspace(s0, s1, cp.proj_samples)
    best_d2 = 1e9; best_s = car.s_prev
    for si in ss:
        px, py, _, _ = path.eval(si)
        d2 = (x - px)**2 + (y - py)**2
        if d2 < best_d2: best_d2 = d2; best_s = float(si)
    car.s_prev = best_s 
    
    xd, yd, psi_d, kappa = path.eval(best_s)
    ex = x - xd; ey = y - yd
    e_y = -np.sin(psi_d)*ex + np.cos(psi_d)*ey
    e_psi = wrap_to_pi(psi - psi_d)
    s_lat = e_psi + cp.lam * e_y
    u_ref = min(cp.u_max, float(np.sqrt(cp.ay_max / (abs(kappa) + 1e-3))))
    
    # Braking Logic
    dist_to_end = path.s_max - best_s
    if dist_to_end < 5.0: u_ref = min(u_ref, cp.u_max * (dist_to_end / 5.0))
    if dist_to_end < 0.2: u_ref = 0.0

    delta_ff = np.arctan(car.p.L * kappa)
    delta_axle = delta_ff - cp.k_delta * smc_smooth(s_lat, cp.phi)
    delta_axle = clamp(delta_axle, -np.deg2rad(cp.steer_limit_deg), np.deg2rad(cp.steer_limit_deg))
    
    if u_ref == 0.0 and abs(u) < 0.1: Fx_des = 0.0
    else: Fx_des = (car.p.du*u) - cp.k_u * smc_smooth((u - u_ref), cp.phi_u)
    tau = clamp(0.5 * car.p.r_w * Fx_des, -cp.tau_max, cp.tau_max)
    
    return {"FL": delta_axle, "FR": delta_axle}, {"RL": tau, "RR": tau}, best_s

# ============================================================
# 5. HYBRID A* PLANNER
# ============================================================
class HybridAStar:
    def __init__(self, x_lim, y_lim, cell_size=1.0, angle_res=15):
        self.cell_size = cell_size
        self.angle_res = np.deg2rad(angle_res)
        self.min_x, self.max_x = x_lim
        self.min_y, self.max_y = y_lim
        self.motion_primitives = [-35, -20, 0, 20, 35] 
        self.step_size = 1.5 
        self.collision_margin = 0.35 

    class Node:
        def __init__(self, x, y, yaw, cost, parent, steer):
            self.x, self.y, self.yaw = x, y, yaw
            self.cost, self.parent, self.steer = cost, parent, steer
        def __lt__(self, other): return self.cost < other.cost
            
    def get_grid_index(self, node):
        return (int(round(node.x / self.cell_size)),
                int(round(node.y / self.cell_size)),
                int(round(node.yaw / self.angle_res)))

    def is_valid(self, x, y, obstacles):
        if not (self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y):
            return False
        for obs in obstacles:
            if obs.is_collision(x, y, self.collision_margin):
                return False
        return True

    def plan_segment(self, start, goal_pose, obstacles):
        sx, sy, syaw = start
        gx, gy, gyaw = goal_pose
        start_node = self.Node(sx, sy, syaw, 0.0, None, 0.0)
        open_set = []; heapq.heappush(open_set, (0.0, start_node))
        closed_set = {}
        final_node = None; max_iter = 4000; itr = 0
        w_dist = 1.0; w_head = 2.5 

        while open_set and itr < max_iter:
            itr += 1
            _, current = heapq.heappop(open_set)
            dist_to_goal = np.hypot(current.x - gx, current.y - gy)
            angle_diff = abs(wrap_to_pi(current.yaw - gyaw))
            
            if dist_to_goal < 1.2 and angle_diff < np.deg2rad(60):
                final_node = current; break

            idx = self.get_grid_index(current)
            if idx in closed_set: continue
            closed_set[idx] = current

            for deg in self.motion_primitives:
                steer = np.deg2rad(deg)
                L = 0.35
                beta = np.arctan(0.5 * np.tan(steer))
                nx = current.x + self.step_size * np.cos(current.yaw + beta)
                ny = current.y + self.step_size * np.sin(current.yaw + beta)
                nyaw = wrap_to_pi(current.yaw + (self.step_size / L) * np.sin(beta))

                if self.is_valid(nx, ny, obstacles):
                    new_cost = current.cost + self.step_size
                    new_cost += abs(deg)*0.02 + abs(deg - current.steer)*0.05
                    h = w_dist*np.hypot(nx - gx, ny - gy) + w_head*abs(wrap_to_pi(nyaw - gyaw))
                    new_node = self.Node(nx, ny, nyaw, new_cost, current, deg)
                    heapq.heappush(open_set, (new_cost + h, new_node))
        
        if final_node is None: return None
        path = []
        n = final_node
        while n is not None:
            path.append([n.x, n.y]); n = n.parent
        return path[::-1], final_node.yaw

# ============================================================
# 6. INTERACTIVE SIM MANAGER
# ============================================================
class InteractiveSim:
    def __init__(self):
        self.p = AckermannParams()
        self.cp = SMCParams()
        self.car = AckermannCar(self.p, x=3, y=5, th=0)
        
        self.obstacles = [] # Solid obstacles
        self.oil_patches = [] # Oil patches
        self.obs_patches = [] # Viz patches
        self.waypoints = []; self.path = None; self.running = False; self.dt = 0.05
        self.planner = HybridAStar((0, 50), (0, 50))
        
        # --- Config States (Controlled by Checkbuttons) ---
        self.enable_wind = True
        self.enable_oil = True

        # Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(10, 9))
        plt.subplots_adjust(bottom=0.20) # Made room for checkbuttons
        self.ax.set_xlim(0, 40); self.ax.set_ylim(0, 30)
        self.ax.set_aspect('equal'); self.ax.grid(True, color='lightgray')
        self.ax.set_facecolor('#f0f0f0') 

        # --- Viz ---
        self.body_patch = patches.Polygon([[0,0],[0,0]], closed=True, fc='tab:red', ec='k', lw=1.5, zorder=10)
        self.ax.add_patch(self.body_patch)
        self.wheel_patches = []
        for _ in range(4):
            wp = patches.Polygon([[0,0],[0,0]], closed=True, fc='k', ec='k', zorder=11)
            self.ax.add_patch(wp); self.wheel_patches.append(wp)

        self.path_line, = self.ax.plot([], [], 'g--', lw=2, alpha=0.6)
        self.markers, = self.ax.plot([], [], 'r.', markersize=12, markeredgecolor='k')

        self.setup_buttons()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.ani = FuncAnimation(self.fig, self.update, interval=40, blit=False, cache_frame_data=False)
        
        self.ax.set_title("1. Config Disturbances -> 2. Rand Obs -> 3. Points -> 4. PLAN")
        self.gen_obstacles(None)

    def setup_buttons(self):
        # 1. Action Buttons
        ax_obs = plt.axes([0.1, 0.02, 0.15, 0.06])
        ax_plan = plt.axes([0.3, 0.02, 0.15, 0.06])
        ax_go = plt.axes([0.5, 0.02, 0.15, 0.06])
        ax_rst = plt.axes([0.7, 0.02, 0.15, 0.06])
        
        self.btn_obs = Button(ax_obs, 'Rand Obs', color='gray', hovercolor='skyblue')
        self.btn_obs.on_clicked(self.gen_obstacles)
        self.btn_plan = Button(ax_plan, 'PLAN', color='gray', hovercolor='skyblue')
        self.btn_plan.on_clicked(self.plan_route)
        self.btn_go = Button(ax_go, 'EXECUTE', color='gray', hovercolor='skyblue')
        self.btn_go.on_clicked(self.start_sim)
        self.btn_rst = Button(ax_rst, 'Reset', color='gray', hovercolor='skyblue')
        self.btn_rst.on_clicked(self.reset_car)
        
        # 2. Toggle Switches (CheckButtons)
        ax_check = plt.axes([0.02, 0.4, 0.12, 0.15], frameon=False) # Left side
        self.chk = CheckButtons(ax_check, ['Wind Gust', 'Oil Patches'], [True, True])
        self.chk.on_clicked(self.toggle_disturbances)
        
    def toggle_disturbances(self, label):
        if label == 'Wind Gust': self.enable_wind = not self.enable_wind
        elif label == 'Oil Patches': self.enable_oil = not self.enable_oil
        # If oil turned off/on, we should regenerate visual indication? 
        # For simplicity, we just ignore the physics in update loop.
        print(f"Disturbances: Wind={self.enable_wind}, Oil={self.enable_oil}")

    def draw_car(self):
        x, y, th = self.car.state[0:3]
        steer = self.car.delta_f_actual 
        p = self.p
        body_verts = get_rect_verts(x, y, p.body_length, p.body_width, th)
        self.body_patch.set_xy(body_verts)
        offsets = [[-p.Lr, p.T/2], [-p.Lr, -p.T/2], [p.Lf, p.T/2], [p.Lf, -p.T/2]]
        for i, off in enumerate(offsets):
            w_center = np.array([x, y]) + rot2(th) @ np.array(off)
            w_angle = th + (steer if i >= 2 else 0.0)
            w_verts = get_rect_verts(w_center[0], w_center[1], p.wheel_length, p.wheel_width, w_angle)
            self.wheel_patches[i].set_xy(w_verts)

    def gen_obstacles(self, event):
        self.running = False
        for p in self.obs_patches: p.remove()
        self.obs_patches = []
        self.obstacles = []
        self.oil_patches = []
        
        # Solid Rectangles
        for _ in range(8):
            ox, oy = np.random.uniform(8, 35), np.random.uniform(2, 28)
            obs = RectObs(ox, oy, np.random.uniform(1.5, 4), np.random.uniform(1.5, 4), np.random.uniform(0, 3.14))
            self.obstacles.append(obs)
            self.obs_patches.append(obs.draw(self.ax))

        # Solid Circles
        for _ in range(5):
            ox, oy = np.random.uniform(8, 35), np.random.uniform(2, 28)
            obs = CircleObs(ox, oy, np.random.uniform(0.8, 1.5))
            self.obstacles.append(obs)
            self.obs_patches.append(obs.draw(self.ax))

        # --- OIL PATCHES (New) ---
        for _ in range(5):
            ox, oy = np.random.uniform(8, 35), np.random.uniform(2, 28)
            obs = OilPatch(ox, oy, np.random.uniform(1.5, 3.0), mu_val=0.2)
            self.oil_patches.append(obs)
            self.obs_patches.append(obs.draw(self.ax))
            
        self.reset_car(None)

    def reset_car(self, event):
        self.running = False; self.car.reset(3, 5, 0)
        self.waypoints = []; self.path = None
        self.path_line.set_data([], []); self.markers.set_data([], [])
        self.draw_car()
        self.ax.set_title("Ready. Select Waypoints.")
        plt.draw()

    def on_click(self, event):
        if event.inaxes != self.ax or self.running: return
        self.waypoints.append((event.xdata, event.ydata))
        pts = np.array(self.waypoints)
        self.markers.set_data(pts[:,0], pts[:,1])
        self.ax.set_title(f"Points: {len(self.waypoints)} | Click 'PLAN'")
        plt.draw()

    def plan_route(self, event):
        if not self.waypoints: return
        self.ax.set_title("Planning...")
        plt.pause(0.01)
        full_path_pts = []
        curr = (self.car.state[0], self.car.state[1], self.car.state[2])

        for w in self.waypoints:
            goal = (w[0], w[1], 0) # Heuristic goal
            res = self.planner.plan_segment(curr, goal, self.obstacles) # Ignore oil for planning
            if res is None:
                self.ax.set_title("Planning Failed!"); return
            
            segment_pts, final_yaw = res
            if full_path_pts: full_path_pts.extend(segment_pts[1:])
            else: full_path_pts.extend(segment_pts)
            
            # Correct update of current state for next segment
            last_pt = segment_pts[-1]
            curr = (last_pt[0], last_pt[1], final_yaw)

        # Spline Smoothing
        filtered = [full_path_pts[0]]
        for p in full_path_pts[1:]:
             if np.linalg.norm(np.array(p)-np.array(filtered[-1])) > 0.6: filtered.append(p)
        if len(filtered) < 3: filtered = np.linspace(filtered[0], filtered[-1], 5)

        self.path = SplinePath(filtered)
        ss = np.linspace(0, self.path.s_max, 300)
        px = [self.path.eval(s)[0] for s in ss]; py = [self.path.eval(s)[1] for s in ss]
        self.path_line.set_data(px, py)
        self.ax.set_title("Path Planned! Click 'EXECUTE'")
        plt.draw()

    def start_sim(self, event):
        if self.path: self.running = True; self.ax.set_title("Executing...")

    def update(self, frame):
        if self.running and self.path:
            # 1. Sensor Noise
            noise_pos = np.random.normal(0, 0.10, 2) 
            noise_th  = np.random.normal(0, np.deg2rad(2.0))
            noisy_car = AckermannCar(self.p)
            noisy_car.state = self.car.state.copy()
            noisy_car.state[0] += noise_pos[0]
            noisy_car.state[1] += noise_pos[1]
            noisy_car.state[2] += noise_th
            noisy_car.s_prev = self.car.s_prev
            
            delta, tau, s_curr = run_smc_step(noisy_car, self.path, self.cp)
            self.car.s_prev = s_curr 

            # 2. Wind Gust (Controlled by CheckButton)
            ext_force = (0, 0, 0)
            if self.enable_wind and (frame % 120 < 5): 
                ext_force = (0, 100.0, 0) # Kick!

            # 3. Oil Patch Logic (Controlled by CheckButton)
            current_mu = 0.9
            if self.enable_oil:
                for oil in self.oil_patches:
                    if oil.contains(self.car.state[0], self.car.state[1]):
                        current_mu = oil.mu # Low friction
                        self.ax.set_title("⚠️ OIL SLICK! ⚠️")
                        break

            # 4. Physics Step
            self.car.step(delta, tau, self.dt, ext_force=ext_force, current_mu=current_mu)
            
            if self.path.s_max - s_curr < 0.3 and self.car.state[3] < 0.1:
                self.running = False; self.ax.set_title("Reached!")
                
        self.draw_car()
        return self.body_patch, *self.wheel_patches

if __name__ == "__main__":
    sim = InteractiveSim()
    plt.show()