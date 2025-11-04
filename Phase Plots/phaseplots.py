import numpy as np
import matplotlib.pyplot as plt

def _wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def compute_errors(q_ref, q, qdot_ref, qdot, wrap_heading=True):
    q_ref = np.asarray(q_ref); q = np.asarray(q)
    qdot_ref = np.asarray(qdot_ref); qdot = np.asarray(qdot)
    assert q_ref.shape == q.shape == qdot_ref.shape == qdot.shape, "All inputs must have same shape (N,3)."

    e = q_ref - q
    if wrap_heading:
        e[:, 2] = _wrap_pi(e[:, 2])

    e_dot = qdot_ref - qdot
    return e, e_dot

def _moving_avg(x, k=5):
    if k <= 1: return x
    # simple centered moving average (edge-handling by reflection)
    pad = k//2
    xp = np.pad(x, (pad, pad), mode='edge')
    w = np.ones(k)/k
    return np.convolve(xp, w, mode='valid')

def plot_phase_portraits(q_ref, q, qdot_ref, qdot,
                         wrap_heading=True,
                         add_vector_field=False,
                         kp=5.0, kd=2.0,
                         titles=(r"$e_x$", r"$e_y$", r"$e_\theta$"),
                         force_center=True,
                         smooth_rate_window=1,   # set >1 to smooth the plotted e_dot
                         show_stats=True):
    """
    Draw three phase portraits (e_i vs e_dot_i). Optionally:
      - overlays a PD vector field e' = e_dot, e_dot' = -kd*e_dot - kp*e
      - forces symmetric axes around 0 for visual convergence
      - lightly smooths the displayed e_dot to reduce numerical noise
      - prints convergence stats (initial/final norms)
    """
    e, e_dot = compute_errors(q_ref, q, qdot_ref, qdot, wrap_heading=wrap_heading)

    # Convergence stats
    if show_stats:
        e_norm0   = np.linalg.norm(e[0])
        ed_norm0  = np.linalg.norm(e_dot[0])
        e_normN   = np.linalg.norm(e[-1])
        ed_normN  = np.linalg.norm(e_dot[-1])
        print(f"[Phase Portrait] ||e(0)||={e_norm0:.3g}, ||e(T)||={e_normN:.3g} | "
              f"||edot(0)||={ed_norm0:.3g}, ||edot(T)||={ed_normN:.3g}")

    # Optionally smooth e_dot for PLOTTING (does not change stats above)
    e_dot_disp = e_dot.copy()
    if smooth_rate_window and smooth_rate_window > 1:
        for i in range(3):
            e_dot_disp[:, i] = _moving_avg(e_dot[:, i], k=smooth_rate_window)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axs):
        # main curve
        ax.plot(e[:, i], e_dot_disp[:, i], lw=1.5)
        ax.set_xlabel(titles[i])
        ax.set_ylabel(r"$\dot{e_%s}$" % titles[i][3:-1] if i < 2 else r"$\dot{e_\theta}$")

        # axes + origin marker
        ax.axhline(0, color='k', lw=0.8)
        ax.axvline(0, color='k', lw=0.8)
        ax.plot(0, 0, 'k+', ms=8, mew=1.5)  # show the equilibrium explicitly

        # start / end
        ax.plot(e[0, i],  e_dot_disp[0, i],  'o', ms=5, label='start')
        ax.plot(e[-1, i], e_dot_disp[-1, i], 'x', ms=6, label='end')

        # vector field (PD surrogate)
        if add_vector_field:
            # field bounds: use symmetric limits around zero that cover data
            ex_max  = max(abs(np.min(e[:, i])), abs(np.max(e[:, i])), 1e-6)
            ed_max  = max(abs(np.min(e_dot[:, i])), abs(np.max(e_dot[:, i])), 1e-6)
            ex_lim  = 1.2 * ex_max
            ed_lim  = 1.2 * ed_max
            ee  = np.linspace(-ex_lim, ex_lim, 25)
            dee = np.linspace(-ed_lim, ed_lim, 25)
            E, DE = np.meshgrid(ee, dee)

            dE_dt  = DE
            dDE_dt = -kd*DE - kp*E
            ax.quiver(E, DE, dE_dt, dDE_dt, angles="xy", width=0.002, alpha=0.55)

        # force symmetric limits around 0 so convergence is visually obvious
        if force_center:
            ex_max  = max(abs(np.min(e[:, i])), abs(np.max(e[:, i])), 1e-6)
            ed_max  = max(abs(np.min(e_dot_disp[:, i])), abs(np.max(e_dot_disp[:, i])), 1e-6)
            ax.set_xlim([-1.1*ex_max, 1.1*ex_max])
            ax.set_ylim([-1.1*ed_max, 1.1*ed_max])

        ax.grid(True, ls=':', alpha=0.5)
        ax.set_title(f"Phase portrait: {titles[i]} vs rate")

    fig.tight_layout()
    plt.show()


# --- Optional: quick demo with synthetic data (delete if not needed) ---
if __name__ == "__main__":
    # Fake example: small spiraling errors for illustration
    N = 2000
    t = np.linspace(0, 10, N)
    q_ref = np.column_stack([0.5*np.sin(0.3*t), 0.5*np.cos(0.25*t), 0.2*np.sin(0.2*t)])
    q     = q_ref - np.column_stack([np.exp(-0.3*t)*np.sin(1.2*t),
                                     np.exp(-0.25*t)*np.cos(1.0*t),
                                     0.5*np.exp(-0.2*t)*np.sin(0.8*t)])
    qdot_ref = np.gradient(q_ref, t, axis=0)
    qdot     = np.gradient(q, t, axis=0)

    plot_phase_portraits(q_ref, q, qdot_ref, qdot,
                         wrap_heading=True,
                         add_vector_field=True,
                         kp=5.0, kd=2.0)
