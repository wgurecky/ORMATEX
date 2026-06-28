##############################################################################
# Adjoint-method ODE parameter inference with a smooth observation reference
#
# Companion to ``ex_adjoint_inference.py``.  The key simplification: instead
# of discrete point observations (which inject Dirac-delta kicks into the
# adjoint state at each observation time), we replace the noisy data with a
# single smooth reference signal y_hat(t) -- a cubic spline interpolant of
# the observations -- and use a running-in-time squared-error loss
#
#     L(theta) = (1 / (2T)) * integral_{0}^{T} ||y(t; theta) - y_hat(t)||^2 dt
#
# Now dL/dy(t) = (y(t) - y_hat(t)) / T is a *smooth* function of t.  The
# adjoint becomes one continuous ODE
#
#     d lambda /dt  = -J_y(t,y)^T  lambda  -  (y(t) - y_hat(t)) / T
#     d g     /dt  = -J_theta(t,y,theta)^T lambda
#
# with terminal conditions lambda(T) = 0, g(T) = 0 and g(0) = dL/dtheta.
#
# Because the adjoint no longer has discrete jumps, the entire backward
# integration is a *single* call to ormatex' exprb2 integrator (no per-
# segment loop, no observation-index bookkeeping).
#
# JAX is used only to obtain   dF/dy   and   dF/dtheta
# via jax.jacfwd on the closed-form pure-jnp ``_frhs`` (not to differentiate
# through the integrator -- the adjoint method itself is the alternative to
# that).
#
# Run with the project pixi environment, e.g.:
#     pixi run python examples/ex_adjoint_smooth_inference.py
##############################################################################
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

jax.config.update("jax_enable_x64", True)

from ormatex_py.ode_sys import OdeSys, CustomJacLinOp
from ormatex_py.integrate_wrapper import integrate
n_step = 0


# ---------------------------------------------------------------------------
# Parameter-bearing, jax-traceable LV system (identical to the discrete
# observation example).  Repeated here so the file is self-contained.
# ---------------------------------------------------------------------------
class LotkaVolterra(OdeSys):
    alpha: jax.Array
    beta: jax.Array
    delta: jax.Array
    gamma: jax.Array

    def __init__(self, alpha, beta, delta, gamma):
        super().__init__()
        self.alpha = jnp.asarray(alpha, dtype=jnp.float64)
        self.beta = jnp.asarray(beta, dtype=jnp.float64)
        self.delta = jnp.asarray(delta, dtype=jnp.float64)
        self.gamma = jnp.asarray(gamma, dtype=jnp.float64)

    @jax.jit
    def _frhs(self, t, y, **kwargs):
        prey = self.alpha * y[0] - self.beta * y[0] * y[1]
        pred = self.delta * y[0] * y[1] - self.gamma * y[1]
        return jnp.array([prey, pred])


PARAM_NAMES = ("alpha", "beta", "delta", "gamma")


def with_params(theta):
    return LotkaVolterra(theta[0], theta[1], theta[2], theta[3])


# ---------------------------------------------------------------------------
# Forward solve on a fixed uniform grid; returns the full trajectory
# Y[0..K] as a stacked (K+1, n) jnp array.
# ---------------------------------------------------------------------------
def forward_solve(theta, y0, dt, n_steps, method="epi3"):
    sys = with_params(theta)
    t0 = 0.0
    y0_j = jnp.asarray(y0, dtype=jnp.float64)
    res = integrate(sys, y0_j, t0, dt, n_steps, method=method, tol_fdt=1e-12)
    return jnp.stack(list(res.y_res))


# ---------------------------------------------------------------------------
# Synthetic data: forward truth with high-accuracy dopri5, sample at the
# observation grid indices, add gaussian noise.
# ---------------------------------------------------------------------------
def gen_data(theta_true, y0, dt, n_steps, obs_stride, noise_sd, seed=0):
    sys = with_params(theta_true)
    t0 = 0.0
    y0_j = jnp.asarray(y0, dtype=jnp.float64)
    res = integrate(sys, y0_j, t0, dt, n_steps, method="dopri5")
    Y_true = np.asarray(jnp.stack(list(res.y_res)))
    obs_idx = np.arange(0, n_steps + 1, obs_stride)
    rng = np.random.default_rng(seed)
    y_obs = Y_true[obs_idx] + noise_sd * rng.standard_normal(Y_true[obs_idx].shape)
    return Y_true, obs_idx, y_obs


# ---------------------------------------------------------------------------
# Smooth observation reference: cubic-spline interpolant of the noisy
# observations, evaluated once on the forward grid.  Held constant across
# optimization iterations (the data does not change).
#
# This is the only addition vs. the discrete-observation example.  With
# y_hat(t) continuous and smooth (C^2), the loss gradient w.r.t. the forward
# state becomes (y(t) - y_hat(t)) / T -- no Dirac deltas, no jumps.
# ---------------------------------------------------------------------------
def build_reference(obs_idx, y_obs, n_steps, dt):
    t_grid = np.linspace(0.0, dt * n_steps, n_steps + 1)
    t_obs = t_grid[obs_idx]
    # Per-state-dim cubic spline (Cubicspline requires sorted ascending t).
    cs = CubicSpline(t_obs, y_obs, axis=0)
    y_ref = cs(t_grid)               # (K+1, n)
    return jnp.asarray(y_ref, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Running-in-time squared-error loss (right-Riemann approximation matches
# the adjoint's forcing term evaluated at each backward-step start).
#     L = (h / (2T)) sum_{k=1..K} ||Y[k] - y_ref[k]||^2
# ---------------------------------------------------------------------------
def loss_smooth(theta, y0, dt, n_steps, y_ref, T_total):
    Y = forward_solve(theta, y0, dt, n_steps)
    h = float(dt)
    diff = Y[1:] - y_ref[1:]
    return h * jnp.sum(diff * diff) / (2.0 * T_total)


# ---------------------------------------------------------------------------
# Augmented adjoint ODE system for the smooth-loss case.
#
# State z = (lambda, g) of size n + p, with the forward state-indexed
# Jacobians Jy_all[k] (n,n), Jt_all[k] (n,p), forward trajectory Y[k+1]
# (n,), and reference y_ref[k+1] (n,) all stored as equinox leaves.  The
# ODE is affine in z with a smooth additive forcing, sampled by the time
# argument t:
#
#     d lambda /dt = -J_y(k)^T lambda  -  (y_k - y_ref(k)) / T
#     d g     /dt = -J_theta(k)^T lambda
#
# where k indexes the forward-step we are currently undoing (k_lookup =
# round(t/h) matches the right-endpoint Riemann convention used in
# ``loss_smooth``: at backward-step start t = t_{k+1} the integrator
# samples once at that t and uses the Jacobian of forward step k -> k+1
# and tracks Y[k+1] - y_ref[k+1]).  Clip to the valid index range.
#
# Per-segment the RHS Jacobian in z is constant (independent of z), so
# ``CustomJacLinOp`` with the materialized A_k makes ``exprb2`` reproduce
# expm(dt*A_k) z + (particular-solution) * f exactly per step.
# ---------------------------------------------------------------------------
class SmoothAugAdjoint(OdeSys):
    Jy_all: jax.Array
    Jt_all: jax.Array
    Y: jax.Array
    y_ref: jax.Array
    n: int = eqx.field(static=True)
    p: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    h: float = eqx.field(static=True)
    T_total: float = eqx.field(static=True)

    def __init__(self, Jy_all, Jt_all, Y, y_ref, n, p, K, h, T_total):
        super().__init__()
        self.Jy_all = jnp.asarray(Jy_all, dtype=jnp.float64)
        self.Jt_all = jnp.asarray(Jt_all, dtype=jnp.float64)
        self.Y = jnp.asarray(Y, dtype=jnp.float64)
        self.y_ref = jnp.asarray(y_ref, dtype=jnp.float64)
        self.n = n
        self.p = p
        self.K = K
        self.h = float(h)
        self.T_total = float(T_total)

    @jax.jit
    def _frhs(self, t, z, **kwargs):
        # k_lookup points at the right endpoint of the forward-interval we
        # are currently undoing; the forward step itself used left-endpoint
        # Jacobian Jy[k_lookup - 1].
        k_lookup = jnp.clip(jnp.round(t / self.h).astype(int), 1, self.K)
        k_jac = k_lookup - 1
        Jy_k = self.Jy_all[k_jac]
        Jt_k = self.Jt_all[k_jac]
        y_k = self.Y[k_lookup]
        yref_k = self.y_ref[k_lookup]
        lam = z[:self.n]
        forcing = (y_k - yref_k) / self.T_total
        dlam = -Jy_k.T @ lam - forcing
        dg = -Jt_k.T @ lam
        return jnp.concatenate([dlam, dg])

#     @jax.jit
#     def _fjac(self, t, z, **kwargs):
#         k_lookup = jnp.clip(jnp.round(t / self.h).astype(int), 1, self.K)
#         k_jac = k_lookup - 1
#         Jy_k = self.Jy_all[k_jac]
#         Jt_k = self.Jt_all[k_jac]
#         A = jnp.zeros((self.n + self.p, self.n + self.p), dtype=jnp.float64)
#         A = A.at[:self.n, :self.n].set(-Jy_k.T)
#         A = A.at[self.n:, :self.n].set(-Jt_k.T)
#         fdt = jnp.zeros(z.shape)
#         return CustomJacLinOp(t, z, self.frhs, A, fdt, frhs_kwargs=kwargs)


# ---------------------------------------------------------------------------
# Adjoint gradient of the smooth loss.  Returns dL/dtheta as a numpy array.
#
# 1. Forward solve, store the full trajectory Y[0..K].
# 2. Vectorize via vmap: build Jy_all[k], Jt_all[k] for k = 0..K-1.
# 3. Build the SmoothAugAdjoint OdeSys with the per-step Jacobians, the
#    forward trajectory, and the pre-built smooth reference.
# 4. Single backward integrate() call: march the augmented state from T
#    back to 0 with dt = -h and nsteps = K.
# 5. Return the g-part of z at t=0 as dL/dtheta.
# ---------------------------------------------------------------------------
def adjoint_grad_smooth(theta, y0, dt, n_steps, y_ref, T_total):
    theta_j = jnp.asarray(theta, dtype=jnp.float64)
    n = 2
    p = theta_j.shape[0]
    K = n_steps
    h = float(dt)

    # 1. forward trajectory
    Y = forward_solve(theta_j, y0, dt, K)

    # 2. per-step Jacobians at forward-step left endpoints (t_k, Y[k]).
    sys = with_params(theta_j)
    t_grid = jnp.arange(K) * h
    Y_left = Y[:K]
    f_y = lambda t, y: sys._frhs(t, y)
    f_th = lambda t, y, th: with_params(th)._frhs(t, y)
    Jy_all = jax.vmap(jax.jacfwd(f_y, argnums=1), in_axes=(0, 0))(t_grid, Y_left)
    th_bcast = jnp.broadcast_to(theta_j, (K, p))
    Jt_all = jax.vmap(jax.jacfwd(f_th, argnums=2), in_axes=(0, 0, 0))(t_grid, Y_left, th_bcast)

    # 3. construct augmented adjoint OdeSys
    aug = SmoothAugAdjoint(Jy_all, Jt_all, Y, y_ref, n, p, K, h, T_total)

    # 4. single backward integration, z(T) = 0
    z0 = jnp.zeros(n + p, dtype=jnp.float64)
    t0_back = float(K * h)
    res = integrate(aug, z0, t0_back, -h, K, method="epi3", tol_fdt=1e-12)

    # 5. g-part at t=0 is dL/dtheta
    z_final = jnp.asarray(np.asarray(res.y_res[-1]).flatten(), dtype=jnp.float64)
    return np.asarray(z_final[n:])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(do_plot=True):
    # problem setup -- identical to ex_adjoint_inference.py so the two
    # examples produce directly comparable results.
    theta_true = np.array([1.1, 0.4, 0.5, 0.4], dtype=np.float64)
    y0 = jnp.array([0.1, 0.2], dtype=jnp.float64)
    dt = 0.05
    n_steps = 200
    obs_stride = 10
    noise_sd = 1e-1
    seed = 7
    T_total = dt * n_steps

    # synthetic data
    Y_true, obs_idx, y_obs = gen_data(theta_true, y0, dt, n_steps, obs_stride, noise_sd, seed=seed)
    print("generated %d noisy observations (stride=%d, noise_sd=%.0e)" %
          (obs_idx.shape[0], obs_stride, noise_sd))

    # build the smooth observation reference (constant across the optimisation)
    y_ref = build_reference(obs_idx, y_obs, n_steps, dt)

    # initial guess (deliberately off)
    theta0 = np.array([0.7, 0.7, 0.2, 0.7], dtype=np.float64)

    # optimize using the smooth adjoint gradient under positivity bounds.
    print("\n# optimizing (SLSQP, bounded, smooth adjoint)")
    bounds = [(1e-3, 10.0)] * 4
    bad_sentinel = 1.0e8

    def obj_and_grad(th):
        try:
            L = float(loss_smooth(jnp.asarray(th), y0, dt, n_steps, y_ref, T_total))
            if not np.isfinite(L):
                return bad_sentinel, np.zeros_like(th)
            g = adjoint_grad_smooth(th, y0, dt, n_steps, y_ref, T_total)
        except (AssertionError, FloatingPointError, ValueError):
            return bad_sentinel, np.zeros_like(th)
        if not np.isfinite(g).all():
            return bad_sentinel, np.zeros_like(th)
        return L, g

    def callback_f(x):
        # called at end of each optimizer step
        global n_step
        print(f"Step: {n_step}, x: %s" % str(x))
        n_step += 1

    res = minimize(obj_and_grad, theta0, jac=True, method="SLSQP",
                   bounds=bounds, callback=callback_f,
                   options={"ftol": 1e-8, "maxiter": 80, "disp": True})
    theta_hat = res.x

    print("\n# recovered parameters")
    print("%-7s %10s %10s" % ("param", "true", "recovered"))
    for nm, t, hh in zip(PARAM_NAMES, theta_true, theta_hat):
        print("%-7s %10.5f %10.5f" % (nm, t, hh))
    rel_err = np.linalg.norm(theta_hat - theta_true) / np.linalg.norm(theta_true)
    print("relative parameter error: %.3e" % rel_err)

    if do_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Y_hat = np.asarray(forward_solve(jnp.asarray(theta_hat), y0, dt, n_steps))
        y_ref_np = np.asarray(y_ref)
        t_all = np.linspace(0.0, dt * n_steps, n_steps + 1)
        t_obs = t_all[obs_idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_all, Y_true[:, 0], "k-", lw=1.0, alpha=0.6, label="prey (truth)")
        ax.plot(t_all, Y_true[:, 1], "k--", lw=1.0, alpha=0.6, label="pred (truth)")
        ax.plot(t_all, y_ref_np[:, 0], "-", color="tab:gray", lw=1.2, label=r"$\hat y$ prey (ref)")
        ax.plot(t_all, y_ref_np[:, 1], "--", color="tab:gray", lw=1.2, label=r"$\hat y$ pred (ref)")
        ax.plot(t_all, Y_hat[:, 0], "b-", lw=1.2, label="prey (fit)")
        ax.plot(t_all, Y_hat[:, 1], "b--", lw=1.2, label="pred (fit)")
        ax.plot(t_obs, y_obs[:, 0], "r.", ms=4.0, label="prey obs")
        ax.plot(t_obs, y_obs[:, 1], "g.", ms=4.0, label="pred obs")
        ax.set_xlabel("time")
        ax.set_ylabel("population")
        ax.set_title("Smooth-reference adjoint LV inference (rel err %.2e)" % rel_err)
        ax.grid(ls="--", alpha=0.4)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        out = "adjoint_smooth_inference_lv.png"
        fig.savefig(out, dpi=120)
        print("wrote %s" % out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true",
                        help="disable matplotlib plotting")
    args = parser.parse_args()
    main(do_plot=not args.no_plot)
