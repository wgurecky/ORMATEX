"""
PyMFEM-based port of `advection_diffusion_1d.py`.

Assemble matrices for a spectral finite element discretization of a 1D
advection-diffusion equation with optional SUPG stabilization, using the
PyMFEM (MFEM 4.x) Python bindings instead of scikit-fem.

The implementation mirrors `AdDiffSEM` from `advection_diffusion_1d.py`:
  - GLL (Gauss-Lobatto) quadrature is forced on every domain integrator so
    the mass matrix is *exactly* diagonal when combined with the H1 nodal
    (Gauss-Lobatto) basis of order p.
  - The conservative advection-diffusion operator with SUPG stabilization is
    composed from built-in MFEM integrators:
        A = DiffusionIntegrator(nu_eff)
          + ConservativeConvectionIntegrator([vel], a=+1)        [-<vel*u, grad v>]
          + BoundaryMassIntegrator(vel . n)        (non-periodic only, Robin)
        M = MassIntegrator(1) + ConservativeConvectionIntegrator([tau_u . vel], a=-1)
        b = DomainLFIntegrator(src) + DomainLFGradIntegrator([-tau_u . vel . src])
    where nu_eff = nu + tau_u * vel**2 and tau_u = tau * h / (2 * vel).
  - Dirichlet BC: left boundary only (essential), handled by manually zeroing
    the rows of essential DoFs and setting the diagonal to 1 (matches
    `skfem.enforce(..., overwrite=True)`, which touches rows only).
  - Periodic BC: MFEM's 1D periodic mesh construction gives negative Jacobians
    on the wrap element, so periodicity is imposed by merging the two endpoint
    DoFs at the scipy level (a projection P @ A @ P.T).

Supports spatially varying `vel` and `src` through the `field_fns` dict; when
absent, constant coefficients (from `params`) are used.
"""

import numpy as np
import scipy as sp
import scipy.sparse

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsp

import mfem.ser as mfs

from ormatex_py.progression import element_line_pp_nodal as el_nodal
from ormatex_py.ode_sys import MatrixLinOp, DiagLinOp
from ormatex_py.ode_sys import OdeSplitSys
from ormatex_py import integrate_wrapper


# ---------------------------------------------------------------------------
# Default field functions (match the originals)
# ---------------------------------------------------------------------------

def src_f(x, **kwargs):
    """Source term. x: array, returns scalar/array."""
    return 0.0 * np.asarray(x).sum(axis=0) if np.asarray(x).ndim else np.asarray(x) * 0.0


def vel_f(x, vel, **kwargs):
    """Velocity field. x: array, returns vel broadcast."""
    return 0.0 * np.asarray(x) + vel


def tau_upwind(tau, h, vel):
    """SUPG upwind parameter: tau * h / (2 * vel)."""
    return tau * h / (2.0 * vel)


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def make_1d_mesh(nrefs: int, periodic: bool = False) -> mfs.Mesh:
    """
    Build a uniform 1D mesh on [0, 1] with `2**nrefs` elements.
    Boundary attributes: left=1, right=2.  For periodic, the mesh is built
    without boundary elements (endpoints are merged at DoF level later).
    """
    ne = 2 ** nrefs
    nv = ne + 1
    if periodic:
        # ponytail: don't auto-generate bdr elements—periodicity via DoF merge
        mesh = mfs.Mesh(1, nv, ne, 0)
        for x in np.linspace(0.0, 1.0, nv):
            mesh.AddVertex(float(x), 0.0, 0.0)
        for i in range(ne):
            mesh.AddSegment(i, i + 1, 1)
        mesh.FinalizeTopology(False)
    else:
        mesh = mfs.Mesh(1, nv, ne, 2)
        for x in np.linspace(0.0, 1.0, nv):
            mesh.AddVertex(float(x), 0.0, 0.0)
        for i in range(ne):
            mesh.AddSegment(i, i + 1, 1)
        mesh.AddBdrPoint(0, 1)   # left boundary, attr 1
        mesh.AddBdrPoint(ne, 2)   # right boundary, attr 2
        mesh.FinalizeTopology(False)
    return mesh


# ---------------------------------------------------------------------------
# GLL quadrature rule construction
# ---------------------------------------------------------------------------

def make_gll_ir(p: int) -> mfs.IntegrationRule:
    """
    Build an MFEM IntegrationRule from the Gauss-Lobatto points and weights
    on the reference segment [0, 1], reusing the project's GLL table.
    """
    X, W = el_nodal.GLL_quad()(p + 1)
    pts = X[0]
    n = len(pts)
    ir = mfs.IntegrationRule(n)
    for i in range(n):
        ir.IntPoint(i).Set1w(float(pts[i]), float(W[i]))
    return ir


# ---------------------------------------------------------------------------
# PyMFEM coefficient wrappers (for spatially varying fields)
# ---------------------------------------------------------------------------

class _ScalarFieldCoeff(mfs.PyCoefficient):
    """Wrap a Python callable f(x, **params) -> scalar into an MFEM Coefficient."""
    def __init__(self, fn, params):
        super().__init__()
        self._fn = fn
        self._params = params

    def EvalValue(self, V):
        x = np.array([V[0]])
        return float(np.asarray(self._fn(x=x, **self._params)).flatten()[0])


class _VectorFieldCoeff(mfs.VectorPyCoefficient):
    """Wrap a Python callable f(x, **params) -> 1D vector into an MFEM VectorCoefficient."""
    def __init__(self, fn, params):
        super().__init__(1)
        self._fn = fn
        self._params = params

    def EvalValue(self, V):
        x = np.array([V[0]])
        val = np.asarray(self._fn(x=x, **self._params)).flatten()
        return [float(val[0])]


class _VelDotNCoeff(mfs.PyCoefficient):
    """Coefficient (vel . n) on boundary. In 1D: n=-1 at x~0, n=+1 at x~1."""
    def __init__(self, vel_fn, params):
        super().__init__()
        self._vel_fn = vel_fn
        self._params = params

    def EvalValue(self, V):
        x = V[0]
        # ponytail: 1D only—n = sign(x - 0.5). For higher dims use face Jacobian.
        n = 1.0 if x > 0.5 else -1.0
        vel_val = float(np.asarray(self._vel_fn(x=np.array([x]), **self._params)).flatten()[0])
        return vel_val * n


def _make_scalar_coeff(fn_or_val, params, is_vector_field=False):
    """
    Return (ConstantCoefficient or VectorConstantCoefficient) if fn_or_val is a
    scalar, else a PyCoefficient/VectorPyCoefficient wrapping the callable.
    """
    if callable(fn_or_val):
        if is_vector_field:
            return _VectorFieldCoeff(fn_or_val, params)
        else:
            return _ScalarFieldCoeff(fn_or_val, params)
    else:
        if is_vector_field:
            return mfs.VectorConstantCoefficient(mfs.Vector([float(fn_or_val)]))
        else:
            return mfs.ConstantCoefficient(float(fn_or_val))


def _make_tau_u_coeff(tau, h, vel_callable, vel_const, params):
    """Build the tau_upwind coefficient. If vel is spatially varying, tau_u(x)=tau*h/(2*vel(x))."""
    if callable(vel_callable):
        def tau_u_fn(x, **kw):
            v = np.asarray(vel_callable(x=x, **kw)).flatten()
            return tau * h / (2.0 * v)
        return _ScalarFieldCoeff(tau_u_fn, params)
    else:
        return mfs.ConstantCoefficient(float(tau * h / (2.0 * vel_const)))


# ---------------------------------------------------------------------------
# DenseMatrix / Vector extraction helpers
# ---------------------------------------------------------------------------

def _spmat_to_scipy(A: mfs.SparseMatrix) -> sp.sparse.csr_matrix:
    Ip = np.asarray(A.GetIArray())
    Jp = np.asarray(A.GetJArray())
    Dp = np.asarray(A.GetDataArray())
    return sp.sparse.csr_matrix((Dp, Jp, Ip), shape=(A.NumRows(), A.NumCols()))


def _vec_to_np(v: mfs.Vector) -> np.ndarray:
    return np.asarray(v.GetDataArray()).copy()


# ---------------------------------------------------------------------------
# Periodic DoF merge
# ---------------------------------------------------------------------------

def _merge_periodic(mat: sp.sparse.csr_matrix, b: np.ndarray, xs: np.ndarray,
                    dof_left: int, dof_right: int):
    """
    Merge two endpoint DoFs (dof_right into dof_left) via a projection matrix P.
    Returns (merged_csr, merged_b, merged_xs) of size (n-1).
    P[merged_0, dof_left] = 1, P[merged_0, dof_right] = 1, identity elsewhere.
    """
    n = mat.shape[0]
    P = np.zeros((n - 1, n))
    P[0, dof_left] = 1.0
    P[0, dof_right] = 1.0
    row = 1
    for i in range(n):
        if i in (dof_left, dof_right):
            continue
        P[row, i] = 1.0
        row += 1
    P_csr = sp.sparse.csr_matrix(P)
    merged = P_csr @ mat @ P_csr.T
    merged_b = P @ b
    merged_xs = P @ xs
    return merged, merged_b, merged_xs


def _enforce_dirichlet(mat: sp.sparse.csr_matrix, b: np.ndarray,
                       ess_dofs: np.ndarray) -> sp.sparse.csr_matrix:
    """
    Zero the rows of `ess_dofs`, set diagonal entries to 1, and zero the
    corresponding b entries. Matches `skfem.enforce(..., overwrite=True)`,
    which only touches rows (columns of ess dofs are preserved in interior rows).
    """
    mat = mat.tolil(copy=True)
    b = b.copy()
    for d in ess_dofs:
        mat[d, :] = 0.0
        mat[d, d] = 1.0
        b[d] = 0.0
    return mat.tocsr(), b


# ---------------------------------------------------------------------------
# AdDiffSEMMfem — mirrors AdDiffSEM
# ---------------------------------------------------------------------------

class AdDiffSEMMfem:
    """
    Assemble matrices for spectral finite element discretization of a 1D
    advection-diffusion equation using PyMFEM.

    p: polynomial basis order (H1 Gauss-Lobatto nodal basis)
    nu: physical diffusion coefficient
    vel: advection velocity (scalar constant or spatially varying via field_fns)
    tau: SUPG stabilization scale. 0 for no stabilization.
    """

    def __init__(self, mesh: mfs.Mesh, p: int = 1, field_fns: dict = {},
                 params: dict = {}, **kwargs):
        self.params = {
            "nu": params.get("nu", 5e-3),
            "vel": params.get("vel", 1.0),
            "tau": params.get("tau", 0.0),
        }
        self.p = p
        self.mesh = mesh

        # Determine velocity / source callables
        self._vel_fn = field_fns.get("vel_f", None)
        self._src_fn = field_fns.get("src_f", None)

        # FE collection + space
        # H1 with GaussLobatto basis (default btype) → nodal basis on GLL points
        self.fec = mfs.H1_FECollection(p, 1)
        self.fes = mfs.FiniteElementSpace(mesh, self.fec)
        mesh.SetNodalFESpace(self.fes)

        # GLL integration rule (exact diagonal mass)
        self.gll_ir = make_gll_ir(p)

        # Uniform element size h (ponytail: uniform mesh assumption; for non-uniform
        # use element-level Weight in a custom integrator)
        self.h = 1.0 / mesh.GetNE()

        self.dirichlet_bd = None
        self._periodic = (mesh.GetNBE() == 0)
        if not self._periodic:
            # Mark left boundary (attr 1) as Dirichlet
            marker = mfs.intArray([1, 0])
            ess = mfs.intArray()
            self.fes.GetEssentialTrueDofs(marker, ess)
            self.dirichlet_bd = np.asarray(ess.GetDataArray()).copy() if ess.Size() > 0 else np.array([], dtype=int)

    def _build_coefficients(self, **kwargs):
        """Build all MFEM coefficients needed for A, M, b."""
        params = {**self.params, **kwargs}
        vel_const = params["vel"]
        nu = params["nu"]
        tau = params["tau"]

        # Velocity: field or constant; the "raw" callable or scalar
        if self._vel_fn is not None:
            vel_for_coeff = partial(self._vel_fn, **params)
            vel_is_callable = True
        else:
            vel_for_coeff = vel_const
            vel_is_callable = False

        # tau_upwind
        if callable(vel_for_coeff):
            def tau_u_fn(x, **kw):
                v = np.asarray(self._vel_fn(x=x, **kw)).flatten()
                return tau * self.h / (2.0 * v)
            tau_u_val = None  # it's a field
        else:
            tau_u_val = tau * self.h / (2.0 * vel_const) if vel_const != 0 else 0.0

        # nu_eff = nu + tau_u * vel^2  (SUPG advection stabilization = added diffusion)
        if callable(vel_for_coeff) and tau != 0.0:
            def nu_eff_fn(x, **kw):
                v = np.asarray(self._vel_fn(x=x, **kw)).flatten()
                tu = tau * self.h / (2.0 * v)
                return nu + tu * v**2
            nu_eff = _ScalarFieldCoeff(nu_eff_fn, params)
        else:
            nu_eff = mfs.ConstantCoefficient(float(nu + tau_u_val * vel_const**2))

        # Velocity vector coefficient for advection
        vel_vec = _make_scalar_coeff(vel_for_coeff if vel_is_callable else vel_const,
                                     params, is_vector_field=True)

        # tau_u * vel vector coefficient for mass SUPG
        if tau != 0.0:
            if callable(vel_for_coeff):
                def tau_u_vel_fn(x, **kw):
                    v = np.asarray(self._vel_fn(x=x, **kw)).flatten()
                    return tau * self.h / (2.0 * v) * v
                tau_u_vel = _VectorFieldCoeff(tau_u_vel_fn, params)
            else:
                tau_u_vel = mfs.VectorConstantCoefficient(mfs.Vector([float(tau_u_val * vel_const)]))
        else:
            tau_u_vel = mfs.VectorConstantCoefficient(mfs.Vector([0.0]))

        # Source
        if self._src_fn is not None:
            src = _ScalarFieldCoeff(partial(self._src_fn, **params), params)
            src_const_val = None
        else:
            src = mfs.ConstantCoefficient(0.0)
            src_const_val = 0.0

        # SUPG RHS correction: -tau_u * vel * src  (vector coefficient)
        if tau != 0.0:
            if self._src_fn is not None or callable(vel_for_coeff):
                def supg_rhs_fn(x, **kw):
                    v = np.asarray((self._vel_fn or (lambda x, **k: k["vel"]))(x=x, **kw)).flatten()
                    if self._src_fn is not None:
                        s = np.asarray(self._src_fn(x=x, **kw)).flatten()
                    else:
                        s = np.zeros_like(v)
                    tu = (tau * self.h / (2.0 * v))[0]
                    return [-float(tu * v[0] * s[0])]
                supg_rhs = _VectorFieldCoeff(supg_rhs_fn, params)
            else:
                supg_rhs = mfs.VectorConstantCoefficient(mfs.Vector([-float(tau_u_val * vel_const * src_const_val)]))
        else:
            supg_rhs = mfs.VectorConstantCoefficient(mfs.Vector([0.0]))

        # Robin: vel . n on boundary
        if not self._periodic:
            veldotn = _VelDotNCoeff(vel_for_coeff if vel_is_callable else (lambda x, **kw: vel_const),
                                     params)
        else:
            veldotn = None

        return {
            "nu_eff": nu_eff,
            "vel_vec": vel_vec,
            "tau_u_vel": tau_u_vel,
            "src": src,
            "supg_rhs": supg_rhs,
            "veldotn": veldotn,
        }

    def assemble(self, **kwargs):
        """
        Assemble A (advection-diffusion + SUPG), M (mass + SUPG), b (RHS + SUPG).
        Returns (jA, jMl, jb) as jax arrays (matching AdDiffSEM.assemble).
        """
        coeffs = self._build_coefficients(**kwargs)

        # ---- Assemble A ----
        bfA = mfs.BilinearForm(self.fes)
        diff_int = mfs.DiffusionIntegrator(coeffs["nu_eff"], self.gll_ir)
        bfA.AddDomainIntegrator(diff_int)
        conv_int = mfs.ConservativeConvectionIntegrator(coeffs["vel_vec"], 1.0)
        conv_int.SetIntRule(self.gll_ir)
        bfA.AddDomainIntegrator(conv_int)
        if not self._periodic:
            bfA.AddBoundaryIntegrator(mfs.BoundaryMassIntegrator(coeffs["veldotn"]))
        bfA.Assemble()
        bfA.Finalize()

        # ---- Assemble M ----
        bfM = mfs.BilinearForm(self.fes)
        bfM.AddDomainIntegrator(mfs.MassIntegrator(mfs.ConstantCoefficient(1.0), self.gll_ir))
        if self.params["tau"] != 0.0:
            supg_m_int = mfs.ConservativeConvectionIntegrator(coeffs["tau_u_vel"], -1.0)
            supg_m_int.SetIntRule(self.gll_ir)
            bfM.AddDomainIntegrator(supg_m_int)
        bfM.Assemble()
        bfM.Finalize()

        # ---- Assemble b ----
        lf = mfs.LinearForm(self.fes)
        lf.AddDomainIntegrator(mfs.DomainLFIntegrator(coeffs["src"]))
        if self.params["tau"] != 0.0:
            lf_grad = mfs.DomainLFGradIntegrator(coeffs["supg_rhs"])
            lf_grad.SetIntRule(self.gll_ir)
            lf.AddDomainIntegrator(lf_grad)
        lf.Assemble()

        # ---- Extract raw arrays ----
        A_sp = _spmat_to_scipy(bfA.SpMat())
        M_sp = _spmat_to_scipy(bfM.SpMat())
        b_np = _vec_to_np(lf)

        # ---- DoF coordinates ----
        nodes = self.mesh.GetNodes()
        xs = _vec_to_np(nodes)

        # ---- Boundary conditions ----
        if self._periodic:
            # Merge endpoint DoFs (dof for x=0 and x=1)
            dof_left = int(np.argmin(xs))
            dof_right = int(np.argmax(xs))
            A_sp, b_np, xs = _merge_periodic(A_sp, b_np, xs, dof_left, dof_right)
            # M is still full-size; merge it with a size-matching dummy b/xs.
            M_sp, _, _ = _merge_periodic(M_sp, np.zeros(M_sp.shape[0]),
                                         np.zeros(M_sp.shape[0]), dof_left, dof_right)
        elif self.dirichlet_bd is not None and len(self.dirichlet_bd) > 0:
            # Dirichlet on left only: zero rows/cols of ess dofs, set diag=1, b=0.
            # Matches skfem.enforce(A, b, D=..., overwrite=True).
            A_sp, b_np = _enforce_dirichlet(A_sp, b_np, self.dirichlet_bd)
            b_zero = np.zeros_like(b_np)
            M_sp, _ = _enforce_dirichlet(M_sp, b_zero, self.dirichlet_bd)

        # ---- Mass diagonal (lumped row-sum, matches `M @ ones` in the original) ----
        Ml = np.asarray(M_sp.sum(axis=1)).flatten().copy()

        # ---- Provide jax arrays ----
        jMl = jnp.asarray(Ml)
        jA = jsp.BCOO.from_scipy_sparse(A_sp.tocoo())
        jb = jnp.asarray(b_np)

        return jA, jMl, jb

    def collocation_points(self):
        """Return DoF x-coordinates as jax array."""
        nodes = self.mesh.GetNodes()
        xs = _vec_to_np(nodes)
        if self._periodic:
            dof_left = int(np.argmin(xs))
            dof_right = int(np.argmax(xs))
            _, _, xs = _merge_periodic(sp.sparse.identity(len(xs)), np.zeros_like(xs), xs,
                                       dof_left, dof_right)
        return jnp.asarray(xs)

    def ode_sys(self, **kwargs):
        return AffineLinearSEMMfem(self, **kwargs)


# ---------------------------------------------------------------------------
# ODE System classes (mirror AffineLinearSEM / NonautonomousSEM)
# ---------------------------------------------------------------------------

class AffineLinearSEMMfem(OdeSplitSys):
    """
    ODE System associated with an affine-linear sparse Jacobian problem,
    using the PyMFEM assembler. The Jacobian is constant in t and u.
    """
    A: jsp.JAXSparse
    Ml: jax.Array
    b: jax.Array
    dirichlet_bd: np.ndarray

    def __init__(self, sys_assembler: AdDiffSEMMfem, *args, **kwargs):
        self.A, self.Ml, self.b = sys_assembler.assemble(**kwargs)

        if sys_assembler.dirichlet_bd is not None and len(sys_assembler.dirichlet_bd) > 0:
            self.dirichlet_bd = sys_assembler.dirichlet_bd
        else:
            self.dirichlet_bd = np.array([], dtype=int)

        super().__init__()

    @jax.jit
    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        f = (self.b - self.A @ u) / self.Ml
        f = f.at[self.dirichlet_bd].set(0.0)
        return f

    @jax.jit
    def _fl(self, t: float, u: jax.Array, **kwargs):
        return MatrixLinOp(-self.A / self.Ml[:, None])

    def _fm(self, t: float, u: jax.Array, **kwargs):
        return DiagLinOp(self.Ml)


class NonautonomousSEMMfem(AffineLinearSEMMfem):
    """
    Same as AffineLinearSEMMfem but with nonautonomous Dirichlet boundary
    conditions (time-dependent boundary data).
    """

    @jax.jit
    def dirichlet_dt(self, t: float):
        wc, ww = 0.3, 0.05
        vel = 0.5
        dirichlet_fun = \
            lambda time: jnp.exp(-(torus_distance(0.0, (wc + time * vel)) / (2 * ww)) ** 2.0)
        return jax.grad(dirichlet_fun)(t)

    @jax.jit
    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        f = (self.b - self.A @ u) / self.Ml
        f = f.at[self.dirichlet_bd].set(self.dirichlet_dt(t))
        return f


def torus_distance(x, xp):
    """Distance of two points on a torus (up to equivalence)."""
    dx = jnp.abs(x % 1 - xp % 1)
    return jnp.where(dx > 0.5, 1.0 - dx, dx)


# ---------------------------------------------------------------------------
# Main / CLI (mirrors advection_diffusion_1d.py __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import argparse

    jax.config.update("jax_enable_x64", True)
    print(f"Running on {jax.devices()}.")

    parser = argparse.ArgumentParser()
    parser.add_argument("-ic", help="one of [square, zero, gauss]", type=str, default="gauss")
    parser.add_argument("-mr", help="mesh refinement", type=int, default=7)
    parser.add_argument("-tau", help="supg stabilization constant. 0 for no supg stab.", type=float, default=0.0)
    parser.add_argument("-nu", help="physical diffusion coefficient. 0 for no diffusion.", type=float, default=1.0e-16)
    parser.add_argument("-p", help="basis order", type=int, default=2)
    parser.add_argument("-method", help="time step method", type=str, default="epi3")
    parser.add_argument("-phikv_method", help="time step method", type=str, default="taylor")
    parser.add_argument("-spec_method", help="time step method", type=str, default="arnoldi")
    parser.add_argument("-pfd_method", help="partial frac decomp method", type=str, default="CN")
    parser.add_argument("-nsteps", help="number of steps", type=int, default=10)
    parser.add_argument("-per", help="impose periodic BC", action="store_true")
    parser.add_argument("-dt", help="time step size", type=float, default=0.01)
    parser.add_argument("-leja_a", help="leja scale", type=float, default=-1.0)
    parser.add_argument("-leja_c", help="leja scale", type=float, default=10.0)
    parser.add_argument("-krylov_reuse", help="recycle krylov in leja", action="store_true", default=False)
    parser.add_argument("-dd_method", help="divided difference method", type=str, default="taylor")
    parser.add_argument("-nonautonomous", help="run nonautonomous system with external forcing", action="store_true", default=False)
    args = parser.parse_args()

    periodic = args.per
    mesh = make_1d_mesh(args.mr, periodic=periodic)
    nelements = mesh.GetNE()
    print(f"Mesh: nelements={nelements}, nv={mesh.GetNV()}, nbe={mesh.GetNBE()}")

    if args.ic == "zero":
        vel = 0.1
        nu = 1.0
    else:
        vel = 0.5
        nu = args.nu
    param_dict = {"nu": nu, "vel": vel, "tau": args.tau}
    field_dict = {}

    sem = AdDiffSEMMfem(mesh, p=args.p, params=param_dict, field_fns=field_dict)
    if not args.nonautonomous:
        ode_sys = AffineLinearSEMMfem(sem)
    else:
        ode_sys = NonautonomousSEMMfem(sem)
    t = 0.0

    xs = np.asarray(sem.collocation_points())

    dist = lambda x, xp: jnp.abs(x - xp)
    if periodic or args.nonautonomous:
        dist = lambda x, xp: torus_distance(x, xp)

    if args.ic == "square":
        startx, endx = 0.1, 0.4
        meanx, dxhalf = (endx + startx) / 2., (endx - startx) / 2.
        g_prof = lambda x: np.where(dist(meanx, x) < dxhalf, 1., 0.)
        y0_profile = g_prof(xs)
        y0 = jnp.asarray(y0_profile)
        g_prof_exact = lambda t, x: g_prof(x - t * vel)
    elif args.ic == "zero":
        g_prof = lambda x: np.zeros(x.shape)
        y0_profile = g_prof(xs)
        y0 = jnp.asarray(y0_profile)
        g_prof_exact = lambda t, x: g_prof(x - t * vel)
    else:
        gauss_scale = 1.0
        wc, ww = 0.3, 0.05
        var = ww ** 2.0
        g_prof = lambda x: gauss_scale * np.exp(-(dist(x, wc) / (2 * ww)) ** 2.0)

        def g_prof_exact(t, x):
            out = np.zeros(x.shape)
            dwidth = 1.0
            shifts = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]) * dwidth
            ns = len(shifts)
            for s in shifts:
                out += np.exp(-(s + torus_distance(x - t * vel, wc)) ** 2.0 / (4 * var + 4 * nu * t))
            norm_const = np.sqrt(4 * var) / (np.sqrt((4 * var + 4 * nu * t)))
            out *= norm_const
            out *= gauss_scale
            return out

        y0_profile = g_prof_exact(0.0, xs)
        y0 = jnp.asarray(y0_profile)

    # Dirichlet BC initial data
    if sem.dirichlet_bd is not None and len(sem.dirichlet_bd) > 0:
        if args.ic == "zero":
            y_dir = 1
        else:
            y_dir = g_prof(np.array([0.]))
        y0 = y0.at[sem.dirichlet_bd].set(y_dir)

    t0 = 0.0
    nsteps = args.nsteps
    dt = args.dt
    tf = dt * nsteps
    method = args.method
    pfd_method = args.pfd_method
    res = integrate_wrapper.integrate(
        ode_sys, y0, t0, dt, nsteps, method,
        max_krylov_dim=300, iom=2, pfd_method=pfd_method,
        leja_c=args.leja_c, leja_a=args.leja_a, dd_method=args.dd_method,
        logging=True, phikv_method=args.phikv_method, krylov_reuse=args.krylov_reuse,
        spec_method=args.spec_method, spec_iter=30, tol=1e-13,
    )
    t_res, y_res = res.t_res, res.y_res

    y_exact_res = []
    for t in t_res:
        y_exact_res.append(g_prof_exact(t, xs))

    si = xs.argsort()
    sx = xs[si]
    mesh_spacing = float(1.0 / nelements)
    cfl = dt * vel / mesh_spacing
    plt.figure()
    for i in range(nsteps + 1):
        if i == nsteps or i == 0:
            t = t_res[i]
            y = y_res[i][si]
            y_exact = y_exact_res[i][si]
            plt.plot(sx, y, label='t=%0.4f' % t)
            plt.plot(sx, y_exact, ls='--', label='exact t=%0.4f' % t)
    plt.legend()
    plt.grid(ls='--')
    plt.ylabel('u')
    plt.xlabel('x')
    plt.title("Method=%s, $C$=%0.2e, $v_{adv.}$=%0.2e \n $tau$=%0.2e, $\\Delta$ t=%0.2e, $\\Delta$ x=%0.2e" % (method, cfl, vel, args.tau, dt, mesh_spacing))
    plt.savefig('adv_diff_1d_mfem_%s_%s_%s_%s.png' % (method, str(args.mr), str(args.ic), str(tf)))
    plt.close()

    print("CFL: %0.4f, Ndof: %d" % (cfl, xs.size))

    err = y_exact - y
    l2 = np.sqrt(np.sum(err ** 2 * ode_sys.Ml))
    l1 = np.linalg.norm(err * ode_sys.Ml, 1)
    linf = np.linalg.norm(err, np.inf)
    print("mesh_spacing: %0.4e, CFL=%0.4f, L1=%0.4e, L2=%0.4e, Linf=%0.4e" % (mesh_spacing, cfl, l1, l2, linf))