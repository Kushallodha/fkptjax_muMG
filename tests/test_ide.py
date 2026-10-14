"""Scalar tests for fkptjax.ide (IDE kernel coefficients, notes notation).

Reference values: collaborator's fkpt-DMdilution guide, Sec. 11
(g0 = 0.2, zt = 2, dz = 0.2, flat LCDM Om0 = 0.3, no radiation, z_ini = 50),
quoted to four decimals in the DM-dilution notation; compared through
:func:`fkptjax.ide.to_DMdilution_notation`.
"""

import jax
import jax.numpy as jnp
import diffrax
import numpy as np
import pytest

from fkptjax.ide import (
    F2_G2, IDEKernelCoeffs, TabulatedBackground, TanhDilutionLCDM, eds_initial_state,
    growing_mode_initial_state, ide_kernel_coeffs, ir_sigma2_tot, to_DMdilution_notation, Z_INI,
)

BENCH = TanhDilutionLCDM(Om0=0.3, g0=0.2, zt=2.0, dz=0.2)
LCDM = TanhDilutionLCDM(Om0=0.3, g0=0.0, zt=2.0, dz=0.2)

# z: (D_DM/D_LCDM, f, fv, R, A, B, C, u0, u2, Q_eq DM, Q_eq LCDM)
REFERENCE = {
    0.5: (0.8944, 0.6057, 0.8057, 1.1099, 1.3300, 0.8172, 0.2564,
          0.4731, 0.6368, 0.6225, 0.5726),
    0.0: (0.8435, 0.3652, 0.5652, 1.1642, 1.4822, 0.7158, 0.3832,
          0.4987, 0.6655, 0.6443, 0.5742),
}
REF_ATOL = 1e-4  # four-decimal rounding plus margin


def _g_at(background, z):
    return background(-jnp.log1p(z))[2]


def q_eq(c):
    """Equilateral reduced bispectrum Q_eq = 2 F2(k, k, mu = -1/2) (guide eq. 28), from
    the notes' F2; equals the guide's (7 + 9B + 12C)/28."""
    return 2.0 * F2_G2(-0.5, 2.0, c)[0]


@pytest.mark.parametrize("z", sorted(REFERENCE))
def test_collaborator_reference_values(z):
    c = ide_kernel_coeffs(z, BENCH)
    l = ide_kernel_coeffs(z, LCDM)
    t = to_DMdilution_notation(c)
    got = (c.D / l.D, t["f"], t["fv"], t["R"], t["A"], t["B"], t["C"], t["u0"], t["u2"],
           q_eq(c), q_eq(l))
    np.testing.assert_allclose(got, REFERENCE[z], rtol=0, atol=REF_ATOL)
    np.testing.assert_allclose(q_eq(c), (7.0 + 9.0 * t["B"] + 12.0 * t["C"]) / 28.0, rtol=1e-12)
    assert float(c.Gamma1) == pytest.approx(REFERENCE[z][3], abs=REF_ATOL)  # uadv = R = Gamma1


@pytest.mark.parametrize("z", [0.0, 0.5, 1.0, 2.0, 3.0])
def test_continuity_identities(z):
    """Kinematic / continuity checks, not tests of the dynamics.

    fQ = f + g, fPsi = fQ/Gamma1 and F2(k, -k) = G2(k, -k) = 0 follow from one
    line of the linear system or from the kernel algebra. The dynamics is
    constrained by the reference table, the Eulerian A,B,C system, the
    time-basis tests (test_ide_kernels.py) and the g = 0 / EdS limits.
    """
    c = ide_kernel_coeffs(z, BENCH)
    g = _g_at(BENCH, z)
    np.testing.assert_allclose(c.fQ, c.f + g, rtol=1e-10)
    np.testing.assert_allclose(c.fPsi * c.Gamma1, c.fQ, rtol=1e-12)
    # F2(k, -k) = G2(k, -k) = 0 (momentum conservation): mu = -1, k1/k2 + k2/k1 = 2.
    np.testing.assert_allclose(F2_G2(-1.0, 2.0, c), (0.0, 0.0), atol=1e-12)


def _eulerian_ABC(z_out, background, z_ini=Z_INI):
    """Independent Eulerian second-order system (guide eq. 6)."""
    g_of_N = lambda N: background(N)[2]
    dg_dN = jax.grad(g_of_N)

    def rhs(N, y, args):
        E, A_Phi, g = background(N)
        Xi = -(E * g + dg_dN(N)) / A_Phi
        D, V, XA, XAp, XB, XBp, XC, XCp = y
        f = (V - g * D) / D
        D2 = D * D
        S = (D2 * (A_Phi * (1.0 - Xi) + 2.0 * f * g),
             D2 * (A_Phi * (1.0 + Xi) - 2.0 * g * (2.0 * f + g)),
             D2 * (3.0 * f * g + g * g - A_Phi * Xi))
        out = [V - g * D, -E * V + A_Phi * D]
        for X, Xp, s in ((XA, XAp, S[0]), (XB, XBp, S[1]), (XC, XCp, S[2])):
            out += [Xp, -(E + g) * Xp + A_Phi * (1.0 + Xi) * X + s]
        return jnp.stack(out)

    N0, N1 = -jnp.log1p(z_ini), -jnp.log1p(z_out)
    a = jnp.exp(N0)
    x0 = 3.0 / 7.0 * a * a
    y0 = jnp.array([a, a, x0, 2 * x0, x0, 2 * x0, 0.0, 0.0])
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Tsit5(), t0=N0, t1=N1, dt0=0.01, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-14),
        max_steps=100000)
    D, _, XA, _, XB, _, XC, _ = sol.ys[-1]
    return tuple(7.0 * X / (3.0 * D * D) for X in (XA, XB, XC))


@pytest.mark.parametrize("z", [0.0, 0.5, 2.0])
def test_lagrangian_matches_eulerian_ABC(z):
    t = to_DMdilution_notation(ide_kernel_coeffs(z, BENCH, solver='adaptive'))
    np.testing.assert_allclose((t["A"], t["B"], t["C"]), _eulerian_ABC(z, BENCH), rtol=1e-7, atol=1e-9)


def test_eds_limit():
    eds = TanhDilutionLCDM(Om0=1.0, g0=0.0, zt=2.0, dz=0.2)
    c = ide_kernel_coeffs(0.5, eds, solver='adaptive')
    expected = dict(f=1.0, fQ=1.0, Gamma1=1.0, fPsi=1.0, L=1.0, Lp=0.0, WL=0.0,
                    F=1.0, Fp=0.0, WF=0.0, S=1.0, WS=0.0)
    for name, val in expected.items():
        np.testing.assert_allclose(getattr(c, name), val, atol=1e-9, err_msg=name)
    t = to_DMdilution_notation(c)
    expected = dict(A=1.0, B=1.0, C=0.0, Ap=0.0, Bp=0.0, Cp=0.0, u0=3.0 / 7.0, u2=4.0 / 7.0)
    for name, val in expected.items():
        np.testing.assert_allclose(t[name], val, atol=1e-9, err_msg=name)
    # EdS F2 and G2 (notes Sec. 4.2): 5/7 + 2/7 mu^2 and 3/7 + 4/7 mu^2, plus the shift term
    mu, ratio = 0.3, 2.5
    F2, G2 = F2_G2(mu, ratio, c)
    np.testing.assert_allclose(F2, 5 / 7 + 2 / 7 * mu**2 + mu / 2 * ratio, atol=1e-9)
    np.testing.assert_allclose(G2, 3 / 7 + 4 / 7 * mu**2 + mu / 2 * ratio, atol=1e-9)


@pytest.mark.parametrize("z", [0.0, 0.5])
def test_g0_limit_matches_stock_kernel_constants(z):
    """g = 0: IDE scalars reduce to the stock fkpt (LCDM) kernel constants.

    Stock A = L and A'/f = L'/f; CFD3 = F and CFD3p = F + F'/(3f). Both solves
    start at z_ini = 1000: the stock third-order initial data are not on the
    growing mode and leave a ~1e-3 transient in CFD3 at the stock default start
    (xnow = -3.912), which decays as (a_ini/a)^2.
    """
    from fkptjax.jax_ode import kernel_constants_jax
    from fkptjax import mg_jax as bj

    z_ini = 1000.0
    c = ide_kernel_coeffs(z, LCDM, z_ini=z_ini, solver='adaptive')
    np.testing.assert_allclose((c.Gamma1 - 1.0, c.fQ - c.f, c.WL, c.WF, c.WS),
                               0.0, atol=1e-12)
    P = bj.pack_constants_jnp(om=0.3, ol=0.7, kind=bj.LCDM)
    KA, KAp, KR1, KR1p = kernel_constants_jax(
        c.f, P, -jnp.log1p(z_ini), -jnp.log1p(z), solver='adaptive',
        rtol=1e-10, atol=1e-20)
    np.testing.assert_allclose(KA, c.L, rtol=1e-9)
    np.testing.assert_allclose(KAp / c.f, c.Lp / c.f, rtol=1e-7)
    np.testing.assert_allclose(KR1, c.F, rtol=1e-5)
    np.testing.assert_allclose(KR1p, c.F + c.Fp / (3.0 * c.f), rtol=1e-5)


def test_start_redshift_converged():
    c50 = ide_kernel_coeffs(0.5, BENCH, z_ini=50.0)
    c200 = ide_kernel_coeffs(0.5, BENCH, z_ini=200.0)
    for name in IDEKernelCoeffs._fields:
        if name in ('z_out', 'D'):
            continue
        np.testing.assert_allclose(getattr(c50, name), getattr(c200, name),
                                   rtol=1e-5, atol=1e-7, err_msg=name)


def test_tabulated_background_must_cover_range():
    N = jnp.linspace(-jnp.log1p(20.0), 0.0, 500)  # starts after z_ini = 50
    E, A_Phi, g = BENCH(N)
    tab = TabulatedBackground(N=N, E=E, A_Phi=A_Phi, g=g)
    with pytest.raises(Exception, match="does not cover"):
        jax.block_until_ready(ide_kernel_coeffs(0.5, tab))  # errors surface on sync


def test_tabulated_background_must_be_uniform():
    N = jnp.linspace(-jnp.log1p(Z_INI) - 0.02, 0.0, 500)
    N = N.at[100].add(1e-3)
    E, A_Phi, g = BENCH(N)
    with pytest.raises(Exception, match="uniform"):
        jax.block_until_ready(ide_kernel_coeffs(0.5, TabulatedBackground(N=N, E=E, A_Phi=A_Phi, g=g)))


ZS = jnp.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])


def test_rk4_matches_adaptive():
    """Default fixed-step RK4 vs the adaptive reference: < 1e-6 absolute in every
    coefficient (D: relative) for the benchmark and a smooth background."""
    smooth = TanhDilutionLCDM(Om0=0.3, g0=0.2, zt=2.0, dz=1.0)
    for bg in (BENCH, smooth):
        ref = ide_kernel_coeffs(ZS, bg, solver='adaptive', rtol=1e-12, atol=1e-16)
        c = ide_kernel_coeffs(ZS, bg)
        np.testing.assert_allclose(c.D / ref.D, 1.0, atol=1e-6)
        for name in IDEKernelCoeffs._fields:
            if name in ('z_out', 'D'):
                continue
            np.testing.assert_allclose(getattr(c, name), getattr(ref, name), rtol=0, atol=1e-6,
                                       err_msg=name)


def test_vector_z_out_matches_scalar_calls():
    vec = ide_kernel_coeffs(ZS, BENCH)
    assert vec.fQ.shape == ZS.shape and vec.z_out.shape == ZS.shape
    for i, z in enumerate(ZS):
        one = ide_kernel_coeffs(z, BENCH)
        for name in IDEKernelCoeffs._fields:
            if name == 'D':
                np.testing.assert_allclose(vec.D[i], one.D, rtol=1e-6)
            else:
                np.testing.assert_allclose(getattr(vec, name)[i], getattr(one, name),
                                           rtol=0, atol=1e-6, err_msg=name)
    adaptive = ide_kernel_coeffs(ZS[::-1], BENCH, solver='adaptive')  # unsorted z order
    np.testing.assert_allclose(adaptive.fQ[::-1], vec.fQ, atol=1e-6)


def test_tabulated_background_matches_analytic():
    N = jnp.linspace(-jnp.log1p(Z_INI), 0.0, 4000)
    E, A_Phi, g = BENCH(N)
    tab = TabulatedBackground(N=N, E=E, A_Phi=A_Phi, g=g)
    c_tab = ide_kernel_coeffs(0.5, tab)
    c_ana = ide_kernel_coeffs(0.5, BENCH)
    for name in IDEKernelCoeffs._fields:
        np.testing.assert_allclose(getattr(c_tab, name), getattr(c_ana, name),
                                   rtol=1e-5, atol=1e-7, err_msg=name)


def test_pytree_jit_and_gradients():
    c = ide_kernel_coeffs(0.5, BENCH)
    leaves = jax.tree_util.tree_leaves(c)
    assert len(leaves) == len(IDEKernelCoeffs._fields)
    assert all(jnp.ndim(x) == 0 for x in leaves)
    assert float(c.z_out) == 0.5

    def fQ_of(g0, z):
        return ide_kernel_coeffs(z, BENCH._replace(g0=g0)).fQ

    dg0, dz = jax.grad(fQ_of, argnums=(0, 1))(0.2, 0.5)
    h = 1e-5
    fd_g0 = (fQ_of(0.2 + h, 0.5) - fQ_of(0.2 - h, 0.5)) / (2 * h)
    fd_z = (fQ_of(0.2, 0.5 + h) - fQ_of(0.2, 0.5 - h)) / (2 * h)
    np.testing.assert_allclose(dg0, fd_g0, rtol=1e-5)
    np.testing.assert_allclose(dz, fd_z, rtol=1e-5)


def test_ir_sigma2_tot():
    c = ide_kernel_coeffs(0.5, BENCH)
    s2, ds2 = 20.0, 5.0
    # mu = 0: Gamma1^2 Sigma^2.  mu = 1: (Gamma1 + fQ)^2 Sigma^2 (notes Sec. 6).
    np.testing.assert_allclose(ir_sigma2_tot(0.0, s2, ds2, c), c.Gamma1 ** 2 * s2, rtol=1e-12)
    np.testing.assert_allclose(ir_sigma2_tot(1.0, s2, ds2, c), (c.Gamma1 + c.fQ) ** 2 * s2, rtol=1e-12)
    # Gamma1 = 1, fPsi = f recovers FOLPS's formula.
    lc = c._replace(Gamma1=1.0, fPsi=c.f)
    mu = 0.6
    folps = (1 + c.f * mu**2 * (2 + c.f)) * s2 + (c.f * mu)**2 * (mu**2 - 1) * ds2
    np.testing.assert_allclose(ir_sigma2_tot(mu, s2, ds2, lc), folps, rtol=1e-12)


def test_growing_mode_initial_state_reduces_to_eds():
    eds = TanhDilutionLCDM(Om0=1.0, g0=0.0, zt=2.0, dz=0.2)
    N_ini = -np.log1p(Z_INI)
    np.testing.assert_allclose(growing_mode_initial_state(eds, N_ini),
                               eds_initial_state(np.exp(N_ini)), rtol=1e-12)


def test_growing_mode_ic_solvers_and_gradients():
    """ic='growing': RK4 vs adaptive, and jit/grad, as for the EdS initial data."""
    ref = ide_kernel_coeffs(ZS, BENCH, solver='adaptive', rtol=1e-12, atol=1e-16, ic='growing')
    c = ide_kernel_coeffs(ZS, BENCH, ic='growing')
    for name in IDEKernelCoeffs._fields:
        if name not in ('z_out', 'D'):
            np.testing.assert_allclose(getattr(c, name), getattr(ref, name), rtol=0, atol=1e-6,
                                       err_msg=name)
    fQ_of = lambda g0: ide_kernel_coeffs(0.5, BENCH._replace(g0=g0), ic='growing').fQ
    h = 1e-5
    np.testing.assert_allclose(jax.grad(fQ_of)(0.2), (fQ_of(0.2 + h) - fQ_of(0.2 - h)) / (2 * h),
                               rtol=1e-5)
