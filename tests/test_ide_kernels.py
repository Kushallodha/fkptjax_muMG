"""Kernel-level tests for the IDE branch of calculate_jax.

(a) g = 0 regression: the stock fkpt-approximation path fed our own g = 0
    coefficients must reproduce the IDE branch in every k-function.
(b) Independent Eulerian time basis (collaborator's guide Sec. 4.1, eqs. 9, 10,
    16-18) at g != 0: F2, G2 and the p -> -p symmetrised F3K, G3K; and the general
    (non-collapsed) F3 of the notes' Lagrangian map, including the Sigma_3 sector (S, WS).
(c) Momentum conservation, IR limits and the soft P22 + P13 cancellation.
"""

import itertools

import diffrax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path

from fkptjax.calculate_jax import JaxCalculator
from fkptjax.ide import (
    F2_G2, F3K_G3K, TanhDilutionLCDM, Z_INI, ide_kernel_coeffs,
)
from fkptjax.util import setup_kfunctions

BENCH = TanhDilutionLCDM(Om0=0.3, g0=0.2, zt=2.0, dz=0.2)
LCDM = TanhDilutionLCDM(Om0=0.3, g0=0.0, zt=2.0, dz=0.2)
DATA = Path(__file__).parent / "data" / "test_data.npz"


@pytest.fixture(scope="module")
def linear():
    d = np.load(DATA)
    return d["k_in"], d["P_wiggle"], d["P_nowiggle"]


def _calculator(k_in, kmin=0.001, kmax=0.5, Nk=40, nquadSteps=200):
    calc = JaxCalculator()
    calc.initialize(setup_kfunctions(k_in, kmin, kmax, Nk, nquadSteps, 10, 10))
    return calc


# ----------------------------------------------------------------------------
# (a) g = 0 regression against the stock fkpt-approximation path
# ----------------------------------------------------------------------------

def test_g0_regression_against_stock_path(linear):
    k_in, P, Pnw = linear
    c = ide_kernel_coeffs(0.5, LCDM)
    calc = _calculator(k_in)
    ones = np.ones_like(k_in)
    stock = calc.evaluate_jax(
        P, Pnw, c.f * ones, A=c.L, ApOverf0=c.Lp / c.f, CFD3=c.F,
        CFD3p=c.F + c.Fp / (3.0 * c.f), sigma2v=0.0, f0=c.f)
    ide = calc.evaluate_jax(
        P, Pnw, c.fQ * ones, A=1.0, ApOverf0=0.0, CFD3=1.0, CFD3p=1.0,
        sigma2v=0.0, f0=c.fQ, ide=c)
    for name, s, i in zip(stock._fields, stock, ide):
        scale = float(jnp.max(jnp.abs(s)))
        np.testing.assert_allclose(i, s, rtol=0, atol=1e-11 * scale, err_msg=name)


# ----------------------------------------------------------------------------
# (b) Eulerian time basis (guide Sec. 4.1)
# ----------------------------------------------------------------------------

# EdS third-order sources s_j (delta) and t_j (Theta), per unit a^3 (guide eq. 10).
_S = np.array([5 / 7, 2 / 7, 3 / 7, 4 / 7, 0.0, 0.0])
_T = np.array([0.0, 0.0, 0.0, 0.0, 6 / 7, 8 / 7])
_C3 = (3.5 * _S + _T) / 9.0
_D3 = 3.0 * _C3 - _S


def _time_basis(z_out, background, z_ini=Z_INI):
    """Integrate guide eq. 9 for every structure; return D, fQ and the basis
    coefficients (aal, bal, abe, bbe) and (x_j, y_j) at z_out."""

    def rhs(N, y, args):
        E, A_Phi, g = background(N)
        D, V, aal, bal, abe, bbe = y[:6]
        x, yy = y[6:12], y[12:18]
        sdel = jnp.stack([V * aal, V * abe, D * bal, D * bbe, 0.0 * D, 0.0 * D])
        sth = jnp.stack([0.0 * D, 0.0 * D, 0.0 * D, 0.0 * D, 2 * V * bal, 2 * V * bbe])
        pair = lambda x, y, sd, st: (-g * x + y + sd, -E * y + A_Phi * x + st)
        daal, dbal = pair(aal, bal, V * D, 0.0)
        dabe, dbbe = pair(abe, bbe, 0.0, V * V)
        dx, dy = pair(x, yy, sdel, sth)
        return jnp.concatenate([jnp.stack([V - g * D, -E * V + A_Phi * D,
                                           daal, dbal, dabe, dbbe]), dx, dy])

    N0, N1 = -np.log1p(z_ini), -np.log1p(z_out)
    a = np.exp(N0)
    y0 = jnp.concatenate([jnp.array([a, a, 5 / 7 * a**2, 3 / 7 * a**2,
                                     2 / 7 * a**2, 4 / 7 * a**2]),
                          jnp.asarray(_C3 * a**3), jnp.asarray(_D3 * a**3)])
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Tsit5(), t0=N0, t1=N1, dt0=0.01, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=1e-11, atol=1e-16),
        max_steps=100000)
    y = np.asarray(sol.ys[-1])
    D, V = y[0], y[1]
    return D, V / D, y[2:6], y[6:12], y[12:18]


def _alpha(q1, q2):
    return np.dot(q1 + q2, q1) / np.dot(q1, q1)


def _beta(q1, q2):
    q12 = q1 + q2
    return np.dot(q12, q12) * np.dot(q1, q2) / (2.0 * np.dot(q1, q1) * np.dot(q2, q2))


def _alpha_s(q1, q2):
    return 0.5 * (_alpha(q1, q2) + _alpha(q2, q1))


def _structures(ks, ka, kb):
    """M_1..M_6 for single momentum ks and pair (ka, kb) (guide eq. 16)."""
    kab = ka + kb
    if np.dot(kab, kab) < 1e-24 * (np.dot(ka, ka) + np.dot(kb, kb)):
        return np.zeros(6)
    a_s, b = _alpha_s(ka, kb), _beta(ka, kb)
    a1, a2, b1 = _alpha(ks, kab), _alpha(kab, ks), _beta(ks, kab)
    return np.array([a1 * a_s, a1 * b, a2 * a_s, a2 * b, b1 * a_s, b1 * b])


@pytest.fixture(scope="module", params=[0.0, 0.5, 1.5])
def basis_and_coeffs(request):
    z = request.param
    return _time_basis(z, BENCH), ide_kernel_coeffs(z, BENCH, solver='adaptive')


@pytest.mark.parametrize("k2, mu", [(0.3, 0.7), (1.0, -0.4), (2.5, 0.1), (0.8, -0.95)])
def test_F2_G2_against_time_basis(basis_and_coeffs, k2, mu):
    (D, fQ, (aal, bal, abe, bbe), _, _), c = basis_and_coeffs
    k1v = np.array([0.0, 0.0, 1.0])
    k2v = k2 * np.array([np.sqrt(1 - mu * mu), 0.0, mu])
    a_s, b = _alpha_s(k1v, k2v), _beta(k1v, k2v)
    F2_ref = (aal * a_s + abe * b) / D**2
    G2_ref = (bal * a_s + bbe * b) / (fQ * D**2)
    F2, G2 = F2_G2(mu, 1.0 / k2 + k2, c)
    np.testing.assert_allclose([F2, G2], [F2_ref, G2_ref], rtol=1e-8)


@pytest.mark.parametrize("r, x", [(0.2, 0.3), (0.5, -0.6), (1.0, 0.45), (1.7, 0.9), (3.0, -0.2)])
def test_F3K_G3K_against_time_basis(basis_and_coeffs, r, x):
    (D, fQ, _, xj, yj), c = basis_and_coeffs
    k = np.array([0.0, 0.0, 1.0])
    p = r * np.array([np.sqrt(1 - x * x), 0.0, x])
    M = _structures(p, k, -p) + _structures(-p, k, p)
    F3_ref = np.dot(xj, M) / (3.0 * D**3)
    G3_ref = np.dot(yj, M) / (3.0 * fQ * D**3)
    F3a, G3a = F3K_G3K(r, x, c)
    F3b, G3b = F3K_G3K(r, -x, c)
    np.testing.assert_allclose([0.5 * (F3a + F3b), 0.5 * (G3a + G3b)],
                               [F3_ref, G3_ref], rtol=1e-8)


def _gamma(a, b):
    return 1.0 - np.dot(a, b) ** 2 / (np.dot(a, a) * np.dot(b, b))


def _F3_lagrangian(k1, k2, k3, c, sigma3=True):
    """General symmetric F3(k1, k2, k3) from the notes' mass-weighted Lagrangian map
    delta(k) = int d^3q e^{-ik.q} [W(q) e^{-ik.Psi(q)} - 1], expanded to third order with

        -ik.Psi1(p)          = Gamma1 (k.p)/p^2,                      w1 = W1 = 1 - Gamma1,
        -ik.Psi2(pa, pb)     = (k.pab)/pab^2 3/14 L gamma_ab,         w2 = 3/14 WL gamma_ab,
        -ik.Psi3(p1, p2, p3) = 5/42 F G_3 - S/18 Sigma_3,             w3 = 5/42 WF G_3 - WS/18 Sigma_3,

    G_3 = Sym[gamma_ab gamma_{c,ab}] and Sigma_3 = 1 - mu12^2 - mu23^2 - mu31^2 + 2 mu12 mu23 mu31.
    The normalisations follow from the amplitudes of notes Sec. 3.2-3.3 (DL = 3/7 L D^2,
    DWL = 3/7 WL D^2, DF = 5/21 F D^3, DS = -S D^3/3, ...); W' + g W = g J reproduces the
    DWL, DWF and DWS equations. ``sigma3=False`` drops the Sigma_3 terms (control).
    """
    G1, L, WL, F, WF, S, WS = (float(getattr(c, n)) for n in ("Gamma1", "L", "WL", "F", "WF", "S", "WS"))
    W1 = 1.0 - G1
    ps = (k1, k2, k3)
    k = k1 + k2 + k3

    def X1(p):
        return G1 * np.dot(k, p) / np.dot(p, p)

    def collapsed(pa, pb):  # pa + pb = 0: gamma_ab = 0 removes the pair
        return np.dot(pa + pb, pa + pb) < 1e-24 * (np.dot(pa, pa) + np.dot(pb, pb))

    def X2(pa, pb):
        if collapsed(pa, pb):
            return 0.0
        pab = pa + pb
        return np.dot(k, pab) / np.dot(pab, pab) * 3.0 / 14.0 * L * _gamma(pa, pb)

    def w2(pa, pb):
        return 3.0 / 14.0 * WL * _gamma(pa, pb)

    singles = [(ps[i], ps[(i + 1) % 3], ps[(i + 2) % 3]) for i in range(3)]  # (pc; pa, pb)
    G3 = np.mean([0.0 if collapsed(pa, pb) else _gamma(pa, pb) * _gamma(pc, pa + pb)
                  for pc, pa, pb in singles])
    n = [p / np.linalg.norm(p) for p in ps]
    m12, m23, m31 = np.dot(n[0], n[1]), np.dot(n[1], n[2]), np.dot(n[2], n[0])
    Sig3 = 1.0 - m12**2 - m23**2 - m31**2 + 2.0 * m12 * m23 * m31
    s3 = 1.0 if sigma3 else 0.0
    X3 = 5.0 / 42.0 * F * G3 - s3 * S / 18.0 * Sig3
    w3 = 5.0 / 42.0 * WF * G3 - s3 * WS / 18.0 * Sig3
    cross = np.mean([W1 * X2(pa, pb) + w2(pa, pb) * X1(pc) + X1(pc) * X2(pa, pb)
                     + 0.5 * W1 * X1(pa) * X1(pb) for pc, pa, pb in singles])
    return w3 + X3 + cross + X1(k1) * X1(k2) * X1(k3) / 6.0


def _cos(a, b):
    return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))


def _F3_eq3122(k1, k2, k3, c):
    """Notes eq. 39 (eq. 3.122 of the original notes), implemented literally: k = k1 + k2 + k3, q = k2 + k3, averaged over
    the 6 permutations of (k1, k2, k3):

        F3 = { 1/6 (Gamma3 + W3) + 3/14 Gamma2(k2, k3) (k.q/q^2) [Gamma1 k.k1/k1^2 + W1]
               + 1/2 W2(k2, k3) Gamma1 k.k1/k1^2 + 1/6 Gamma1^3 K123
               + 1/2 W1 Gamma1^2 (k.k2)(k.k3)/(k2^2 k3^2) }_sym

    with Gamma2(k2, k3) = L (1 - mu23^2) (notes eq. 30), W2 = 3/7 WL (1 - mu23^2),
    Gamma3 = 5/7 F gamma23 gamma1q - S/3 Sigma3, W3 = 5/7 WF gamma23 gamma1q - WS/3 Sigma3,
    Sigma3 = 1 - 3 mu23^2 + 2 mu12 mu23 mu31, K123 = prod_a k.ka/ka^2.
    """
    G1, L, WL, F, WF, S, WS = (float(getattr(c, n)) for n in ("Gamma1", "L", "WL", "F", "WF", "S", "WS"))
    W1 = 1.0 - G1
    k = k1 + k2 + k3

    def X(p):
        return np.dot(k, p) / np.dot(p, p)

    tot = 0.0
    for a, b, d in itertools.permutations((k1, k2, k3)):
        q = b + d
        mu23, mu12, mu31, mu1q = _cos(b, d), _cos(a, b), _cos(d, a), _cos(a, q)
        g23, g1q = 1.0 - mu23**2, 1.0 - mu1q**2
        Sig3 = 1.0 - 3.0 * mu23**2 + 2.0 * mu12 * mu23 * mu31
        Gam3 = 5.0 / 7.0 * F * g23 * g1q - S / 3.0 * Sig3
        W3 = 5.0 / 7.0 * WF * g23 * g1q - WS / 3.0 * Sig3
        Gam2 = L * g23
        W2 = 3.0 / 7.0 * WL * g23
        tot += ((Gam3 + W3) / 6.0 + 3.0 / 14.0 * Gam2 * X(q) * (G1 * X(a) + W1)
                + 0.5 * W2 * G1 * X(a) + G1**3 * X(k1) * X(k2) * X(k3) / 6.0
                + 0.5 * W1 * G1**2 * X(b) * X(d))
    return tot / 6.0


# generic, non-coplanar triangles (Sigma_3 != 0), in units of the first leg
_TRIANGLES = [
    ((1.0, 0.0, 0.0), (0.3, 0.8, 0.1), (-0.2, 0.4, 0.9)),
    ((0.0, 0.0, 1.0), (0.9, -0.3, 0.5), (0.4, 1.2, -0.7)),
    ((0.5, 0.5, 0.0), (-0.1, 0.2, 0.6), (1.5, -0.4, 0.3)),
    ((1.0, 0.2, -0.3), (-0.6, 0.9, 0.2), (0.1, -0.5, -1.1)),
]


@pytest.mark.parametrize("tri", _TRIANGLES)
def test_general_F3_with_sigma3_sector_against_time_basis(basis_and_coeffs, tri):
    """Non-collapsed F3(k1, k2, k3) at g != 0, three independent ways: the mass-weighted
    Lagrangian map, the notes' eq. 39 and the Eulerian time basis, with all third-order
    amplitudes (F, WF and the Sigma_3 pair S, WS); agreement ~1e-12 (asserted at 1e-10).
    Dropping the Sigma_3 terms must spoil the agreement (the test sees S, WS)."""
    (D, fQ, _, xj, _), c = basis_and_coeffs
    k1, k2, k3 = (np.asarray(v, dtype=float) for v in tri)
    ref = sum(np.dot(xj, _structures(ks, ka, kb))
              for ks, ka, kb in ((k1, k2, k3), (k2, k3, k1), (k3, k1, k2))) / (3.0 * D**3)
    lpt = _F3_lagrangian(k1, k2, k3, c)
    np.testing.assert_allclose(lpt, ref, rtol=1e-10)
    np.testing.assert_allclose(_F3_eq3122(k1, k2, k3, c), ref, rtol=1e-10)
    np.testing.assert_allclose(_F3_eq3122(k1, k2, k3, c), lpt, rtol=1e-13)
    assert abs(_F3_lagrangian(k1, k2, k3, c, sigma3=False) / ref - 1.0) > 1e-3


def test_general_F3_reduces_to_collapsed_F3K(basis_and_coeffs):
    """F3(k, p, -p) of the general map equals the p -> -p symmetrised F3K (k = 1)."""
    _, c = basis_and_coeffs
    k = np.array([0.0, 0.0, 1.0])
    for r, x in ((0.4, 0.3), (1.3, -0.55)):
        p = r * np.array([np.sqrt(1 - x * x), 0.0, x])
        F3a, _ = F3K_G3K(r, x, c)
        F3b, _ = F3K_G3K(r, -x, c)
        np.testing.assert_allclose(_F3_lagrangian(k, p, -p, c), 0.5 * (F3a + F3b), rtol=1e-10)


# ----------------------------------------------------------------------------
# (c) Momentum conservation, IR limits, soft cancellation
# ----------------------------------------------------------------------------

def test_second_order_vanish_at_k_minus_k():
    c = ide_kernel_coeffs(0.5, BENCH)
    np.testing.assert_allclose(F2_G2(-1.0, 2.0, c), (0.0, 0.0), atol=1e-12)


@pytest.mark.parametrize("x", [0.2, 0.6, 0.95])
def test_F3K_G3K_IR_limit(x):
    """r -> 0: symmetrised F3K, G3K -> -Gamma1^2 x^2 / (6 r^2)."""
    c = ide_kernel_coeffs(0.5, BENCH)
    target = -float(c.Gamma1)**2 * x * x / 6.0
    for r in (1e-3, 1e-4):
        F3a, G3a = F3K_G3K(r, x, c)
        F3b, G3b = F3K_G3K(r, -x, c)
        np.testing.assert_allclose(r * r * 0.5 * (F3a + F3b), target, rtol=10 * r)
        np.testing.assert_allclose(r * r * 0.5 * (G3a + G3b), target, rtol=10 * r)


def _soft_ratio(calc, P, Pb, Pnw, c, c13):
    """|delta(P22 + P13)| / |delta P13| for a low-k bump P -> Pb, with P22 from
    coefficients c and P13 from c13."""
    ones = np.ones_like(P)

    def run(Pk, cc):
        return calc.evaluate_jax(Pk, Pnw, cc.fQ * ones, A=1.0, ApOverf0=0.0, CFD3=1.0,
                                 CFD3p=1.0, sigma2v=0.0, f0=cc.fQ, ide=cc)

    out = {}
    for field in ("P22dd", "P22uu", "P13dd", "P13uu"):
        cc = c if field.startswith("P22") else c13
        out[field] = np.asarray(getattr(run(Pb, cc), field)[0] - getattr(run(P, cc), field)[0])
    return [np.max(np.abs(out["P22" + s] + out["P13" + s]) / np.abs(out["P13" + s]))
            for s in ("dd", "uu")]


def test_soft_P22_P13_cancellation(linear):
    """Adding power at q0 << k shifts P22 and P13 by ~ -+Gamma1^2 k^2 sigma_bump^2 P(k);
    the shifts cancel with consistent kernels (notes Sec. 6). Control: P13 with Gamma1 -> 1
    (the EdS-kernel mismatch) leaves ~(Gamma1^2 - 1) uncancelled."""
    k_in, P, Pnw = linear
    q0, width = 0.003, 0.15
    Pb = P * (1.0 + 0.5 * np.exp(-0.5 * (np.log(k_in / q0) / width) ** 2))
    calc = _calculator(k_in, kmin=0.08, kmax=0.2, Nk=8, nquadSteps=800)
    c = ide_kernel_coeffs(0.5, BENCH)

    ratios = _soft_ratio(calc, P, Pb, Pnw, c, c)
    assert max(ratios) < 0.02, ratios

    control = _soft_ratio(calc, P, Pb, Pnw, c, c._replace(Gamma1=1.0))
    assert min(control) > 0.1, control
