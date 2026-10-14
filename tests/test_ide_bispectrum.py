"""Tree-level IDE bispectrum (``fkptjax.ide_bispectrum``) tests."""

import os

os.environ.setdefault("FOLPS_BACKEND", "jax")  # FOLPS under jit

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fkptjax.ide import TanhDilutionLCDM, ide_kernel_coeffs, to_DMdilution_notation
from fkptjax.ide_bispectrum import get_calculator_cls, ide_bispectrum_poles

ff = pytest.importorskip("folps.folps")

DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "test_data.npz"
Z = 0.5
BENCH = TanhDilutionLCDM(Om0=0.3, g0=0.2, zt=2.0, dz=0.2)
LCDM = TanhDilutionLCDM(Om0=0.3, g0=0.0, zt=2.0, dz=0.2)
BPARS = jnp.array([1.8, 0.4, -0.3, 1.0, -2.0, 0.3, 0.2, 0.0])  # b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG
KK = np.linspace(0.02, 0.15, 8)
SUGIYAMA = ["B000", "B202", "B110", "B112"]


@pytest.fixture(scope="module")
def linear():
    d = np.load(DATA)
    return d["k_in"], d["P_wiggle"], d["P_nowiggle"]


@pytest.mark.parametrize("basis", ["sugiyama", "scoccimarro"])
def test_lcdm_limit_matches_folps(linear, basis):
    """g = 0 (Gamma1 = 1, WL = 0, fPsi = fQ = f): FOLPS's fk calculator with calA = L, calAp = L', f(k) = f."""
    k, pw, pnw = linear
    c = ide_kernel_coeffs(Z, LCDM)
    assert abs(float(c.Gamma1) - 1.0) < 1e-12 and abs(float(c.WL)) < 1e-12
    table = [k, pw, pnw] + [np.ones_like(k) * v for v in (c.f, c.L, c.Lp)]
    calc = ff.BispectrumCalculator_fk(model="EFT")
    if basis == "sugiyama":
        kp, mp = np.stack([KK, 1.3 * KK], 1), SUGIYAMA
        ref = calc.Sugiyama_Bell(c.f, BPARS, table, k1k2pairs=kp, qpar=1.02, qper=0.98, multipoles=mp,
                                 interpolation_method="linear")
    else:
        kp, mp = np.stack([KK, 1.2 * KK, 1.5 * KK], 1), ["B0", "B2", "B4"]
        ref = calc.Scoccimarro_Bell(kp, c.f, BPARS, 1.02, 0.98, table, multipoles=mp)[:3]
    B = ide_bispectrum_poles(c, BPARS, kp, (k, pw, pnw), qpar=1.02, qper=0.98, multipoles=mp, basis=basis)
    np.testing.assert_allclose(B, jnp.asarray(ref), rtol=1e-12)


def _reference_integrand(c, b, k1, k2, x12, mu1, phi, k, pw, pnw, Sigma2, dSigma2):
    """Independent numpy transcription of the IDE tree-level B(k1, k2, x12, mu1, phi) (no AP, no FoG),
    written in the DM-dilution notation (A, B, R, u0, u2, fv) via to_DMdilution_notation."""
    b1, b2, bs, c1, c2, Bshot, Pshot, _ = (float(v) for v in b)
    t = to_DMdilution_notation(c)
    A, B_, R, u0, u2, fv, fPsi = (float(t[n]) for n in ("A", "B", "R", "u0", "u2", "fv", "fPsi"))
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * x12)
    mu2 = np.sqrt(1 - mu1**2) * np.sqrt(1 - x12**2) * np.cos(phi) + mu1 * x12
    mu3 = -(k1 * mu1 + k2 * mu2) / k3
    legs = [(k1, mu1), (k2, mu2), (k3, mu3)]

    def pIR(q, m):
        s2 = R**2 * ((1 + fPsi * m**2 * (2 + fPsi)) * Sigma2 + fPsi**2 * m**2 * (m**2 - 1) * dSigma2)
        p, pn = np.interp(q, k, pw), np.interp(q, k, pnw)
        return pn + (p - pn) * np.exp(-s2 * q**2)

    def Z1(q, m):
        return b1 + fv * m**2 - (c1 * m**2 + c2 * m**4) * q**2

    def Z2(qi, mi, qj, mj):
        # cosine between the legs from the closed triangle; F2 / G2v in the DM-dilution form
        qs = [l[0] for l in legs]
        qk = [q for q in qs if q is not qi and q is not qj][0]
        x = (qk**2 - qi**2 - qj**2) / (2 * qi * qj)
        adv = R * x / 2 * (qi / qj + qj / qi)
        F2 = 0.5 + 3 / 14 * A + (0.5 - 3 / 14 * B_) * x**2 + adv
        G2 = u0 + u2 * x**2 + adv
        km = qi * mi + qj * mj
        musq = km**2 / qk**2
        return (b2 / 2 + bs / 2 * (x**2 - 1 / 3) + b1 * F2 + fv * musq * G2
                + km / 2 * fv * (mi / qi * (b1 + fv * mj**2) + mj / qj * (b1 + fv * mi**2)))

    P = [pIR(q, m) for q, m in legs]
    Zl = [Z1(q, m) for q, m in legs]
    tot = 0.0
    for i, j in ((0, 1), (1, 2), (2, 0)):
        tot = tot + 2 * Z2(legs[i][0], legs[i][1], legs[j][0], legs[j][1]) * Zl[i] * P[i] * Zl[j] * P[j]
    shot = sum((b1 * Bshot + Pshot * fv * m**2) * Z * p for (q, m), Z, p in zip(legs, Zl, P)) + Pshot**2
    return tot + shot


def test_integrand_matches_independent_reference(linear):
    """g != 0: B(k1, k2, x12, mu1, phi) vs a numpy transcription of the IDE Z1/Z2 and BAO damping."""
    k, pw, pnw = linear
    c = ide_kernel_coeffs(Z, BENCH)
    assert float(c.Gamma1) > 1.05 and float(c.fQ / c.f) > 1.2  # far from LCDM
    calc = get_calculator_cls()(c)
    sigma2v, Sigma2, dSigma2 = calc.sigmas(jnp.asarray(k), jnp.asarray(pw))
    rng = np.random.default_rng(1)
    n = 50
    k1, k2 = rng.uniform(0.02, 0.2, n), rng.uniform(0.02, 0.2, n)
    x12, mu1, phi = rng.uniform(-0.95, 0.95, n), rng.uniform(-0.95, 0.95, n), rng.uniform(0, np.pi, n)
    got = calc.bispectrum(k1, k2, x12, mu1, phi, c.fQ, sigma2v, Sigma2, dSigma2, BPARS, 1.0, 1.0,
                          [jnp.asarray(k), jnp.asarray(pw), jnp.asarray(pnw)], interpolation_method="linear")
    ref = _reference_integrand(c, BPARS, k1, k2, x12, mu1, phi, k, pw, pnw, float(Sigma2), float(dSigma2))
    np.testing.assert_allclose(got, ref, rtol=1e-10)


def test_jit_and_gradient(linear):
    k, pw, pnw = linear
    kp = np.stack([KK, KK], 1)

    def b000(g0):
        c = ide_kernel_coeffs(Z, TanhDilutionLCDM(0.3, g0, 2.0, 0.2))
        return ide_bispectrum_poles(c, BPARS, kp, (k, pw, pnw))[0, 4]

    fn = jax.jit(b000)
    np.testing.assert_allclose(fn(0.2), b000(0.2), rtol=1e-12)
    h = 1e-5
    np.testing.assert_allclose(jax.grad(fn)(0.2), (fn(0.2 + h) - fn(0.2 - h)) / (2 * h), rtol=1e-6)


def test_redshift_smearing_hook(linear):
    k, pw, pnw = linear
    c = ide_kernel_coeffs(Z, BENCH)
    kp = np.stack([KK, KK], 1)
    base = ide_bispectrum_poles(c, BPARS, kp, (k, pw, pnw))
    np.testing.assert_allclose(ide_bispectrum_poles(c, BPARS, kp, (k, pw, pnw),
                                                    redshift_smearing=lambda kmu: jnp.ones_like(kmu)), base, rtol=1e-14)
    smeared = ide_bispectrum_poles(c, BPARS, kp, (k, pw, pnw), redshift_smearing=lambda kmu: jnp.exp(-(3.0 * kmu)**2))
    assert np.all(np.abs(smeared[0]) < np.abs(base[0]))


def test_rejects_bad_multipoles(linear):
    k, pw, pnw = linear
    c = ide_kernel_coeffs(Z, BENCH)
    with pytest.raises(ValueError, match="Sugiyama"):
        ide_bispectrum_poles(c, BPARS, np.stack([KK, KK], 1), (k, pw, pnw), multipoles=["B0"])
    with pytest.raises(ValueError, match="basis"):
        ide_bispectrum_poles(c, BPARS, np.stack([KK, KK], 1), (k, pw, pnw), basis="foo")
