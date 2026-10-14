"""Builder (``ide=``) and RSD-wrapper tests for the IDE path."""

import os

os.environ.setdefault("FOLPS_BACKEND", "jax")  # FOLPS under jit

import importlib.util
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import roots_legendre

from fkptjax.ide import TanhDilutionLCDM, ide_kernel_coeffs, ir_sigma2_tot
from fkptjax.kfuncs_to_tables import (
    Kfuncs_to_tables, Kfuncs_to_tables_jax, Kfuncs_to_tables_rescale_jax,
)
from fkptjax.pipelines import binning_jax_poles, full_method_poles, rescale_branch_poles
from fkptjax.rsd import pack_fkpt_bias, tables_to_pkmu, tables_to_poles

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tests" / "data" / "test_data.npz"
GRID = dict(kmin=1e-3, kmax=0.3, Nk_kernel=40, nquadSteps=150)
Z = 0.5
BENCH = TanhDilutionLCDM(Om0=0.3, g0=0.2, zt=2.0, dz=0.2)
LCDM = TanhDilutionLCDM(Om0=0.3, g0=0.0, zt=2.0, dz=0.2)
PARS = dict(b1=1.8, b2=0.4, bs2=-0.3, b3nl=0.2, alpha0=5.0, alpha2=-3.0, alpha4=1.0,
            ctilde=2.0, alpha0shot=0.5, alpha2shot=-1.0, PshotP=3000.0)


@pytest.fixture(scope="module")
def linear():
    d = np.load(DATA)
    return d["k_in"], d["P_wiggle"], d["P_nowiggle"]


@pytest.fixture(scope="module")
def bench_tables(linear):
    k, P, Pnw = linear
    c = ide_kernel_coeffs(Z, BENCH)
    tw, tn, kc = Kfuncs_to_tables_jax(k, P, Pnw, z=Z, Om=0.3, ide=c, **GRID)
    return c, tw, tn, kc


def _mu_grid(table):
    mu, w = roots_legendre(8)
    mu, wmu = jnp.asarray(mu), 0.5 * jnp.asarray(w)  # project_to_poles wants sum(wmu) = 1
    kt = jnp.asarray(table[0])
    kq = kt[(kt > 0.01) & (kt < 0.25)]
    kap = jnp.broadcast_to(kq[:, None], (kq.size, mu.size))
    return kap, jnp.broadcast_to(mu[None, :], kap.shape), mu, wmu, (kt > 0.01) & (kt < 0.25)


def _load_from_main(relpath, name, tmp_path, package=None):
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"main:{relpath}"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git or the main branch is not available")
    path = tmp_path / f"{name}.py"
    path.write_text(src)
    # package: load as a submodule so main's relative imports resolve to it
    spec = importlib.util.spec_from_file_location(f"{package}.{name}" if package else name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_tables_identical(a, b):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert np.array_equal(np.asarray(x), np.asarray(y)), f"column {i} differs"


# ----------------------------------------------------------------------------
# ide=None is bit-identical to main
# ----------------------------------------------------------------------------

def test_ide_none_builders_and_rsd_bit_identical_to_main(linear, tmp_path):
    k, P, Pnw = linear
    main_k2t = _load_from_main("src/fkptjax/kfuncs_to_tables.py", "k2t_main", tmp_path)
    main_rsd = _load_from_main("src/fkptjax/rsd.py", "rsd_main", tmp_path)
    for beyond_eds in (False, True):
        kw = dict(z=Z, Om=0.3, beyond_eds=beyond_eds, model="LCDM", **GRID)
        new = Kfuncs_to_tables_jax(k, P, Pnw, **kw)
        old = main_k2t.Kfuncs_to_tables_jax(k, P, Pnw, **kw)
        _assert_tables_identical(new[0] + new[1], old[0] + old[1])
        new = Kfuncs_to_tables(k, P, Pnw, **kw)
        old = main_k2t.Kfuncs_to_tables(k, P, Pnw, **kw)
        _assert_tables_identical(new[0] + new[1], old[0] + old[1])
    tw, tn = new[0], new[1]
    kap, muap, mu, wmu, _ = _mu_grid(tw)
    pars = pack_fkpt_bias(PARS)
    a = tables_to_poles(1.0, kap, muap, pars, tw + tn, mu, wmu, ncols=len(tw))
    b = main_rsd.tables_to_poles(1.0, kap, muap, pars, tw + tn, mu, wmu, ncols=len(tw))
    assert np.array_equal(np.asarray(a), np.asarray(b))


# ----------------------------------------------------------------------------
# Argument checks
# ----------------------------------------------------------------------------

def test_ide_argument_checks(linear):
    k, P, Pnw = linear
    c = ide_kernel_coeffs(Z, BENCH)
    for builder in (Kfuncs_to_tables, Kfuncs_to_tables_jax):
        with pytest.raises(ValueError, match="redshift"):
            builder(k, P, Pnw, z=1.0, Om=0.3, ide=c, **GRID)
        with pytest.raises(ValueError, match="f0 must equal"):
            builder(k, P, Pnw, z=Z, Om=0.3, ide=c, f0=0.7, **GRID)
        with pytest.raises(ValueError, match="fk"):
            builder(k, P, Pnw, z=Z, Om=0.3, ide=c, fk=np.ones_like(k), **GRID)
        with pytest.raises(ValueError, match="only used together"):
            builder(k, P, Pnw, z=Z, Om=0.3, f0_linear=0.7, **GRID)
    with pytest.raises(ValueError, match="rescale_PS"):
        Kfuncs_to_tables(k, P, Pnw, z=Z, Om=0.3, ide=c, rescale_PS=True, **GRID)
    with pytest.raises(ValueError, match="does not support ide"):
        Kfuncs_to_tables_rescale_jax(k, P, Pnw, z=Z, Om=0.3, ide=c, **GRID)


def test_f0_column_and_f0_linear(linear, bench_tables):
    k, P, Pnw = linear
    c, tw, tn, kc = bench_tables
    assert kc is c
    assert float(tw[-1]) == float(c.fQ) and float(tn[-1]) == float(c.fQ)
    np.testing.assert_array_equal(np.asarray(tw[2]), 1.0)  # f(k)/f0
    f0_lin = 0.81
    tw2, tn2, _ = Kfuncs_to_tables_jax(k, P, Pnw, z=Z, Om=0.3, ide=c, f0_linear=f0_lin,
                                       f0=f0_lin, **GRID)
    assert float(tw2[-1]) == f0_lin and float(tn2[-1]) == f0_lin
    for i in range(1, 29):  # everything but the f0 column is unchanged
        np.testing.assert_array_equal(np.asarray(tw2[i]), np.asarray(tw[i]))


def test_eager_and_jax_builders_agree(linear, bench_tables):
    k, P, Pnw = linear
    c, tw, tn, _ = bench_tables
    ew, en, _ = Kfuncs_to_tables(k, P, Pnw, z=Z, Om=0.3, ide=c, **GRID)
    for x, y in zip(ew + en, tw + tn):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-12, atol=0)


# ----------------------------------------------------------------------------
# g = 0 regression against the stock fkpt builder
# ----------------------------------------------------------------------------

def test_g0_ide_builder_matches_stock_jax_builder(linear):
    """Stock path started at z = 1000 (xnow = -ln 1001), where its third-order
    start-up transient has decayed. Residual ~3e-6 in P13: the stock CFD3 is
    still 3e-6 from converged there, plus its f0 estimator (2e-7)."""
    k, P, Pnw = linear
    c0 = ide_kernel_coeffs(Z, LCDM, z_ini=1000.0, solver="adaptive")
    stock = Kfuncs_to_tables_jax(k, P, Pnw, z=Z, Om=0.3, beyond_eds=True,
                                 fkpt_approximation=True, model="LCDM",
                                 xnow=-np.log(1001.0), **GRID)
    ide = Kfuncs_to_tables_jax(k, P, Pnw, z=Z, Om=0.3, ide=c0, **GRID)
    for t_s, t_i in ((stock[0], ide[0]), (stock[1], ide[1])):
        for i, (a, b) in enumerate(zip(t_s, t_i)):
            a, b = np.asarray(a), np.asarray(b)
            scale = max(float(np.max(np.abs(a))), 1e-300)
            np.testing.assert_allclose(b, a, rtol=0, atol=1e-5 * scale, err_msg=f"column {i}")


# ----------------------------------------------------------------------------
# RSD wrapper
# ----------------------------------------------------------------------------

def test_rsd_wrapper_R1_reproduces_folps(bench_tables):
    c, tw, tn, _ = bench_tables
    kap, muap, mu, wmu, _ = _mu_grid(tw)
    pars = pack_fkpt_bias(PARS)
    kw = dict(ncols=len(tw))
    stock = tables_to_poles(1.0, kap, muap, pars, tw + tn, mu, wmu, **kw)
    ours = tables_to_poles(1.0, kap, muap, pars, tw + tn, mu, wmu, **kw,
                           ide=c._replace(Gamma1=1.0, fPsi=tw[-1]))
    np.testing.assert_allclose(np.asarray(ours), np.asarray(stock), rtol=1e-13, atol=0)
    # the physical damping (Gamma1 = 1.11, fPsi = 0.73 here) does change the result
    real = tables_to_poles(1.0, kap, muap, pars, tw + tn, mu, wmu, **kw, ide=c)
    assert float(jnp.max(jnp.abs(real - stock) / jnp.abs(stock))) > 1e-4


def _zero_loops(table):
    t = list(table)
    for i in range(3, 29):  # loop, bias, TNS columns and sigma2w
        t[i] = jnp.zeros_like(jnp.asarray(t[i]))
    return tuple(t)


def test_linear_multipoles_are_kaiser_with_fQ(bench_tables):
    """Benchmark tables with the loops zeroed: the multipoles are the Kaiser
    formulas with fQ, and with IR resummation the tree level is
    (b1 + fQ mu^2)^2 [Pnw + e (P - Pnw)(1 + k^2 S)], with one S = Sigma2_tot(IDE)."""
    c, tw, tn, _ = bench_tables
    kap, muap, mu, wmu, sel = _mu_grid(tw)
    b1, fQ = 1.8, float(c.fQ)
    lin = pack_fkpt_bias({**{k: 0.0 for k in PARS}, "b1": b1})
    tz = _zero_loops(tw) + _zero_loops(tn)
    n = len(tw)

    poles = tables_to_poles(1.0, kap, muap, lin, tz, mu, wmu, ncols=n, IR_resummation=False, ide=c)
    Pq = jnp.asarray(tw[1])[sel]
    kaiser = jnp.stack([(b1**2 + 2 / 3 * b1 * fQ + fQ**2 / 5) * Pq,
                        (4 / 3 * b1 * fQ + 4 / 7 * fQ**2) * Pq,
                        8 / 35 * fQ**2 * Pq])
    np.testing.assert_allclose(np.asarray(poles), np.asarray(kaiser), rtol=1e-12)

    pkmu = tables_to_pkmu(kap, muap, lin, tz, ncols=n, IR_resummation=True, ide=c)
    S = ir_sigma2_tot(muap, tn[-3], tn[-2], c)
    damp = jnp.exp(-kap**2 * S)
    Pnw = jnp.asarray(tn[1])[sel][:, None]
    expected = (b1 + fQ * muap**2) ** 2 * (Pnw + damp * (Pq[:, None] - Pnw) * (1 + kap**2 * S))
    np.testing.assert_allclose(np.asarray(pkmu), np.asarray(expected), rtol=1e-12)


# ----------------------------------------------------------------------------
# pipelines.py
# ----------------------------------------------------------------------------

def _pipe_kwargs(linear, table):
    k, P, Pnw = linear
    kap, muap, mu, wmu, _ = _mu_grid(table)
    return dict(k=k, pk=P, pk_now=Pnw, jac=1.0, kap=kap, muap=muap,
                pars=pack_fkpt_bias(PARS), mu=mu, wmu=wmu)


def test_pipelines_ide_none_bit_identical_to_main(linear, bench_tables, tmp_path):
    main_pipe = _load_from_main("src/fkptjax/pipelines.py", "_pipelines_main", tmp_path,
                                package="fkptjax")
    kw = dict(**_pipe_kwargs(linear, bench_tables[1]), z=Z, Om=0.3, beyond_eds=True,
              model="LCDM", **GRID)
    for name in ("binning_jax_poles", "full_method_poles"):
        new_poles, new_state = getattr(__import__("fkptjax.pipelines", fromlist=[name]), name)(**kw)
        old_poles, old_state = getattr(main_pipe, name)(**kw)
        assert np.array_equal(np.asarray(new_poles), np.asarray(old_poles)), name
        _assert_tables_identical(new_state.table_all, old_state.table_all)
        assert new_state.ide is None


def test_pipelines_ide_route(linear, bench_tables):
    c, tw, tn, _ = bench_tables
    kw = dict(**_pipe_kwargs(linear, tw), z=Z, Om=0.3, **GRID)
    poles, state = binning_jax_poles(**kw, ide=c)
    assert state.ide is c and state.kernel_constants is c
    direct = tables_to_poles(1.0, kw["kap"], kw["muap"], kw["pars"], tw + tn, kw["mu"],
                             kw["wmu"], ncols=len(tw), ide=c)
    np.testing.assert_array_equal(np.asarray(poles), np.asarray(direct))
    eager, _ = full_method_poles(**kw, ide=c)
    np.testing.assert_allclose(np.asarray(eager), np.asarray(poles), rtol=1e-12)
    _, state_lin = binning_jax_poles(**kw, ide=c, f0_linear=0.81)
    assert float(state_lin.f0) == 0.81
    with pytest.raises(ValueError, match="does not support ide"):
        rescale_branch_poles(**kw, ide=c)
