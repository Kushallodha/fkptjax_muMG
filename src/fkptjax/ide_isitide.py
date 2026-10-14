"""
ide_isitide.py
--------------
Optional adapter: ISiTIDE (CAMB-based IDE Boltzmann code) background ->
:class:`fkptjax.ide.TabulatedBackground`.

ISiTIDE is imported lazily (only its results object is passed in), so fkptjax
does not depend on it.

Sign convention (ISiTIDE): rho_c' + 3 H rho_c = Q, rho_de' + 3 H (1+w) rho_de = -Q,
with +Q = energy into CDM and Q^mu parallel to u_c (no momentum transfer in the
CDM frame). The dilution rate of the cdm+baryon contrast is

    g = (Qbar - delta_Q / delta_cb) / (H rhobar_cb)        (delta_cb' + g delta_cb + ... = 0),

i.e. Qbar/(H rhobar_cb) when delta_Q = 0,
so g > 0 dilutes delta_cb. Supported couplings (smooth dark energy, delta_b = delta_c):

    IDEModel1, IDEModel1noppf, IDEModel1thaw:  Q = beta H  rho_de  ->  g = beta rho_de / rho_cb
    IDEModel3, IDEModel3thaw:                  Q = beta H0 rho_de  ->  g = beta (H0/H) rho_de / rho_cb
    IDEModel2, IDEModel2noppf, IDEModel2thaw:  Q = beta H  rho_c   ->  g = 0
    IDEModel4, IDEModel4thaw:                  Q = beta H0 rho_c   ->  g = 0
    IDEModel5:  Q = beta H  rho_de rho_c/(rho_de + rho_c)  ->  g = beta f_c s (1 - s)
    IDEModel6:  Q = beta H0 rho_de rho_c/(rho_de + rho_c)  ->  g = beta (H0/H) f_c s (1 - s)

with s = rho_c/(rho_c + rho_de) and f_c = rho_c/rho_cb. For Q proportional to rho_c the
coupling is local in the CDM density (delta_Q = Q delta_c at every order), so the dilution
cancels exactly and the IDE kernels are exact with g = 0; the coupling acts only through the
background (E, A_Phi).

Models 5/6 use a linear-coupling approximation: g is exact at linear order, but the cb
continuity equation also gets a nonlinear term g2 delta^2 with g2 = g (1 - s) (from
Q(delta) proportional to (1 + delta)/(1 + s delta)), which the IDE kernels drop. Adding its
second-order effect (local constants in F2 and G2) changes the one-loop multipoles by
<~0.2% at |beta| = 0.1 and 0.2-0.6% at |beta| = 0.3 (z = 0.5, k = 0.02-0.25 h/Mpc, b1 = 2),
about 3-4x less at z = 1; a free b2 absorbs part of the change in P0 only. The local
third-order terms (g2, g3) are not included either.
"""

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import CubicSpline

from fkptjax.ide import TabulatedBackground, Z_INI

_DE_COUPLED = ("IDEModel1", "IDEModel1noppf", "IDEModel1thaw", "IDEModel3", "IDEModel3thaw")
_C_COUPLED = ("IDEModel2", "IDEModel2noppf", "IDEModel2thaw", "IDEModel4", "IDEModel4thaw")
_MIXED_COUPLED = ("IDEModel5", "IDEModel6")
_H0_COUPLED = ("IDEModel3", "IDEModel3thaw", "IDEModel4", "IDEModel4thaw", "IDEModel6")
_SUPPORTED = _DE_COUPLED + _C_COUPLED + _MIXED_COUPLED
SUPPORTED_MODELS = _SUPPORTED
"""ISiTIDE dark_energy_model names :func:`ide_background` accepts."""


def ide_grid(z_ini=Z_INI, z_out_min=0.0, n_grid=4000, pad=0.02):
    """Uniform grid in N = ln a covering [ln a(z_ini) - pad, ln a(z_out_min)].

    Tabulate the background on it for every ``ide_kernel_coeffs(z_out, bg, z_ini=z_ini)``
    call with z_out >= z_out_min (cubic interpolation in between nodes).
    """
    return np.linspace(-np.log1p(z_ini) - pad, -np.log1p(z_out_min), int(n_grid))


def ide_background(model, beta, N, efunc, Omega_cdm, Omega_b, Omega_de):
    """IDE :class:`~fkptjax.ide.TabulatedBackground` from background functions on a uniform N grid.

    Parameters
    ----------
    model : str
        ISiTIDE ``dark_energy_model`` (see :data:`SUPPORTED_MODELS`).
    beta : float
        Coupling parameter.
    N : array
        Uniform, increasing grid in N = ln a (e.g. :func:`ide_grid`).
    efunc, Omega_cdm, Omega_b, Omega_de : arrays on N
        H(z)/H0 and the density parameters rho_x(z)/rho_crit(z), as returned by
        cosmoprimo's ``Background.efunc``/``Omega_cdm``/``Omega_b``/``Omega_de`` (or CAMB's
        ``get_Omega``) for the same IDE cosmology.

    Returns
    -------
    TabulatedBackground with E = 2 + dlnH/dN (cubic-spline derivative of ln efunc),
    A_Phi = 1.5 Omega_cb(z) and g as in the module docstring (H0/H = 1/efunc).
    """
    model = str(model)
    if model not in _SUPPORTED:
        raise ValueError(f"ide_background: dark_energy_model={model!r} is not supported; "
                         f"supported IDE models: {', '.join(_SUPPORTED)}.")
    N, efunc, Om_c, Om_b, Om_de = (np.asarray(x, dtype=np.float64)
                                   for x in (N, efunc, Omega_cdm, Omega_b, Omega_de))
    beta = float(beta)
    Om_cb = Om_c + Om_b
    E = 2.0 + CubicSpline(N, np.log(efunc))(N, 1)
    A_Phi = 1.5 * Om_cb
    if model in _DE_COUPLED:
        g = beta * Om_de / Om_cb
    elif model in _C_COUPLED:
        g = np.zeros_like(Om_cb)
    else:
        Om_c_de = Om_c + Om_de
        if np.any(np.abs(Om_c_de) <= 1e-8 * (np.abs(Om_c) + np.abs(Om_de))):
            raise ValueError(f"ide_background: rho_c + rho_de ~ 0 on the grid for {model!r}; "
                             "the coupling rho_de rho_c/(rho_c + rho_de) is singular there.")
        s = Om_c / Om_c_de
        g = beta * (Om_c / Om_cb) * s * (1.0 - s)
    if model in _H0_COUPLED:
        g = g / efunc
    return TabulatedBackground(N=jnp.asarray(N), E=jnp.asarray(E),
                               A_Phi=jnp.asarray(A_Phi), g=jnp.asarray(g))


def background_from_isitide(results, z_ini=Z_INI, z_out_min=0.0, n_grid=4000, pad=0.02):
    """Tabulate E, A_Phi and g from an ISiTIDE results object (via :func:`ide_background`).

    Parameters
    ----------
    results
        ISiTIDE ``CAMBdata`` (e.g. from ``isitide.get_background(params)``);
        only the background is used.
    z_ini, z_out_min, n_grid, pad
        Grid, see :func:`ide_grid`.

    Returns
    -------
    TabulatedBackground with
        E  = 2 + dlnH/dN    (cubic-spline derivative of ln H on the grid),
        A_Phi = 1.5 Omega_cb(z)   (rho_cb over the critical density, radiation and neutrinos included),
        g  = Q / (H rho_cb) - delta_Q / (H rho_cb delta_cb)  (see the module docstring).
    """
    de = results._get_dark_energy_params()
    model = de["dark_energy_model"]
    if model not in _SUPPORTED:
        raise ValueError(
            f"background_from_isitide: dark_energy_model={model!r} is not supported; "
            f"supported IDE models: {', '.join(_SUPPORTED)}.")
    beta = float(de.get("beta") or 0.0)
    N = ide_grid(z_ini=z_ini, z_out_min=z_out_min, n_grid=n_grid, pad=pad)
    z = 1.0 / np.exp(N) - 1.0
    H = np.asarray(results.hubble_parameter(z), dtype=np.float64)
    Om = {name: np.asarray(results.get_Omega(name, z=z), dtype=np.float64)
          for name in ("cdm", "baryon", "de")}
    return ide_background(model, beta, N, H / float(results.hubble_parameter(0.0)),
                          Om["cdm"], Om["baryon"], Om["de"])
