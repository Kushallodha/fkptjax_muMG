"""ISiTIDE adapter tests (skipped when isitide is not installed)."""

import numpy as np
import pytest

isitide = pytest.importorskip("isitide")

from fkptjax.ide import ide_kernel_coeffs
from fkptjax.ide_isitide import background_from_isitide

COSMO = dict(H0=67.5, ombh2=0.022, omch2=0.12, YHe=0.245, w=-1.0)


@pytest.mark.parametrize("model", ["IDEModel1", "IDEModel3", "IDEModel2", "IDEModel4",
                                   "IDEModel5", "IDEModel6"])
def test_growth_matches_isitide_transfer_functions(model):
    """Sub-horizon D(z)/D(0) from fkptjax.ide vs ISiTIDE's delta_nonu P(k).

    Models 2/4 run with g = 0 (the dilution cancels for Q proportional to rho_c),
    Models 5/6 with g = beta f_c s(1 - s). Needs ISiTIDE with the PPF w = -1 fix.
    """
    zs = [0.0, 0.5, 1.0]
    p = isitide.set_params(dark_energy_model=model, beta=0.2, **COSMO,
                           redshifts=zs, kmax=2.0, WantTransfer=True)
    res = isitide.get_results(p)
    bg = background_from_isitide(res)
    PK = res.get_matter_power_interpolator(nonlinear=False, var1="delta_nonu",
                                           var2="delta_nonu", hubble_units=True, k_hunit=True)
    D0 = float(ide_kernel_coeffs(0.0, bg).D)
    for z in zs[1:]:
        ratio = float(ide_kernel_coeffs(z, bg).D) / D0
        np.testing.assert_allclose(ratio, np.sqrt(PK.P(z, 0.1) / PK.P(0.0, 0.1)), rtol=3e-4)


def test_rejects_non_ide_models():
    p = isitide.set_params(dark_energy_model="ppf", w=-0.9,
                           **{k: v for k, v in COSMO.items() if k != "w"})
    with pytest.raises(ValueError, match="not supported"):
        background_from_isitide(isitide.get_background(p))


def test_growing_mode_ic_removes_z_ini_dependence_model2():
    """Model 2 (always-on coupling): with ic='growing' the second/third-order coefficients
    no longer depend on z_ini (L, F and the Sigma_3 amplitude S; EdS initial data: ~3e-3 at beta = 0.1 between z_ini = 30-100)."""
    p = isitide.set_params(dark_energy_model="IDEModel2", beta=0.1, **COSMO)
    bg = background_from_isitide(isitide.get_background(p), z_ini=100.0)
    zz = np.array([0.5, 1.0])
    ref = ide_kernel_coeffs(zz, bg, z_ini=50.0, ic="growing")
    for zi in (30.0, 100.0):
        c = ide_kernel_coeffs(zz, bg, z_ini=zi, ic="growing")
        for name in ("L", "F", "S"):
            np.testing.assert_allclose(getattr(c, name), getattr(ref, name), rtol=2e-4, err_msg=name)
