"""Return mapping of vonMisesIsotropicHardening at the yield onset, against the exact solution.

Under uniaxial strain ``eps = a e_x`` the von Mises stress is ``2 mu a`` and the flow direction
has ``n_xx = 1``, so the update reduces to ``sig_xx = (lmbda + 2 mu) a - 2 mu dp`` with ``dp``
the root of ``sig_eq - 3 mu dp - sigma_Y(p + dp) = 0`` (zero if the trial is elastic).
"""

import jax
import jax.numpy as jnp
import pytest

import jaxmat.materials as jm
from jaxmat.tensors import SymmetricTensor2

E, NU = 70e3, 0.3
hardening = jm.VoceHardening(sig0=350.0, sigu=500.0, b=1e3)
material = jm.vonMisesIsotropicHardening(
    elasticity=jm.LinearElasticIsotropic(E=E, nu=NU), yield_stress=hardening
)
mu = material.elasticity.mu
lmbda = material.elasticity.lmbda
a0 = 350.0 / (2 * mu)  # onset of the first loading
DELTAS = [-1e-6, -1e-7, -1e-9, 1e-9, 1e-7, 1e-6, 1e-4, 1e-2]


def _strain(a):
    return SymmetricTensor2(array=jnp.array([a, 0.0, 0.0, 0.0, 0.0, 0.0]))


# The update's own body, below `filter_jit` and `assert_max_traces`: the trace counter is shared
# with every other test calling `constitutive_update` (e.g. with integer strains), so a float
# strain here would be its second trace.
_update = jax.jit(jm.vonMisesIsotropicHardening.constitutive_update.__wrapped__.__wrapped__)


def _jaxmat(a, state):
    """sig_xx and dp of one jaxmat step to ``a e_x`` from ``state``."""
    sig, new = _update(material, _strain(a), state, 0.0)
    return sig.array[0], new.internal.p - state.internal.p


def _exact_dp(sig_eq, p_old):
    """Root of the scalar return mapping, by Newton to round-off; derivative w.r.t. ``sig_eq``."""
    f = lambda dp: sig_eq - 3 * mu * dp - hardening(p_old + dp)  # noqa: E731
    slope = lambda dp: 3 * mu + jax.grad(hardening)(p_old + dp)  # noqa: E731
    if f(0.0) <= 0:
        return 0.0, 0.0
    dp = 0.0
    for _ in range(50):
        dp = dp + f(dp) / slope(dp)
    return dp, 1.0 / slope(dp)


def _check(a, state, sig_eq, p_old, sig_xx_old, a_old):
    dp, ddp = _exact_dp(sig_eq, p_old)
    sig_xx = sig_xx_old + (lmbda + 2 * mu) * (a - a_old) - 2 * mu * dp
    dsig_xx = (lmbda + 2 * mu) - 2 * mu * ddp * 2 * mu  # d sig_eq / da = 2 mu
    s, dp_jaxmat = _jaxmat(a, state)
    ds = jax.grad(lambda x: _jaxmat(x, state)[0])(a)
    assert dp_jaxmat >= 0
    assert abs(dp_jaxmat - dp) < 1e-14 + 1e-7 * dp  # the solver's rtol is 1e-8
    assert abs(s - sig_xx) < 1e-9 * abs(sig_xx)
    assert abs(ds - dsig_xx) < 1e-8 * abs(dsig_xx)


@pytest.mark.parametrize("delta", DELTAS)
def test_first_loading_at_the_onset(delta):
    a = a0 * (1 + delta)
    _check(a, material.init_state(), 2 * mu * a, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("delta", DELTAS)
def test_reloading_at_the_previous_yield_stress(delta):
    """Load into the plastic range, unload elastically, reload to just below or above the
    previous yield stress: the onset is now at ``sigma_Y(p1)``."""
    a1 = 1.5 * a0
    _, state = _update(material, _strain(a1), material.init_state(), 0.0)
    _, state = _update(material, _strain(0.0), state, 0.0)
    p1 = state.internal.p
    a = a1 * (1 + delta)
    # back along e_x the trial deviator stays collinear: sig_eq = sigma_Y(p1) + 2 mu (a - a1)
    sig_eq = hardening(p1) + 2 * mu * (a - a1)
    _check(a, state, sig_eq, p1, state.stress.array[0], 0.0)
