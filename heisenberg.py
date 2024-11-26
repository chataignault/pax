from jax import ShapeDtypeStruct
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import PRNGKeyArray, PyTree
from typing import Union
from diffrax import (
    AbstractBrownianPath,
    BrownianIncrement,
    SpaceTimeLevyArea,
    SpaceTimeTimeLevyArea,
)

import equinox as eqx
from diffrax import ControlTerm, diffeqsolve, Dopri5


class BrownianPath(AbstractBrownianPath):
    """
    Arranged version of Brownian Path class to perform Brownian lift to the Heisenberg group
    """

    key: PRNGKeyArray
    shape: PyTree[ShapeDtypeStruct] = eqx.field(static=True)
    levy_area: type[
        Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
    ] = eqx.field(static=True)

    def __init__(self, key, shape):
        self.shape = shape
        self.key = key
        self.levy_area = BrownianIncrement

    @property
    def t0(self):
        return -jnp.inf

    @property
    def t1(self):
        return jnp.inf

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        key = jr.fold_in(self.key, t0)

        return jnp.diag(jr.normal(key, shape=self.shape))


if __name__ == "__main__":
    rng = jr.PRNGKey(0)
    shape = (3,)
    path = BrownianPath(rng, shape)
    print(path.evaluate(0.0))

    # define CDE terms
    y0 = jnp.eye(3)
    control = BrownianPath(rng)

    def vector_field(t, y, args):
        return y @ jnp.diag(jnp.array([1.0, 1.0]), k=1)

    diffusion_term = ControlTerm(vector_field, control)

    solver = Dopri5()
    sol = diffeqsolve(
        terms=diffusion_term,
        solver=solver,
        y0=y0,
        t0=0.0,
        t1=1.0,
        dt0=0.05,
    )

    print(sol.stats)

    print(sol.ys)
