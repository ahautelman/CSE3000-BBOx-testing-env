# BBOx: Black Box Optimization in Jax

BBOx is a tool for Procedural Generation of Industrial-Type Black-Box Optimization Problems
(also known as numerical, derivative-free, optimization). This library is aimed towards
benchmarking existing agent implementations and for generating training
environments for Reinforcement Learning agents. The library implements a number of classical 
optimization functions (under the `Function` interface) that can be modularly
composed with wrappers (under the `FunctionWrapper` interface). 
When an optimal argument to a test funtion is analytical, can be computed, or
can be computed accurately approximated, many of the `FunctionWrapper` implementations 
track the transformation of this argument through a recursive back-up. 

Bookkeeping of states and parameters is managed by the familiar `dm-haiku` library, 
this directly allows us to define proceduraly generated function parameters through the 
`haiku.initializers` module. See the examples down below, or the notebooks in 
`examples/quickstart` to get a general overview of our provided tools.

Since this library is written in Jax, functions can be batched with `jax.vmap` and also 
differentiated with `jax.grad`. In `bbox/examples` we show how this can be used to train
Recurrent Neural Network optimizers by exploiting a training distribution of differentiable
functions. See in our `examples` also the accompanying `bbox-agents` module, which implements
a variety of baselines and agents that can learn to optimize.

#### Installation
The package can be installed with the following options (not yet on PyPI):
```bash
python -m pip install bbox        # Base library
python -m pip install bbox[env]   # jit_env Environments + EnvWrappers
python -m pip install bbox[test]  # For running the unit-tests
```

Note: This repository is still in development, though we do not expect large API changes, some
things may still be refactored.


### Quickstart @ [`function_interface.ipynb`](https://github.com/joeryjoery/bbox/blob/main/examples/quickstart/function_interface.ipynb): Making custom BlackBoxFunctions.

New functions can be created by specifying a base function and wrapping these with
`FunctionWrapper` classes.

```python
# Usage example
import jax
import haiku as hk

from bbox import functions as fx
from bbox import wrappers as wx


@hk.transform
def my_fun(inputs: jax.Array) -> jax.Array:
    # Implement a noisy Ellipsoid function with an affine input transformation.
    f = fx.real.Ellipsoid() 
    
    f = wx.real.Translation(f, x_shift=2.0, y_shift=0.0)
    f = wx.real.UniformRotation(f)
    f = wx.real.WhiteNoise(f, stddev=0.1)
    
    if hk.running_init():
        # Registers function-optimum data at the outer most wrapper.
        f.register_optimum(inputs)
    return f(inputs)


key = jax.random.PRNGKey(123)
key_init, key_apply = jax.random.split(key)

x = jax.random.uniform(key, (2,))
params = my_fun.init(key_init, x)
y = my_fun.apply(params, key_apply, x)

print(x, y)
# >> [0.4878348  0.68895483] 3.0206795
```

We also provide some utilities to parse the parameter containers of the functions to, e.g., 
retrieve the analytical optima (if available). As an example, we can see that the registered
optimal value is the expected value over random keys at the optimal argument:

```python
from bbox import get_param, Parameter

x_opt = get_param(params, Parameter.OPTIMUM_LOCATION)
y_opt = get_param(params, Parameter.OPTIMUM_VALUE)

# Batch evaluate function across random-keys using jax.vmap
y_avg = jax.vmap(lambda k: my_fun.apply(params, k, x_opt))(
    jax.random.split(key, num=10_000)).mean()

print(x_opt, y_opt, y_avg)
# >> [0.03312814 2.8282332 ] 0.012413587 0.012142903
```

### Quickstart with `bbox.envs`

Example: creating a distribution of Sphere functions (the euclidean norm 
of the input vector) where the optimal argument is shifted by a Gaussian 
random variable.

```python

```


# Cite us

If our work was useful to your research, citing us would be appreciated.

```
@software{BBOx2023Github,
  author = {Joery de Vries},
  title = {{BBOx}: Black-Box Optimization in Jax},
  url = {http://github.com/joeryjoery/BBOx},
  version = {0.1.0},
  year = {2023},
}
```

# References

See `examples` for reference re-implementations of the following papers:

- Learning to Learn by Gradient Descent by Gradient Descent [[https://arxiv.org/abs/1606.04474](https://arxiv.org/abs/1606.04474)]
- Optimizing Chemical Reactions with Deep Reinforcement Learning [[https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00492](https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00492)]
- Learning to Learn without Gradient Descent by Gradient Descent [[https://arxiv.org/abs/1611.03824](https://arxiv.org/abs/1611.03824)]
