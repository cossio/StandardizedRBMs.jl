# StandardizedRBMs Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/StandardizedRBMs.jl.jl/blob/master/LICENSE.md)
![](https://github.com/cossio/StandardizedRBMs.jl.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/StandardizedRBMs.jl.jl/branch/master/graph/badge.svg?token=90I3AJIZIG)](https://codecov.io/gh/cossio/StandardizedRBMs.jl.jl)

Train and sample a *standardized* Restricted Boltzmann machine in Julia. The energy is given by:

$$
E(\mathbf{v},\mathbf{h}) = - \sum_{i}\theta_{i}v_{i} - \sum_{\mu}\theta_{\mu}h_{\mu} - \sum_{i\mu}w_{i\mu} \frac{v_{i} - \lambda_{i}}{\sigma_{i}}\frac{h_{\mu} - \lambda_{\mu}}{\sigma_{\mu}}
$$

with some offset parameters $\lambda_i,\lambda_\mu$ and scaling parameters $\sigma_i,\sigma_\mu$.

## Installation

This package is not registered. Install with:

```julia
using Pkg
Pkg.add(url="https://github.com/cossio/StandardizedRBMs.jl")
```

This package does not export any symbols.

## Related

* https://github.com/cossio/RestrictedBoltzmannMachines.jl.