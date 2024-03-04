# StandardizedRestrictedBoltzmannMachines Julia package

Train and sample a *standardized* Restricted Boltzmann machine in Julia. The energy is given by:

$$
E(\mathbf{v},\mathbf{h}) = - \sum_{i}\theta_{i}v_{i} - \sum_{\mu}\theta_{\mu}h_{\mu} - \sum_{i\mu}w_{i\mu} \frac{v_{i} - \lambda_{i}}{\sigma_{i}}\frac{h_{\mu} - \lambda_{\mu}}{\sigma_{\mu}}
$$

with some offset parameters $\lambda_i,\lambda_\mu$ and scaling parameters $\sigma_i,\sigma_\mu$. Usually $\lambda_i,\lambda_\mu$ track the mean activities of visible and hidden units, while $\sigma_i,\sigma_\mu$ track their standard deviations.

## Installation

This package is registered. Install with:

```julia
using Pkg
Pkg.add("StandardizedRestrictedBoltzmannMachines")
```

This package does not export any symbols.

## Related

* https://github.com/cossio/RestrictedBoltzmannMachines.jl.