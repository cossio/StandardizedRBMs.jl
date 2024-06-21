module StandardizedRestrictedBoltzmannMachines

import LinearAlgebra
import Random
import RestrictedBoltzmannMachines
import Statistics

using FillArrays: Falses
using FillArrays: Zeros
using LinearAlgebra: cholesky
using LinearAlgebra: diagm
using LinearAlgebra: Diagonal
using LinearAlgebra: I
using LinearAlgebra: Symmetric
using Optimisers: AbstractRule
using Optimisers: Adam
using Optimisers: setup
using Optimisers: update!
using RestrictedBoltzmannMachines: ∂cgfs
using RestrictedBoltzmannMachines: ∂energy
using RestrictedBoltzmannMachines: ∂energy_from_moments
using RestrictedBoltzmannMachines: ∂free_energy
using RestrictedBoltzmannMachines: ∂interaction_energy
using RestrictedBoltzmannMachines: ∂RBM
using RestrictedBoltzmannMachines: ∂regularize!
using RestrictedBoltzmannMachines: AbstractLayer
using RestrictedBoltzmannMachines: batchcov
using RestrictedBoltzmannMachines: batchmean
using RestrictedBoltzmannMachines: batchvar
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: BinaryRBM
using RestrictedBoltzmannMachines: cgf
using RestrictedBoltzmannMachines: dReLU
using RestrictedBoltzmannMachines: energy
using RestrictedBoltzmannMachines: flatten
using RestrictedBoltzmannMachines: free_energy
using RestrictedBoltzmannMachines: Gaussian
using RestrictedBoltzmannMachines: grad2ave
using RestrictedBoltzmannMachines: hidden_cgf
using RestrictedBoltzmannMachines: infinite_minibatches
using RestrictedBoltzmannMachines: inputs_h_from_v
using RestrictedBoltzmannMachines: inputs_v_from_h
using RestrictedBoltzmannMachines: interaction_energy
using RestrictedBoltzmannMachines: log_pseudolikelihood
using RestrictedBoltzmannMachines: mean_from_inputs
using RestrictedBoltzmannMachines: moments_from_samples
using RestrictedBoltzmannMachines: Potts
using RestrictedBoltzmannMachines: pReLU
using RestrictedBoltzmannMachines: RBM
using RestrictedBoltzmannMachines: ReLU
using RestrictedBoltzmannMachines: rescale_activations!
using RestrictedBoltzmannMachines: sample_from_inputs
using RestrictedBoltzmannMachines: sample_v_from_v
using RestrictedBoltzmannMachines: shift_fields
using RestrictedBoltzmannMachines: shift_fields!
using RestrictedBoltzmannMachines: Spin
using RestrictedBoltzmannMachines: SpinRBM
using RestrictedBoltzmannMachines: var_from_inputs
using RestrictedBoltzmannMachines: wmean
using RestrictedBoltzmannMachines: xReLU
using RestrictedBoltzmannMachines: zerosum!

include("standardized_rbm.jl")
include("standardize.jl")
include("data.jl")
include("constructors.jl")
include("pcd.jl")
include("gauge.jl")
include("regularize.jl")
include("zerosum.jl")

end # module
