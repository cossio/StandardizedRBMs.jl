function RestrictedBoltzmannMachines.∂regularize!(∂::∂RBM, rbm::StandardizedRBM; kwargs...)
    ∂regularize!(∂, RBM(rbm); kwargs...)
end
