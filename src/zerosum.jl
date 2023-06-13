function RestrictedBoltzmannMachines.zerosum!(rbm::StandardizedRBM)
    zerosum!(RBM(rbm))
    return rbm
end
