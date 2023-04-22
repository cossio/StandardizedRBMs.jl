RestrictedBoltzmannMachines.∂regularize!(
    ∂::∂RBM, # unregularized gradient
    rbm::StandardizedRBM;
    l2_fields::Real = 0, # L2 regularization of visible unit fields
    l1_weights::Real = 0, # L1 regularization of weights
    l2_weights::Real = 0, # L2 regularization of weights
    l2l1_weights::Real = 0, # L2/L1 regularziation of weights (10.7554/eLife.39397, Eq. 8)
) = ∂regularize!(∂, RBM(rbm); l2_fields, l1_weights, l2_weights, l2l1_weights)
