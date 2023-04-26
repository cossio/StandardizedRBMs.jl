function RestrictedBoltzmannMachines.pcd!(
    rbm::StandardizedRBM,
    data::AbstractArray;

    batchsize::Int = 1,

    iters::Int = 1, # number of gradient updates

    steps::Int = 1,
    vm::AbstractArray = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),

    moments = moments_from_samples(rbm.visible, data), # sufficient statistics for visible layer

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    # "pseudocount" for estimating variances of v and h and damping
    damping::Real = 1//100,
    ϵv::Real = 0, ϵh::Real = 0,

    # optimiser
    optim::AbstractRule = Adam(),
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
    state = setup(optim, ps),

    rescale_hidden::Bool = true,
    shuffle::Bool = true,

    # called for every gradient update
    callback = Returns(nothing)
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert 0 ≤ damping ≤ 1

    standardize_visible_from_data!(rbm, data; ϵ = ϵv)

    for (iter, (vd,)) in zip(1:iters, infinite_minibatches(data; batchsize, shuffle))
        # update fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)

        # update standardization
        standardize_hidden_from_v!(rbm, vd; damping, ϵ=ϵh)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        rescale_hidden && rescale_hidden_activations!(rbm)

        callback(; rbm, optim, iter, vm, vd, ∂)
    end

    return state, ps
end
