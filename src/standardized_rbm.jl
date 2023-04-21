struct StandardizedRBM{V,H,W,Ov,Oh,Sv,Sh}
    visible::V
    hidden::H
    w::W
    offset_v::Ov
    offset_h::Oh
    scale_v::Sv
    scale_h::Sh
    function StandardizedRBM(
        visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray,
        offset_v::AbstractArray, offset_h::AbstractArray,
        scale_v::AbstractArray, scale_h::AbstractArray
    )
        @assert size(w) == (size(visible)..., size(hidden)...)
        @assert size(visible) == size(offset_v) == size(scale_v)
        @assert size(hidden)  == size(offset_h) == size(scale_h)
        V, H, W = typeof(visible), typeof(hidden), typeof(w)
        Ov, Oh, Sv, Sh = typeof(offset_v), typeof(offset_h), typeof(scale_v), typeof(scale_h)
        return new{V,H,W,Ov,Oh,Sv,Sh}(visible, hidden, w, offset_v, offset_h, scale_v, scale_h)
    end
end

StandardizedRBM(
    rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray, scale_v::AbstractArray, scale_h::AbstractArray
) = StandardizedRBM(rbm.visible, rbm.hidden, rbm.w, offset_v, offset_h, scale_v, scale_h)

function StandardizedRBM(rbm::RBM)
    offset_v = (similar(rbm.w, size(rbm.visible)) .= 0)
    offset_h = (similar(rbm.w, size(rbm.hidden)) .= 0)
    scale_v = (similar(rbm.w, size(rbm.visible)) .= 1)
    scale_h = (similar(rbm.w, size(rbm.hidden)) .= 1)
    return StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)
end

RestrictedBoltzmannMachines.RBM(rbm::StandardizedRBM) = RBM(rbm.visible, rbm.hidden, rbm.w)

standardize_v(rbm::StandardizedRBM, v::AbstractArray) = (v .- rbm.offset_v) ./ rbm.scale_v
standardize_h(rbm::StandardizedRBM, h::AbstractArray) = (h .- rbm.offset_h) ./ rbm.scale_h

function RestrictedBoltzmannMachines.interaction_energy(rbm::StandardizedRBM, v::AbstractArray, h::AbstractArray)
    std_v = standardize_v(rbm, v)
    std_h = standardize_h(rbm, h)
    return interaction_energy(RBM(rbm), std_v, std_h)
end

function RestrictedBoltzmannMachines.inputs_h_from_v(rbm::StandardizedRBM, v::AbstractArray)
    std_v = standardize_v(rbm, v)
    inputs = inputs_h_from_v(RBM(rbm), std_v)
    return inputs ./ rbm.scale_h
end

function RestrictedBoltzmannMachines.inputs_v_from_h(rbm::StandardizedRBM, h::AbstractArray)
    scaled_h = standardize_h(rbm, h)
    inputs = inputs_v_from_h(RBM(rbm), scaled_h)
    return inputs ./ rbm.scale_v
end

function RestrictedBoltzmannMachines.mirror(rbm::StandardizedRBM)
    _rbm = mirror(RBM(rbm))
    return StandardizedRBM(_rbm, rbm.offset_h, rbm.offset_v, rbm.scale_h, rbm.scale_v)
end

function RestrictedBoltzmannMachines.free_energy(rbm::StandardizedRBM, v::AbstractArray)
    E = energy(rbm.visible, v)
    inputs = inputs_h_from_v(rbm, v)
    F = -cgf(rbm.hidden, inputs)
    ΔE = energy(Binary(; θ = rbm.offset_h), inputs)
    return E + F - ΔE
end

function RestrictedBoltzmannMachines.∂free_energy(
    rbm::StandardizedRBM, v::AbstractArray; wts = nothing,
    moments = moments_from_samples(rbm.visible, v; wts)
)
    inputs = inputs_h_from_v(rbm, v)
    ∂v = ∂energy_from_moments(rbm.visible, moments)
    ∂Γ = ∂cgfs(rbm.hidden, inputs)
    h = grad2ave(rbm.hidden, ∂Γ)

    ∂h = reshape(wmean(-∂Γ; wts, dims = (ndims(rbm.hidden.par) + 1):ndims(∂Γ)), size(rbm.hidden.par))
    ∂w = ∂interaction_energy(rbm, v, h; wts)

    return (visible = ∂v, hidden = ∂h, w = ∂w)
end

function RestrictedBoltzmannMachines.∂interaction_energy(
    rbm::StandardizedRBM, v::AbstractArray, h::AbstractArray; wts = nothing
)
    std_v = standardize_v(rbm, v)
    std_h = standardize_h(rbm, h)
    ∂w = ∂interaction_energy(RBM(rbm), std_v, std_h; wts)
    return ∂w
end

function RestrictedBoltzmannMachines.log_pseudolikelihood(rbm::StandardizedRBM, v::AbstractArray)
    return log_pseudolikelihood(unstandardize(rbm), v)
end

"""
    delta_energy(rbm)

Compute the (constant) energy shift with respect to the equivalent normal RBM.
"""
delta_energy(rbm::StandardizedRBM) = interaction_energy(rbm, zero(rbm.offset_v), zero(rbm.offset_h))
delta_energy(rbm::RBM) = 0
