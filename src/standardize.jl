unstandardize(rbm::StandardizedRBM) = RBM(standardize(rbm))
unstandardize(rbm::RBM) = rbm

standardize(rbm::StandardizedRBM) = standardize(rbm, zero.(rbm.offset_v), zero.(rbm.offset_h), one.(rbm.scale_v), one.(rbm.scale_h))
standardize(rbm::RBM) = StandardizedRBM(rbm)

function standardize(
    rbm::StandardizedRBM,
    offset_v::AbstractArray, offset_h::AbstractArray,
    scale_v::AbstractArray, scale_h::AbstractArray
)
    @assert size(rbm.visible) == size(offset_v) == size(scale_v)
    @assert size(rbm.hidden) == size(offset_h) == size(scale_h)
    std_rbm = standardize_visible(rbm, offset_v, scale_v)
    return standardize_hidden(std_rbm, offset_h, scale_h)
end

standardize(
    rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray, scale_v::AbstractArray, scale_h::AbstractArray
) = standardize(standardize(rbm), offset_v, offset_h, scale_v, scale_h)

function standardize_visible(std_rbm::StandardizedRBM, offset_v::AbstractArray, scale_v::AbstractArray)
    @assert size(std_rbm.visible) == size(offset_v) == size(scale_v)

    cv = reshape(scale_v ./ std_rbm.scale_v, size(std_rbm.visible)..., map(one, size(std_rbm.hidden))...)
    Δθ = inputs_h_from_v(std_rbm, offset_v)

    hid = shift_fields(std_rbm.hidden, Δθ)
    w = std_rbm.w .* cv
    rbm = RBM(std_rbm.visible, hid, w)

    return StandardizedRBM(rbm, offset_v, std_rbm.offset_h, scale_v, std_rbm.scale_h)
end

function standardize_hidden(std_rbm::StandardizedRBM, offset_h::AbstractArray, scale_h::AbstractArray)
    @assert size(std_rbm.hidden) == size(offset_h) == size(scale_h)

    ch = reshape(scale_h ./ std_rbm.scale_h, map(one, size(std_rbm.visible))..., size(std_rbm.hidden)...)
    Δθ = inputs_v_from_h(std_rbm, offset_h)

    vis = shift_fields(std_rbm.visible, Δθ)
    w = std_rbm.w .* ch
    rbm = RBM(vis, std_rbm.hidden, w)

    return StandardizedRBM(rbm, std_rbm.offset_v, offset_h, std_rbm.scale_v, scale_h)
end

standardize_visible(rbm::RBM, offset_v::AbstractArray, scale_v::AbstractArray) = standardize_visible(standardize(rbm), offset_v, scale_v)
standardize_hidden(rbm::RBM, offset_h::AbstractArray, scale_h::AbstractArray) = standardize_hidden(standardize(rbm), offset_h, scale_h)

standardize_visible(rbm::StandardizedRBM) = standardize_visible(rbm, zero(rbm.offset_v), ones.(rbm.scale_v))
standardize_hidden(rbm::StandardizedRBM) = standardize_hidden(rbm, zero(rbm.offset_h), ones.(rbm.scale_h))
standardize_visible(rbm::RBM) = standardize(rbm)
standardize_hidden(rbm::RBM) = standardize(rbm)

function standardize!(rbm::StandardizedRBM, offset_v::AbstractArray, offset_h::AbstractArray, scale_v::AbstractArray, scale_h::AbstractArray)
    @assert size(rbm.visible) == size(offset_v) == size(scale_v)
    @assert size(rbm.hidden) == size(offset_h) == size(scale_h)
    standardize_visible!(rbm, offset_v, scale_v)
    standardize_hidden!(rbm, offset_h, scale_h)
    return rbm
end

function standardize_visible!(rbm::StandardizedRBM, offset_v::AbstractArray, scale_v::AbstractArray)
    @assert size(rbm.visible) == size(offset_v) == size(scale_v)

    cv = reshape(scale_v ./ rbm.scale_v, size(rbm.visible)..., map(one, size(rbm.hidden))...)
    Δθ = inputs_h_from_v(rbm, offset_v)

    shift_fields!(rbm.hidden, Δθ)
    rbm.w .= rbm.w .* cv
    rbm.offset_v .= offset_v
    rbm.scale_v .= scale_v

    return rbm
end

function standardize_hidden!(rbm::StandardizedRBM, offset_h::AbstractArray, scale_h::AbstractArray)
    @assert size(rbm.hidden) == size(offset_h) == size(scale_h)

    ch = reshape(scale_h ./ rbm.scale_h, map(one, size(rbm.visible))..., size(rbm.hidden)...)
    Δθ = inputs_v_from_h(rbm, offset_h)

    shift_fields!(rbm.visible, Δθ)
    rbm.w .= rbm.w .* ch
    rbm.offset_h .= offset_h
    rbm.scale_h .= scale_h

    return rbm
end
