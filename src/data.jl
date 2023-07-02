function standardize_visible_from_data!(rbm::StandardizedRBM, data::AbstractArray; wts = nothing, ϵ::Real = 0)
    μ = batchmean(rbm.visible, data; wts)
    ν = batchvar(rbm.visible, data; wts, mean=μ)
    return standardize_visible!(rbm, μ, sqrt.(ν .+ ϵ))
end

function standardize_hidden_from_inputs!(rbm::StandardizedRBM, inputs::AbstractArray; wts = nothing, damping::Real = 0, ϵ::Real = 0)
    μ, ν = hidden_statistics_from_inputs(rbm.hidden, inputs; wts)
    offset_h = (1 - damping) .* rbm.offset_h + damping .* μ
    scale_h = (1 - damping) .* rbm.scale_h.^2 + damping .* sqrt.(ν .+ ϵ)
    return standardize_hidden!(rbm, offset_h, scale_h)
end

function standardize_hidden_from_v!(rbm::StandardizedRBM, v::AbstractArray; wts = nothing, damping::Real = 0, ϵ::Real = 0)
    inputs = inputs_h_from_v(rbm, v)
    standardize_hidden_from_inputs!(rbm, inputs; damping, wts, ϵ)
end

function hidden_statistics_from_inputs(layer::AbstractLayer, inputs::AbstractArray; wts = nothing)
    h_ave = mean_from_inputs(layer, inputs)
    h_var = var_from_inputs(layer, inputs)
    μ = batchmean(layer, h_ave; wts)
    ν_int = batchmean(layer, h_var; wts)
    ν_ext = batchvar(layer, h_ave; wts, mean = μ)
    ν = ν_int + ν_ext # law of total variance
    return (; μ, ν)
end
