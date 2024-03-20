function BinaryStandardizedRBM(
    a::AbstractArray, b::AbstractArray, w::AbstractArray,
    offset_v::AbstractArray, offset_h::AbstractArray,
    scale_v::AbstractArray, scale_h::AbstractArray
)
    rbm = BinaryRBM(a, b, w)
    return StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)
end

function BinaryStandardizedRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    rbm = BinaryRBM(a, b, w)
    return standardize(rbm)
end

function SpinStandardizedRBM(
    a::AbstractArray, b::AbstractArray, w::AbstractArray,
    offset_v::AbstractArray, offset_h::AbstractArray,
    scale_v::AbstractArray, scale_h::AbstractArray
)
    rbm = SpinRBM(a, b, w)
    return StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)
end

function SpinStandardizedRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    rbm = SpinRBM(a, b, w)
    return standardize(rbm)
end
