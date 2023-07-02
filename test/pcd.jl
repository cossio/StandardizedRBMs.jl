using Test: @test
using Statistics: mean
using LinearAlgebra: norm
using StandardizedRestrictedBoltzmannMachines: standardize
using RestrictedBoltzmannMachines: RBM, Spin, pcd!

rbm = standardize(RBM(Spin(; θ=zeros(10)), Spin(; θ=zeros(7)), randn(10, 7) / √10))
@test iszero(rbm.visible.θ) && iszero(rbm.hidden.θ)

data = ones(10, 4)
data[1:3, 2] .= -1
data[:, 3:4] .= -data[:, 1:2] # ensure data has zero mean
@test iszero(mean(data; dims=2))

state, ps = pcd!(
    rbm, data;
    ps = (; w=rbm.w), # train only weights
    steps=10, batchsize=4, iters=1000,
    ϵv=1f-1, ϵh=0f0, damping=1f-1
)

# The fields are not exaclty zero because centering introduces minor numerical fluctuations.
@test norm(rbm.visible.θ) < 1e-13
@test iszero(rbm.hidden.θ)
