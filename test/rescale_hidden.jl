import Random
using Test: @test, @testset, @inferred
using Statistics: mean, var
using Random: bitrand
using RestrictedBoltzmannMachines: RBM, Binary, free_energy, ReLU,
    sample_v_from_v, sample_h_from_h
using StandardizedRBMs: standardize, rescale_hidden_activations!
using FillArrays: Fill

Random.seed!(23)

@testset "rescale_hidden_activations!" begin
    rbm = standardize(
        RBM(Binary(; θ=randn(3)), ReLU(; θ=randn(2), γ=rand(2)), randn(3,2)),
        randn(3), randn(2), rand(3), rand(2)
    )
    rbm.hidden.γ .+= 0.5

    v = sample_v_from_v(rbm, bitrand(size(rbm.visible)..., 20000); steps=100)
    h = sample_h_from_h(rbm, rand(size(rbm.hidden)..., 20000); steps=100)
    ave_v = mean(v; dims=2)
    ave_h = mean(h; dims=2)
    var_v = var(v; dims=2)
    var_h = var(h; dims=2)
    F = free_energy(rbm, v)

    λ = copy(rbm.scale_h)
    rescale_hidden_activations!(rbm)

    @test free_energy(rbm, v) ≈ F .+ sum(log, λ)

    v = sample_v_from_v(rbm, bitrand(size(rbm.visible)..., 20000); steps=100)
    h = sample_h_from_h(rbm, rand(size(rbm.hidden)..., 20000); steps=100)

    @test mean(v; dims=2) ≈ ave_v rtol=0.1
    @test mean(h; dims=2) ≈ ave_h ./ λ rtol=0.1
    @test var(v; dims=2) ≈ var_v rtol=0.1
    @test var(h; dims=2) ≈ var_h ./ λ.^2 rtol=0.1
end
