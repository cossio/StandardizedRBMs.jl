import Random
using Test: @test, @testset, @inferred
using Statistics: mean, var
using Random: bitrand
using RestrictedBoltzmannMachines: RBM, Binary, free_energy, ReLU,
    mean_h_from_v, var_h_from_v, mean_v_from_h, var_v_from_h
using StandardizedRestrictedBoltzmannMachines: standardize, rescale_hidden_activations!

@testset "rescale_hidden_activations!" begin
    rbm = standardize(
        RBM(Binary(; θ=randn(3)), ReLU(; θ=randn(2), γ=rand(2)), randn(3,2)),
        randn(3), randn(2), rand(3), rand(2)
    )
    rbm.hidden.γ .+= 0.5

    v = bitrand(3, 1000)
    F = free_energy(rbm, v)
    h_ave = mean_h_from_v(rbm, v)
    h_var = var_h_from_v(rbm, v)
    v_ave = mean_v_from_h(rbm, h_ave)
    v_var = var_v_from_h(rbm, h_ave)

    λ = copy(rbm.scale_h)
    rescale_hidden_activations!(rbm)

    @test mean_h_from_v(rbm, v) ≈ h_ave ./ λ
    @test var_h_from_v(rbm, v) ≈ h_var ./ λ.^2
    @test mean_v_from_h(rbm, h_ave ./ λ) ≈ v_ave
    @test var_v_from_h(rbm, h_ave ./ λ) ≈ v_var
    @test all(rbm.scale_h .≈ 1)
    @test free_energy(rbm, v) ≈ F .+ sum(log, λ)
end
