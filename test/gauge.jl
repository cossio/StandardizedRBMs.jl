import Random
using Random: bitrand
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: free_energy
using RestrictedBoltzmannMachines: mean_h_from_v
using RestrictedBoltzmannMachines: mean_v_from_h
using RestrictedBoltzmannMachines: RBM
using RestrictedBoltzmannMachines: ReLU
using RestrictedBoltzmannMachines: var_h_from_v
using RestrictedBoltzmannMachines: var_v_from_h
using StandardizedRestrictedBoltzmannMachines: rescale_hidden_activations!
using StandardizedRestrictedBoltzmannMachines: standardize
using Statistics: mean
using Statistics: var
using Test: @inferred
using Test: @test
using Test: @testset

@testset "rescale_hidden_activations!" begin
    rbm = standardize(
        RBM(Binary(; θ=randn(3)), ReLU(; θ=randn(2), γ=0.1 .+ rand(2)), randn(3,2)),
        randn(3), randn(2), 0.1 .+ rand(3), 0.1 .+ rand(2)
    )

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
