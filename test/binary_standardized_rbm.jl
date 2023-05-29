using Test: @test, @testset, @inferred
using Random: bitrand
using Statistics: mean
using LinearAlgebra: I
using RestrictedBoltzmannMachines: RBM, BinaryRBM,
    energy, interaction_energy, free_energy, ∂free_energy,
    inputs_h_from_v, inputs_v_from_h
using StandardizedRestrictedBoltzmannMachines: BinaryStandardizedRBM, StandardizedRBM, delta_energy,
    standardize, unstandardize, standardize_visible, standardize_hidden,
    standardize!, standardize_visible!, standardize_hidden!

using Zygote: gradient

@testset "standardize" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    offset_v = randn(3)
    offset_h = randn(2)
    scale_v = rand(3)
    scale_h = rand(2)

    std_rbm = @inferred standardize(rbm, offset_v, offset_h, scale_v, scale_h)
    @test std_rbm.offset_v ≈ offset_v
    @test std_rbm.offset_h ≈ offset_h
    @test std_rbm.scale_v ≈ scale_v
    @test std_rbm.scale_h ≈ scale_h

    @test std_rbm.w ./ (reshape(scale_v, 3, 1) .* reshape(scale_h, 1, 2)) ≈ rbm.w

    v = bitrand(3, 10)
    h = bitrand(2, 10)
    @test energy(std_rbm, v, h) .- delta_energy(std_rbm) ≈ energy(rbm, v, h)
    @test free_energy(std_rbm, v) .- delta_energy(std_rbm) ≈ free_energy(rbm, v)

    @test iszero(standardize(rbm).offset_v)
    @test iszero(standardize(rbm).offset_h)
    @test all(standardize(rbm).scale_v .== 1)
    @test all(standardize(rbm).scale_h .== 1)
    @inferred standardize(rbm)
    @test iszero(standardize(std_rbm).offset_v)
    @test iszero(standardize(std_rbm).offset_h)
    @test all(standardize(std_rbm).scale_v .== 1)
    @test all(standardize(std_rbm).scale_h .== 1)
    @inferred standardize(std_rbm)
    @test energy(rbm, v, h) ≈ energy(standardize(rbm), v, h) ≈ energy(standardize(std_rbm), v, h)
end

@testset "unstandardize" begin
    std_rbm = @inferred BinaryStandardizedRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2), rand(3), rand(2))
    rbm = @inferred unstandardize(std_rbm)
    @test rbm isa RBM
    @test unstandardize(rbm) == rbm
    v = bitrand(3, 10)
    h = bitrand(2, 10)
    @test energy(std_rbm, v, h) .- delta_energy(std_rbm) ≈ energy(rbm, v, h)
end

@testset "delta_energy" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    std_rbm = @inferred standardize(rbm, randn(3), randn(2), rand(3), rand(2))
    @test iszero(@inferred delta_energy(rbm))
    @test iszero(delta_energy(standardize(std_rbm)))
    @test delta_energy(rbm) isa Real
    @test delta_energy(std_rbm) isa Real
end

@testset "standardize!" begin
    rbm = @inferred BinaryStandardizedRBM(
        randn(3), randn(2), randn(3,2),
        randn(3), randn(2), rand(3), rand(2)
    )
    offset_v = randn(3)
    offset_h = randn(2)
    scale_v = rand(3)
    scale_h = rand(2)

    v = bitrand(3, 10)
    h = bitrand(2, 10)
    E = energy(rbm, v, h) .- delta_energy(rbm)
    F = free_energy(rbm, v) .- delta_energy(rbm)

    standardize!(rbm, offset_v, offset_h, scale_v, scale_h)
    @test rbm.offset_v ≈ offset_v
    @test rbm.offset_h ≈ offset_h
    @test rbm.scale_v ≈ scale_v
    @test rbm.scale_h ≈ scale_h

    @test energy(rbm, v, h) .- delta_energy(rbm) ≈ E
    @test free_energy(rbm, v) .- delta_energy(rbm) ≈ F
end

@testset "∂free energy" begin
    rbm = @inferred BinaryStandardizedRBM(
        randn(3), randn(2), randn(3,2),
        randn(3), randn(2), rand(3), rand(2)
    )
    v = bitrand(size(rbm.visible)..., 10)
    gs = gradient(rbm) do rbm
        mean(free_energy(rbm, v))
    end
    ∂ = ∂free_energy(rbm, v)
    @test ∂.visible ≈ only(gs).visible.par
    @test ∂.hidden ≈ only(gs).hidden.par
    @test ∂.w ≈ only(gs).w
end
