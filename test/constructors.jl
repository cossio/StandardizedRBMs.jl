using Test: @test
using Test: @testset
using RestrictedBoltzmannMachines: Spin
using StandardizedRestrictedBoltzmannMachines: SpinStandardizedRBM

rbm = SpinStandardizedRBM(randn(10), randn(7), randn(10, 7))
@test rbm.visible isa Spin
@test rbm.hidden isa Spin
