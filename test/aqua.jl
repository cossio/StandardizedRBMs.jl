import Aqua
import StandardizedRestrictedBoltzmannMachines

using Test: @testset

@testset "aqua" begin
    Aqua.test_all(
        StandardizedRestrictedBoltzmannMachines;
        ambiguities = (exclude = [reshape],),
    )
end
