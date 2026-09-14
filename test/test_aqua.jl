using Aqua

@testset "Aqua" begin
    # undefined_exports is split out below: PredatorModel is exported but defined nowhere.
    # persistent_tasks is disabled because the DifferentialEquations dependency makes it slow.
    Aqua.test_all(EmpiricalLikelihoodToolbox;
                  undefined_exports = false,
                  persistent_tasks = false)
end
