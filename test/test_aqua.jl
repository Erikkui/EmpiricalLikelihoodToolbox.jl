using Aqua

@testset "Aqua" begin
    # persistent_tasks is disabled because the DifferentialEquations dependency makes it slow.
    Aqua.test_all(EmpiricalLikelihoodToolbox;
                  persistent_tasks = false)
end
