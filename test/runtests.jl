include("testutils.jl")

@testset verbose = true "EmpiricalLikelihoodToolbox.jl" begin
    # Phase 1 -- pure functions, no fixtures
    include("test_utils.jl")
    include("test_ecdf.jl")
    include("test_chamfer.jl")

    # Phase 2 -- component contracts
    include("test_resamplers.jl")
    include("test_models.jl")
    include("test_priors.jl")
    include("test_summaries.jl")

    # Phase 3 -- integration
    include("test_target.jl")
    include("test_loss.jl")
    include("test_mcmc.jl")
    include("test_integration.jl")

    # Phase 4 -- package quality and known issues
    include("test_aqua.jl")
    include("test_known_issues.jl")
end
