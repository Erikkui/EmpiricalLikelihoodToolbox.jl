# src/mcmc/prior_eval.jl -- review section 22.5: one prior contribution only, correct
# parameter-name association, rejection outside support.

@testset "priors" begin

    @testset "evaluate_single_prior" begin
        @test evaluate_single_prior(Normal(0.0, 1.0), 0.0) == logpdf(Normal(0.0, 1.0), 0.0)
        @test evaluate_single_prior(nothing, 123.0) == 0.0       # flat prior contributes nothing
        @test evaluate_single_prior(Uniform(0.0, 1.0), 2.0) == -Inf
    end

    @testset "evaluate_log_prior sums the contributions" begin
        priors = (a = Normal(0.0, 1.0), b = Uniform(0.0, 10.0))
        params = [0.5, 3.0]
        @test evaluate_log_prior(params, priors) ≈
              logpdf(Normal(0.0, 1.0), 0.5) + logpdf(Uniform(0.0, 10.0), 3.0)
    end

    @testset "only the non-nothing priors contribute" begin
        params = [0.5, 3.0, -1.0]
        mixed = (a = Normal(0.0, 1.0), b = nothing, c = nothing)
        @test evaluate_log_prior(params, mixed) ≈ logpdf(Normal(0.0, 1.0), 0.5)

        all_flat = (a = nothing, b = nothing, c = nothing)
        @test evaluate_log_prior(params, all_flat) == 0.0
    end

    @testset "no priors at all is a flat prior" begin
        @test evaluate_log_prior([1.0, 2.0], nothing) == 0.0
    end

    @testset "out-of-support parameters are rejected" begin
        priors = (a = Uniform(0.0, 1.0), b = Uniform(0.0, 1.0))
        @test evaluate_log_prior([0.5, 0.5], priors) > -Inf
        @test evaluate_log_prior([1.5, 0.5], priors) == -Inf      # first out of support
        @test evaluate_log_prior([0.5, -0.5], priors) == -Inf     # second out of support
    end

    @testset "a plain tuple of priors works too" begin
        # keys() of a Tuple is 1:N, so positional tuples are accepted alongside NamedTuples.
        @test evaluate_log_prior([0.5, 3.0], (Normal(0.0, 1.0), Uniform(0.0, 10.0))) ≈
              logpdf(Normal(0.0, 1.0), 0.5) + logpdf(Uniform(0.0, 10.0), 3.0)
    end

    @testset "priors pair with parameters positionally, not by name" begin
        # Documented current behaviour: evaluate_log_prior zips keys(priors) against params[ii]
        # by position. Reordering the NamedTuple therefore changes the result even though the
        # names still map each prior to a parameter. See test_known_issues.jl.
        params = [0.5, 5.0]
        in_order = (a = Uniform(0.0, 1.0), b = Uniform(0.0, 10.0))
        swapped  = (b = Uniform(0.0, 10.0), a = Uniform(0.0, 1.0))

        @test evaluate_log_prior(params, in_order) > -Inf
        @test evaluate_log_prior(params, swapped) == -Inf   # 5.0 scored against Uniform(0,1)
    end
end
