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
        @test evaluate_log_prior(params, priors, (:a, :b)) ≈
              logpdf(Normal(0.0, 1.0), 0.5) + logpdf(Uniform(0.0, 10.0), 3.0)
    end

    @testset "only the non-nothing priors contribute" begin
        params = [0.5, 3.0, -1.0]
        mixed = (a = Normal(0.0, 1.0), b = nothing, c = nothing)
        @test evaluate_log_prior(params, mixed, (:a, :b, :c)) ≈ logpdf(Normal(0.0, 1.0), 0.5)

        all_flat = (a = nothing, b = nothing, c = nothing)
        @test evaluate_log_prior(params, all_flat, (:a, :b, :c)) == 0.0
    end

    @testset "no priors at all is a flat prior" begin
        @test evaluate_log_prior([1.0, 2.0], nothing, (:a, :b)) == 0.0
    end

    @testset "out-of-support parameters are rejected" begin
        priors = (a = Uniform(0.0, 1.0), b = Uniform(0.0, 1.0))
        @test evaluate_log_prior([0.5, 0.5], priors, (:a, :b)) > -Inf
        @test evaluate_log_prior([1.5, 0.5], priors, (:a, :b)) == -Inf      # first out of support
        @test evaluate_log_prior([0.5, -0.5], priors, (:a, :b)) == -Inf     # second out of support
    end

    @testset "a parameter missing from priors falls back to a flat prior" begin
        params = [0.5, 3.0]
        partial = (a = Normal(0.0, 1.0),)   # no entry for :b
        @test evaluate_log_prior(params, partial, (:a, :b)) ≈ logpdf(Normal(0.0, 1.0), 0.5)
    end

    @testset "a plain tuple of priors is no longer supported" begin
        # Only `nothing` and NamedTuple priors are accepted -- a positional tuple can't be safely
        # matched to parameters, so it's a MethodError rather than a silent positional pairing.
        @test_throws MethodError evaluate_log_prior([0.5, 3.0], (Normal(0.0, 1.0), Uniform(0.0, 10.0)), (:a, :b))
    end

    @testset "priors are matched to parameters by name, not NamedTuple key order" begin
        params = [0.5, 5.0]
        in_order = (a = Uniform(0.0, 1.0), b = Uniform(0.0, 10.0))
        swapped  = (b = Uniform(0.0, 10.0), a = Uniform(0.0, 1.0))

        @test evaluate_log_prior(params, in_order, (:a, :b)) == evaluate_log_prior(params, swapped, (:a, :b))
        @test evaluate_log_prior(params, in_order, (:a, :b)) > -Inf
    end
end
