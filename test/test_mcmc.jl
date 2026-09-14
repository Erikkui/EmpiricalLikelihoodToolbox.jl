# src/mcmc/{AM,DRAM,runner}.jl -- review section 22.6: test AM and DRAM against a known
# multivariate Gaussian target, checking posterior mean/covariance and acceptance behaviour.

# A model that stands in for a simulator, paired with a calculate_loss method returning an exact
# Gaussian log-density. This exercises AM and DRAM completely unmodified: the method reproduces the
# real signature, including the rng_seed keyword AM passes for common random numbers.
struct GaussianTestModel <: AbstractSimulationModel
    p1::Float64
    p2::Float64
    all_parameters::Tuple{Vararg{Symbol}}
    active_parameters::Tuple{Vararg{Symbol}}
end
GaussianTestModel(; p1 = 0.0, p2 = 0.0) =
    GaussianTestModel(p1, p2, (:p1, :p2), (:p1, :p2))

const GAUSS_MU = [1.0, -2.0]
const GAUSS_SIGMA = [1.0 0.5; 0.5 2.0]

function EmpiricalLikelihoodToolbox.calculate_loss(
        params, target, ::GaussianTestModel, mcmc_options; rng_seed::UInt64 = rand(UInt64))
    logprior = evaluate_log_prior(params, target.priors)
    isinf(logprior) && return -Inf
    d = params .- GAUSS_MU
    return -0.5 * dot(d, GAUSS_SIGMA \ d) + logprior
end

# A minimal TargetData: the samplers only read `priors` and `options.verbose` off it.
function gaussian_target(; priors = (p1 = nothing, p2 = nothing))
    target, _ = make_target(wiggly(20), StandardECDF(3);
                            priors = priors, training_resamplings = 10)
    return target
end

@testset "MCMC" begin

    @testset "compute_log_path" begin
        dists = [MvNormal(zeros(2), Matrix(1.0I, 2, 2)),
                 MvNormal(zeros(2), Matrix(4.0I, 2, 2))]
        states = [[0.0, 0.0], [1.0, 0.5], [2.0, -1.0]]
        log_pi = [-1.0, -3.0, -2.5]

        @testset "base case is log_pi at the anchor plus the jump density" begin
            @test compute_log_path(states, log_pi, 1:2, dists) ≈
                  log_pi[1] + logpdf(dists[1], states[2] .- states[1])
            # Reversing the path swaps which state anchors it.
            @test compute_log_path(states, log_pi, [2, 1], dists) ≈
                  log_pi[2] + logpdf(dists[1], states[1] .- states[2])
        end

        @testset "recursive step combines the forward path, a rejection and the final jump" begin
            fwd = compute_log_path(states, log_pi, 1:2, dists)
            rev = compute_log_path(states, log_pi, [2, 1], dists)
            log_alpha = min(0.0, rev - fwd)
            expected = fwd + log(-expm1(log_alpha)) + logpdf(dists[2], states[3] .- states[1])  # log1mexp
            @test compute_log_path(states, log_pi, 1:3, dists) ≈ expected
        end

        @testset "a certain intermediate acceptance makes the path impossible" begin
            # log_alpha == 0 means the intermediate move is always accepted, so no path can
            # continue past it through a rejection.
            equal_pi = [-1.0, -1.0, -2.0]
            sym = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
            @test compute_log_path(sym, equal_pi, 1:3, dists) == -Inf
        end
    end

    @testset "AM recovers a known Gaussian" begin
        target = gaussian_target()
        model = GaussianTestModel()
        opts = MCMCOptions(nsteps = 40_000, mcmc_algorithm = AM(proposal_width = 0.5,
                                                                adaptation_interval = 50),
                           initial_params = [1.0, -2.0], update_interval = 10_000_000)

        Random.seed!(20240914)
        results, state = mcmcrun(target, model, opts)

        # mcmcrun swallows exceptions and returns a truncated chain, so completion must be
        # asserted explicitly or a crash would pass silently.
        @test results.current_iter[] == opts.nsteps
        @test size(results.chain) == (2, opts.nsteps)

        burn = 10_000
        chain = results.chain[:, burn:end]
        @test vec(mean(chain, dims = 2)) ≈ GAUSS_MU atol = 0.15
        @test cov(chain') ≈ GAUSS_SIGMA atol = 0.3

        @test 0.1 < results.acceptance[1] < 0.6
        # The proposal covariance must have adapted away from its initial I*proposal_width.
        @test !(state.proposal_cov ≈ Matrix(0.5I, 2, 2))
    end

    @testset "DRAM recovers a known Gaussian" begin
        target = gaussian_target()
        model = GaussianTestModel()
        opts = MCMCOptions(nsteps = 30_000,
                           mcmc_algorithm = DRAM(proposal_width = 0.5, adaptation_interval = 50,
                                                 n_stages = 2, proposal_scale = [1.0, 0.25]),
                           initial_params = [1.0, -2.0], update_interval = 10_000_000)

        Random.seed!(5150)
        results, state = mcmcrun(target, model, opts)

        @test results.current_iter[] == opts.nsteps

        burn = 8_000
        chain = results.chain[:, burn:end]
        @test vec(mean(chain, dims = 2)) ≈ GAUSS_MU atol = 0.2
        @test cov(chain') ≈ GAUSS_SIGMA atol = 0.35

        @testset "acceptance is tracked per stage" begin
            @test length(results.acceptance) == 2
            @test all(≥(0), results.acceptance)
            # Delayed rejection must actually rescue some stage-1 rejections.
            @test results.acceptance[2] > 0
            @test sum(results.acceptance) < 1.0
        end
    end

    @testset "priors shape the posterior" begin
        # A prior truncating p1 below its unconstrained mean must pull the posterior mean down.
        target = gaussian_target(priors = (p1 = Uniform(-5.0, 0.5), p2 = nothing))
        opts = MCMCOptions(nsteps = 20_000, mcmc_algorithm = AM(proposal_width = 0.5),
                           initial_params = [0.0, -2.0], update_interval = 10_000_000)

        Random.seed!(31337)
        results, _ = mcmcrun(target, GaussianTestModel(), opts)

        @test results.current_iter[] == opts.nsteps
        chain = results.chain[:, 5_000:end]
        @test all(≤(0.5 + 1e-9), chain[1, :])        # the chain respects the prior support
        @test mean(chain[1, :]) < GAUSS_MU[1]
    end

    @testset "runner bookkeeping" begin
        target = gaussian_target()
        model = GaussianTestModel()

        @testset "allocates a chain of the requested size" begin
            opts = MCMCOptions(nsteps = 200, mcmc_algorithm = AM(), initial_params = [1.0, -2.0])
            Random.seed!(1)
            results, _ = mcmcrun(target, model, opts)
            @test size(results.chain) == (2, 200)
            @test length(results.sschain) == 200
            @test results.current_iter[] == 200
            @test all(isfinite, results.chain)
            @test all(isfinite, results.sschain)
        end

        @testset "acceptance is reported as a rate" begin
            opts = MCMCOptions(nsteps = 500, mcmc_algorithm = AM(proposal_width = 0.5),
                               initial_params = [1.0, -2.0], update_interval = 10_000_000)
            Random.seed!(2)
            results, _ = mcmcrun(target, model, opts)
            @test 0.0 <= results.acceptance[1] <= 1.0
        end

        @testset "DRAM gets one acceptance slot per stage" begin
            for n_stages in (2, 3)
                opts = MCMCOptions(nsteps = 300,
                                   mcmc_algorithm = DRAM(proposal_width = 0.5, n_stages = n_stages,
                                                         proposal_scale = fill(0.5, n_stages)),
                                   initial_params = [1.0, -2.0], update_interval = 10_000_000)
                Random.seed!(3)
                results, _ = mcmcrun(target, model, opts)
                @test results.current_iter[] == 300
                @test length(results.acceptance) == n_stages
            end
        end

        @testset "a wrong-length prior tuple is silently replaced" begin
            # Documented current behaviour; the review argues this should throw (see
            # test_known_issues.jl). The run must at least still complete.
            bad = gaussian_target(priors = (p1 = Normal(0.0, 1.0),))
            opts = MCMCOptions(nsteps = 100, mcmc_algorithm = AM(), initial_params = [1.0, -2.0])
            Random.seed!(4)
            results, _ = mcmcrun(bad, model, opts)
            @test results.current_iter[] == 100
        end

        @testset "stuck chains get kicked" begin
            # update_interval consecutive rejections trigger a recomputation of the current loss.
            opts = MCMCOptions(nsteps = 2_000, mcmc_algorithm = AM(proposal_width = 1e-12),
                               initial_params = [1.0, -2.0], update_interval = 5)
            Random.seed!(6)
            results, _ = mcmcrun(target, model, opts)
            @test results.current_iter[] == 2_000
            @test results.stuck_kicks[] ≥ 0
        end

        @testset "initial_params defaults to a perturbation of the model's own values" begin
            opts = MCMCOptions(nsteps = 50, mcmc_algorithm = AM(), initial_params = nothing)
            Random.seed!(7)
            results, _ = mcmcrun(target, GaussianTestModel(p1 = 1.0, p2 = -2.0), opts)
            @test results.current_iter[] == 50
        end
    end
end
