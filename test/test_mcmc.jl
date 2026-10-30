# src/mcmc/{AM,DRAM,runner}.jl -- review section 22.6: test AM and DRAM against a known
# multivariate Gaussian target, checking posterior mean/covariance and acceptance behaviour.

# A model that stands in for a simulator, paired with a GSL log_likelihood! method returning an exact
# Gaussian log-density of the parameters set in the model. This exercises AM and DRAM, and the real
# evaluate_log_target! (prior, model update, noise), completely unmodified.
struct GaussianTestModel <: AbstractSimulationModel
    p1::Float64
    p2::Float64
    all_parameters::Tuple{Vararg{Symbol}}
    active_parameters::Tuple{Vararg{Symbol}}
    dt_obs::Float64     # TargetData reads the derivative step from the model
end
GaussianTestModel(; p1 = 0.0, p2 = 0.0, dt_obs = 1.0) =
    GaussianTestModel(p1, p2, (:p1, :p2), (:p1, :p2), dt_obs)

const GAUSS_MU = [1.0, -2.0]
const GAUSS_SIGMA = [1.0 0.5; 0.5 2.0]

function EmpiricalLikelihoodToolbox.log_likelihood!(
        method_data, ::GSL, inference_state, target, model::GaussianTestModel, mcmc_options, rng)
    offset = [model.p1, model.p2] .- GAUSS_MU
    return -0.5 * dot(offset, GAUSS_SIGMA \ offset)
end

# A minimal TargetData: the samplers only read `priors`, `options.verbose` and `model` off it.
function gaussian_target(; priors = (p1 = nothing, p2 = nothing), model = GaussianTestModel())
    target, _ = make_target(wiggly(20), StandardECDF(3);
                            priors = priors, model = model, training_resamplings = 10)
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
        opts = MCMCOptions(nsteps = 40_000, mcmc_algorithm = AM(proposal_width = 0.5,
                                                                adaptation_interval = 50),
                           initial_params = [1.0, -2.0], update_interval = 10_000_000)

        Random.seed!(20240914)
        results, state = mcmcrun(target, opts)

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
        opts = MCMCOptions(nsteps = 30_000,
                           mcmc_algorithm = DRAM(proposal_width = 0.5, adaptation_interval = 50,
                                                 n_stages = 2, proposal_scale = [1.0, 0.25]),
                           initial_params = [1.0, -2.0], update_interval = 10_000_000)

        Random.seed!(5150)
        results, state = mcmcrun(target, opts)

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
        results, _ = mcmcrun(target, opts)

        @test results.current_iter[] == opts.nsteps
        chain = results.chain[:, 5_000:end]
        @test all(≤(0.5 + 1e-9), chain[1, :])        # the chain respects the prior support
        @test mean(chain[1, :]) < GAUSS_MU[1]
    end

    @testset "runner bookkeeping" begin
        target = gaussian_target()

        @testset "allocates a chain of the requested size" begin
            opts = MCMCOptions(nsteps = 200, mcmc_algorithm = AM(), initial_params = [1.0, -2.0])
            Random.seed!(1)
            results, _ = mcmcrun(target, opts)
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
            results, _ = mcmcrun(target, opts)
            @test 0.0 <= results.acceptance[1] <= 1.0
        end

        @testset "DRAM gets one acceptance slot per stage" begin
            for n_stages in (2, 3)
                opts = MCMCOptions(nsteps = 300,
                                   mcmc_algorithm = DRAM(proposal_width = 0.5, n_stages = n_stages,
                                                         proposal_scale = fill(0.5, n_stages)),
                                   initial_params = [1.0, -2.0], update_interval = 10_000_000)
                Random.seed!(3)
                results, _ = mcmcrun(target, opts)
                @test results.current_iter[] == 300
                @test length(results.acceptance) == n_stages
            end
        end

        @testset "a wrong-length prior tuple warns and falls back to flat priors" begin
            # A parameter missing from the priors NamedTuple gets a flat prior and a warning
            # naming it when the target is built, rather than the whole priors tuple being discarded.
            bad = @test_logs (:warn, r"p2") gaussian_target(priors = (p1 = Normal(0.0, 1.0),))
            opts = MCMCOptions(nsteps = 100, mcmc_algorithm = AM(), initial_params = [1.0, -2.0])
            Random.seed!(4)
            results, _ = mcmcrun(bad, opts)
            @test results.current_iter[] == 100
        end

        @testset "mcmcrun runs the model stored in the target" begin
            @test target.model isa GaussianTestModel
        end

        @testset "stuck chains get kicked" begin
            # update_interval consecutive rejections trigger a recomputation of the current loss.
            opts = MCMCOptions(nsteps = 2_000, mcmc_algorithm = AM(proposal_width = 1e-12),
                               initial_params = [1.0, -2.0], update_interval = 5)
            Random.seed!(6)
            results, _ = mcmcrun(target, opts)
            @test results.current_iter[] == 2_000
            @test results.stuck_kicks[] ≥ 0
        end

        @testset "update_interval <= 0 disables the stuck-chain heuristic" begin
            # A huge proposal at the mode is rejected almost always, so the chain is stuck.
            wide = AM(proposal_width = 1e6)
            kicks_for(interval; discard = false) = begin
                opts = MCMCOptions(nsteps = 300, mcmc_algorithm = wide, initial_params = [1.0, -2.0],
                                   update_interval = interval, discard_noisy_updates = discard)
                Random.seed!(8)
                results, _ = mcmcrun(target, opts)
                @test results.current_iter[] == 300
                results.stuck_kicks[]
            end

            @test kicks_for(5) > 0
            @test kicks_for(5; discard = true) > 0
            @test kicks_for(0) == 0
            @test kicks_for(-1) == 0
            @test kicks_for(0; discard = true) == 0
        end

        @testset "an explicit loss_function must match the inference method's loss" begin
            Random.seed!(9)
            matching = MCMCOptions(nsteps = 5, mcmc_algorithm = AM(), loss_function = LogLikelihood(),
                                   initial_params = [1.0, -2.0])
            results, _ = mcmcrun(target, matching)
            @test results.current_iter[] == 5

            mismatched = MCMCOptions(nsteps = 5, mcmc_algorithm = AM(), loss_function = RobustChamfer(),
                                     initial_params = [1.0, -2.0])
            @test_throws ArgumentError mcmcrun(target, mismatched)
            scaled = MCMCOptions(nsteps = 5, mcmc_algorithm = AM(), loss_function = LogLikelihood(2.0),
                                 initial_params = [1.0, -2.0])
            @test_throws ArgumentError mcmcrun(target, scaled)
        end

        @testset "initial_params defaults to a perturbation of the model's own values" begin
            opts = MCMCOptions(nsteps = 50, mcmc_algorithm = AM(), initial_params = nothing)
            Random.seed!(7)
            start_target = gaussian_target(model = GaussianTestModel(p1 = 1.0, p2 = -2.0))
            results, _ = mcmcrun(start_target, opts)
            @test results.current_iter[] == 50
            # The perturbation is 1 % of the model's own values, so the chain starts next to them.
            @test results.chain[:, 1] ≈ [1.0, -2.0] atol = 0.5
        end
    end

    @testset "AdaptiveProposal" begin
        ELT_AP = EmpiricalLikelihoodToolbox
        npar = 3
        n_points = 300
        points = randn(StableRNG(21), npar, n_points) .* [1.0, 2.0, 0.5]
        initial_cov = Matrix(0.1I, npar, npar)

        @testset "the adapted covariance is the scaled sample covariance plus 1e-6 I" begin
            proposal = ELT_AP.AdaptiveProposal(points[:, 1], initial_cov, 50)
            changed = [ELT_AP.update_proposal_covariance!(proposal, points[:, step], step) for step in 1:n_points]
            # Adapted only after step 100, every adaptation_interval steps
            @test findall(changed) == [150, 200, 250, 300]
            expected = cov(points') .* (2.38^2 / npar) + 1e-6 * I
            @test proposal.proposal_cov ≈ expected
            @test proposal.proposal_cov_L * proposal.proposal_cov_L' ≈ expected
            @test proposal.running_mean ≈ vec(mean(points, dims = 2))
        end

        @testset "backup_proposal_adaptation! and restore_proposal_adaptation! round-trip the running mean and covariance" begin
            proposal = ELT_AP.AdaptiveProposal(points[:, 1], initial_cov, 50)
            for step in 1:120
                ELT_AP.update_proposal_covariance!(proposal, points[:, step], step)
            end
            ELT_AP.backup_proposal_adaptation!(proposal)
            saved_mean = copy(proposal.running_mean)
            saved_cov = copy(proposal.running_cov)
            for step in 121:n_points
                ELT_AP.update_proposal_covariance!(proposal, points[:, step], step)
            end
            @test proposal.running_mean != saved_mean
            ELT_AP.restore_proposal_adaptation!(proposal)
            @test proposal.running_mean == saved_mean
            @test proposal.running_cov == saved_cov
        end

        @testset "propose_parameters! draws current + sqrt(scale) L z" begin
            proposal = ELT_AP.AdaptiveProposal(zeros(npar), [2.0 0.5 0.0; 0.5 1.0 0.0; 0.0 0.0 3.0], 50)
            current_params = [1.0, -1.0, 0.5]
            params_out = zeros(npar)
            Random.seed!(99)
            ELT_AP.propose_parameters!(params_out, proposal, current_params; scale = 0.25)
            Random.seed!(99)
            noise = randn(npar)
            @test params_out ≈ current_params .+ sqrt(0.25) .* (proposal.proposal_cov_L * noise)
        end
    end

    @testset "DRAM honours the noisy-likelihood policy" begin
        target = gaussian_target()

        # A huge proposal at the mode is rejected almost always, so the chain is stuck.
        kicks_for(interval; discard = false) = begin
            opts = MCMCOptions(nsteps = 300, mcmc_algorithm = DRAM(proposal_width = 1e6),
                               initial_params = [1.0, -2.0], update_interval = interval,
                               discard_noisy_updates = discard)
            Random.seed!(8)
            results, _ = mcmcrun(target, opts)
            @test results.current_iter[] == 300
            @test all(isfinite, results.sschain)
            results.stuck_kicks[]
        end

        @test kicks_for(5) > 0
        @test kicks_for(5; discard = true) > 0
        @test kicks_for(0) == 0
        @test kicks_for(0; discard = true) == 0
    end

    @testset "reevaluate_current_loss re-evaluates the unmoved current state" begin
        # With a noisy log-target, a re-evaluated state gets new noise at every step, so the stored
        # log-target changes even when the chain does not move. Without re-evaluation it stays.
        target = gaussian_target()
        for algorithm in (AM(proposal_width = 1e6), DRAM(proposal_width = 1e6))
            unmoved_changes(reevaluate) = begin
                opts = MCMCOptions(nsteps = 200, mcmc_algorithm = algorithm,
                                   initial_params = [1.0, -2.0], update_interval = 0,
                                   reevaluate_current_loss = reevaluate, likelihood_noise_scale = 0.1)
                Random.seed!(10)
                results, _ = mcmcrun(target, opts)
                chain = results.chain
                unmoved = [step for step in 2:200 if chain[:, step] == chain[:, step - 1]]
                @test length(unmoved) > 100
                count(step -> results.sschain[step] != results.sschain[step - 1], unmoved)
            end
            @test unmoved_changes(false) == 0
            @test unmoved_changes(true) > 100
        end
    end

    @testset "a mid-chain failure returns the partial chain" begin
        # runner.jl catches the exception, truncates at the last completed iteration and returns.
        target, _ = make_target(wiggly(20), StandardECDF(3); model = ExplodingModel(),
                                resampling_type = NoResampling(), training_resamplings = 10)
        opts = MCMCOptions(nsteps = 200, mcmc_algorithm = AM(), initial_params = [0.0, 0.0],
                           likelihood_noise_scale = 0.0)

        EXPLODE_CALLS[] = 0
        results, _ = mcmcrun(target, opts)

        iters = results.current_iter[]
        @test 0 < iters < opts.nsteps                  # truncated, not a complete run
        @test size(results.chain) == (2, iters)
        @test length(results.sschain) == iters
        @test all(isfinite, results.chain)
    end

    @testset "ResultsBuffer can hold an Int alongside a Ref" begin
        # current_iter and stuck_kicks have independent type parameters, so one can be a plain
        # Int while the other is a Ref.
        @test (try
            EmpiricalLikelihoodToolbox.ResultsBuffer(zeros(2, 3), zeros(3), zeros(2, 2),
                                                     3, zeros(1), Ref(0))
            true
        catch; false; end)
    end
end
