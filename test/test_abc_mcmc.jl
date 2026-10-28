# ABC-MCMC extras -- the distance chain of AM/DRAM runs, the Vihola-Franks ε adaptation and the
# post-correction to a smaller tolerance (abc_post_correct).

const ELT_MCMC = EmpiricalLikelihoodToolbox

# A NormalModel whose simulator fails after FAIL_AFTER_CALLS calls, for the partial-chain path.
Base.@kwdef struct FailingNormalModel <: AbstractSimulationModel
    mu::Float64 = 1.0
    sigma::Float64 = 2.0
    dt_obs::Float64 = 1.0
    all_parameters::Tuple{Vararg{Symbol}} = (:mu, :sigma)
    active_parameters::Tuple{Vararg{Symbol}} = (:mu, :sigma)
end

const FAIL_AFTER_CALLS = Ref(typemax(Int))
const FAILING_MODEL_CALLS = Ref(0)

function EmpiricalLikelihoodToolbox.solve_model(model::FailingNormalModel, t_end::Float64; rng)
    FAILING_MODEL_CALLS[] += 1
    FAILING_MODEL_CALLS[] > FAIL_AFTER_CALLS[] && error("simulated mid-chain failure")
    normal_model = NormalModel(mu = model.mu, sigma = model.sigma)
    return solve_model(normal_model, t_end; rng = rng)
end

# ABC target over NormalModel data: NoResampling, :scaled distance from pilot MADs, ε from the pilot
# quantile unless the method sets it.
function abc_mcmc_target(method; model = NormalModel(mu = 1.0, sigma = 2.0))
    data = solve_model(NormalModel(mu = 1.0, sigma = 2.0), 100.0; rng = StableRNG(3))
    priors = (mu = Uniform(-5.0, 5.0), sigma = Uniform(0.1, 5.0))
    opts = make_options(; N_obs = size(data, 2), inference_method = method,
                        resampling_type = NoResampling())
    target, _ = TargetData(data, JointSummaryStatistics(StandardECDF(8)), opts, model;
                           priors = priors, rng = StableRNG(11))
    return target
end

# Every stored log-target equals log prior + log mean K_ε(ρ) of the stored distances, at the stored ε.
function log_targets_match_distances(results, target)
    kernel = target.options.inference_method.kernel
    active_names = target.model.active_parameters
    for ii in 1:results.current_iter[]
        params = results.chain[:, ii]
        logprior = evaluate_log_prior(params, target.priors, active_names)
        distances = results.distance_chain[:, ii]
        log_likelihood = ELT_MCMC._log_kernel_average(kernel, distances, results.epsilon_chain[ii])
        expected = isfinite(log_likelihood) ? log_likelihood + logprior : -Inf
        expected ≈ results.sschain[ii] || return false
    end
    return true
end

# Number of steps in which the chain moved.
moves(chain) = count(jj -> chain[:, jj] != chain[:, jj - 1], 2:size(chain, 2))

@testset "ABC-MCMC extras" begin

    @testset "AM stores the current state's distances" begin
        method = ABC(n_pilot = 200, epsilon_quantile = 0.1)
        target = abc_mcmc_target(method)
        epsilon = target.options.inference_method.epsilon
        opts = MCMCOptions(nsteps = 400, mcmc_algorithm = AM(proposal_width = 0.05),
                           initial_params = [1.0, 2.0], update_interval = 0)
        Random.seed!(1)
        results, _ = mcmcrun(target, opts)

        @test size(results.distance_chain) == (1, opts.nsteps)
        @test all(==(epsilon), results.epsilon_chain)
        @test results.epsilon == epsilon
        @test log_targets_match_distances(results, target)

        # Uniform kernel: after the first acceptance, every stored state is within ε.
        first_finite = findfirst(isfinite, results.sschain)
        @test !isnothing(first_finite)
        @test all(results.distance_chain[1, first_finite:end] .<= epsilon)
    end

    @testset "n_sim > 1, re-evaluation and stuck kicks keep the distances in sync" begin
        method = ABC(n_pilot = 200, epsilon_quantile = 0.1, kernel = GaussianKernel(), n_sim = 3)
        target = abc_mcmc_target(method)
        for discard_noisy_updates in (false, true)
            opts = MCMCOptions(nsteps = 300, mcmc_algorithm = AM(proposal_width = 0.05),
                               initial_params = [1.0, 2.0], update_interval = 3,
                               reevaluate_current_loss = true,
                               discard_noisy_updates = discard_noisy_updates)
            Random.seed!(2)
            results, _ = mcmcrun(target, opts)
            @test size(results.distance_chain) == (3, opts.nsteps)
            @test all(isfinite, results.distance_chain)
            @test log_targets_match_distances(results, target)
        end
    end

    @testset "GSL and BSL have no distance chain and a NaN ε" begin
        for method in (GSL(), BSL(5))
            target, _ = make_target(wiggly(40), StandardECDF(5); inference_method = method)
            opts = MCMCOptions(nsteps = 20, mcmc_algorithm = AM(), initial_params = [0.0, 1.0])
            results, _ = mcmcrun(target, opts)
            @test size(results.distance_chain) == (0, opts.nsteps)
            @test length(results.epsilon_chain) == opts.nsteps
            @test all(isnan, results.epsilon_chain)
            @test isnan(results.epsilon)
            @test_throws ArgumentError abc_post_correct(results, target, 0.1)
        end
    end

    @testset "EpsilonAdaptation" begin
        adaptation = EpsilonAdaptation(n_adapt = 10)
        @test adaptation.target_acceptance == 0.1
        @test adaptation.step_exponent ≈ 2 / 3
        @test_throws UndefKeywordError EpsilonAdaptation()
        @test_throws ArgumentError EpsilonAdaptation(n_adapt = 0)
        @test_throws ArgumentError EpsilonAdaptation(n_adapt = 10, target_acceptance = 1.0)
        @test_throws ArgumentError EpsilonAdaptation(n_adapt = 10, step_exponent = 0.5)
        @test_throws ArgumentError abc_mcmc_target(ABC(0.5; adapt_epsilon = 10))
    end

    @testset "AM adapts ε during the first n_adapt steps, then freezes it" begin
        n_adapt = 1_000
        method = ABC(n_pilot = 200, adapt_epsilon = EpsilonAdaptation(n_adapt = n_adapt))
        target = abc_mcmc_target(method)
        initial_epsilon = target.options.inference_method.epsilon
        opts = MCMCOptions(nsteps = 3_000, mcmc_algorithm = AM(proposal_width = 0.05),
                           initial_params = [1.0, 2.0], update_interval = 0)
        Random.seed!(3)
        results, _ = mcmcrun(target, opts)

        adapted = results.epsilon_chain[1:n_adapt]
        frozen = results.epsilon_chain[(n_adapt + 1):end]
        @test length(unique(adapted)) > 100
        @test all(==(results.epsilon), frozen)
        @test results.epsilon == adapted[end]
        @test results.epsilon != initial_epsilon
        @test target.options.inference_method.epsilon == initial_epsilon    # the caller's target is unchanged

        # The log-targets recomputed at each new ε match the stored distances.
        @test log_targets_match_distances(results, target)

        # Acceptance after adaptation is near the 0.1 target (loose).
        post_adaptation = results.chain[:, n_adapt:end]
        acceptance = moves(post_adaptation) / (size(post_adaptation, 2) - 1)
        @test 0.03 < acceptance < 0.3
    end

    @testset "DRAM with ABC" begin
        method = ABC(n_pilot = 200, epsilon_quantile = 0.1, kernel = GaussianKernel(), n_sim = 2)
        target = abc_mcmc_target(method)
        opts = MCMCOptions(nsteps = 300, mcmc_algorithm = DRAM(proposal_width = 0.05),
                           initial_params = [1.0, 2.0])
        Random.seed!(4)
        results, _ = mcmcrun(target, opts)
        @test size(results.distance_chain) == (2, opts.nsteps)
        @test all(isfinite, results.distance_chain)
        @test log_targets_match_distances(results, target)
        @test results.epsilon == target.options.inference_method.epsilon

        adaptive = abc_mcmc_target(ABC(n_pilot = 200, adapt_epsilon = EpsilonAdaptation(n_adapt = 10)))
        @test_throws ArgumentError mcmcrun(adaptive, opts)
    end

    @testset "abc_post_correct" begin
        method = ABC(n_pilot = 200, epsilon_quantile = 0.1)
        target = abc_mcmc_target(method)
        opts = MCMCOptions(nsteps = 600, mcmc_algorithm = AM(proposal_width = 0.05),
                           initial_params = [1.0, 2.0], update_interval = 0)
        Random.seed!(5)
        results, _ = mcmcrun(target, opts)
        epsilon = results.epsilon
        burnin = findfirst(isfinite, results.sschain) - 1
        kept_distances = results.distance_chain[1, (burnin + 1):end]
        n_kept = length(kept_distances)

        # δ = ε: equal weights.
        corrected = abc_post_correct(results, target, epsilon; burnin = burnin)
        @test length(corrected.weights) == n_kept
        @test all(corrected.weights .≈ 1 / n_kept)
        @test corrected.ess ≈ n_kept
        @test corrected.samples == results.chain[:, (burnin + 1):end]

        # δ < ε: the samples with ρ > δ get weight 0, the others equal weights.
        delta = median(kept_distances)
        corrected = abc_post_correct(results, target, delta; burnin = burnin)
        within = kept_distances .<= delta
        @test (corrected.weights .> 0) == within
        @test all(corrected.weights[within] .≈ 1 / count(within))
        @test sum(corrected.weights) ≈ 1.0
        @test corrected.ess ≈ count(within)

        @test_throws ArgumentError abc_post_correct(results, target, 1.01 * epsilon)
        @test_throws ArgumentError abc_post_correct(results, target, 0.0)
        @test_throws ArgumentError abc_post_correct(results, target, epsilon; burnin = opts.nsteps)
        # Every weight zero: no state within δ.
        @test_throws ArgumentError abc_post_correct(results, target, 0.5 * minimum(kept_distances);
                                                    burnin = burnin)

        # General kernel, n_sim > 1: weight ∝ mean K_δ(ρ) / mean K_ε(ρ).
        gaussian = abc_mcmc_target(ABC(n_pilot = 200, epsilon_quantile = 0.1,
                                       kernel = GaussianKernel(), n_sim = 2))
        Random.seed!(6)
        gaussian_results, _ = mcmcrun(gaussian, MCMCOptions(nsteps = 100, mcmc_algorithm = AM(),
                                                            initial_params = [1.0, 2.0],
                                                            update_interval = 0))
        gaussian_epsilon = gaussian_results.epsilon
        gaussian_delta = 0.5 * gaussian_epsilon
        kernel_mean(distances, tolerance) = mean(exp.(-0.5 .* (distances ./ tolerance) .^ 2))
        raw = [kernel_mean(column, gaussian_delta) / kernel_mean(column, gaussian_epsilon)
               for column in eachcol(gaussian_results.distance_chain)]
        corrected = abc_post_correct(gaussian_results, gaussian, gaussian_delta)
        @test corrected.weights ≈ raw ./ sum(raw)

        # With adaptation, the burn-in must cover the adaptation phase.
        adaptive = abc_mcmc_target(ABC(n_pilot = 200, adapt_epsilon = EpsilonAdaptation(n_adapt = 200)))
        Random.seed!(7)
        adaptive_results, _ = mcmcrun(adaptive, MCMCOptions(nsteps = 600, mcmc_algorithm = AM(proposal_width = 0.05),
                                                            initial_params = [1.0, 2.0],
                                                            update_interval = 0))
        @test_throws ArgumentError abc_post_correct(adaptive_results, adaptive, adaptive_results.epsilon;
                                                    burnin = 100)
        corrected = abc_post_correct(adaptive_results, adaptive, adaptive_results.epsilon; burnin = 200)
        @test sum(corrected.weights) ≈ 1.0
    end

    @testset "a mid-chain failure slices the distance chain" begin
        model = FailingNormalModel()
        method = ABC(n_pilot = 100, epsilon_quantile = 0.1)
        FAIL_AFTER_CALLS[] = typemax(Int)
        target = abc_mcmc_target(method; model = model)

        FAILING_MODEL_CALLS[] = 0
        FAIL_AFTER_CALLS[] = 50
        opts = MCMCOptions(nsteps = 200, mcmc_algorithm = AM(proposal_width = 0.05),
                           initial_params = [1.0, 2.0], update_interval = 0)
        results, _ = mcmcrun(target, opts)
        FAIL_AFTER_CALLS[] = typemax(Int)

        iters = results.current_iter[]
        @test 0 < iters < opts.nsteps
        @test size(results.chain) == (2, iters)
        @test size(results.distance_chain) == (1, iters)
        @test length(results.epsilon_chain) == iters
        @test results.epsilon == target.options.inference_method.epsilon
        @test log_targets_match_distances(results, target)
    end
end

# Number of ABC distance evaluations, ie. simulations (counted by counting_distance).
const DISTANCE_CALLS = Ref(0)

function counting_distance(simulated, observed)
    DISTANCE_CALLS[] += 1
    return sqrt(sum(abs2, simulated .- observed))
end

# ABC target with informative priors (so early rejection can skip proposals) and the counting
# distance. ε from the pilot quantile, then fixed (unless the method adapts it).
function early_rejection_target(; adapt_epsilon = nothing)
    data = solve_model(NormalModel(mu = 1.0, sigma = 2.0), 100.0; rng = StableRNG(3))
    priors = (mu = Normal(1.0, 0.3), sigma = truncated(Normal(2.0, 0.3), 0.1, 5.0))
    method = ABC(n_pilot = 200, epsilon_quantile = 0.2, distance = counting_distance,
                 adapt_epsilon = adapt_epsilon)
    opts = make_options(; N_obs = size(data, 2), inference_method = method,
                        resampling_type = NoResampling(), n_summaries = 1)
    target, _ = TargetData(data, JointSummaryStatistics(StandardECDF(8)), opts, NormalModel();
                           priors = priors, rng = StableRNG(11))
    return target
end

# AM run with early rejection on or off; returns the results and the number of simulations.
function run_counted(target, early_rejection; nsteps = 600)
    opts = MCMCOptions(nsteps = nsteps, mcmc_algorithm = AM(proposal_width = 0.05),
                       initial_params = [1.0, 2.0], update_interval = 0,
                       early_rejection = early_rejection)
    DISTANCE_CALLS[] = 0
    Random.seed!(7)
    results, _ = mcmcrun(target, opts)
    return results, DISTANCE_CALLS[]
end

same_chains(first, second) = isequal(first.chain, second.chain) &&
                             isequal(first.sschain, second.sschain) &&
                             isequal(first.distance_chain, second.distance_chain) &&
                             isequal(first.epsilon_chain, second.epsilon_chain)

@testset "Early rejection (AM)" begin
    @testset "ABC with fixed ε: same chain, fewer simulations" begin
        target = early_rejection_target()
        results_on, calls_on = run_counted(target, true)
        results_off, calls_off = run_counted(target, false)
        @test same_chains(results_on, results_off)
        @test calls_on < calls_off
        @test results_on.acceptance == results_off.acceptance
        @test log_targets_match_distances(results_on, target)
    end

    @testset "off during ε adaptation" begin
        target = early_rejection_target(adapt_epsilon = EpsilonAdaptation(n_adapt = 300))
        results_on, calls_on = run_counted(target, true; nsteps = 300)
        results_off, calls_off = run_counted(target, false; nsteps = 300)
        @test same_chains(results_on, results_off)
        @test calls_on == calls_off
    end

    # RademacherSplit shuffles the target's index cache with the global RNG, so each run builds
    # its own target after seeding.
    @testset "GSL: no bound, so nothing changes" begin
        chains = map((true, false)) do early_rejection
            Random.seed!(5)
            target, _ = make_target(wiggly(40), StandardECDF(5); inference_method = GSL())
            opts = MCMCOptions(nsteps = 100, mcmc_algorithm = AM(), initial_params = [0.0, 1.0],
                               early_rejection = early_rejection)
            results, _ = mcmcrun(target, opts)
            return results
        end
        @test same_chains(chains[1], chains[2])
    end
end
