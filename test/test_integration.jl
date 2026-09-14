# End-to-end pipeline, review section 18. examples/main.jl cannot be reused directly: it needs
# CairoMakie (not a dependency) and its first line passes `embedding_dim` to RickerModel, which has
# no such field. This replicates the same call sequence headlessly at test scale.

@testset "integration" begin

    N = 40
    model = NormalModel(mu = 0.5, sigma = 1.5)
    truth = solve_model(model, Float64(N); rng = StableRNG(2024))

    resamplers = (
        "RademacherSplit"      => RademacherSplit(),
        "ContiguousBlockSplit" => ContiguousBlockSplit(timeseries_block_size = 12),
        "NoResampling"         => NoResampling(),
        "StandardBootstrap"    => StandardBootstrap(),
        "MovingBlockBootstrap" => MovingBlockBootstrap(block_size = 12),
    )

    methods = ("GSL" => GSL(), "BSL" => BSL(5))

    @testset "$mname / $rname / $sname" for (mname, method) in methods,
                                            (rname, resampler) in resamplers,
                                            (sname, make_stats) in (
            "single"  => () -> (StandardECDF(5),),
            "joint"   => () -> (StandardECDF(5), CumulativeSum(4)),
        )

        Random.seed!(4321)
        opts = MethodsOptions(N_obs = N, verbose = false, training_resamplings = 40,
                              bins_resamplings = 10, resampling_type = resampler,
                              inference_method = method)

        target, tsum = TargetData(truth, JointSummaryStatistics(make_stats()...), opts;
                                  priors = (mu = Normal(0.0, 5.0), sigma = Uniform(0.01, 10.0)),
                                  loss = LogLikelihood())

        @test target.summary_length > 0
        @test all(isfinite, target.obs_mean)
        @test size(tsum, 1) == target.summary_length
        @test target.obs_mean ≈ vec(mean(tsum, dims = 2))

        mcmc_opts = MCMCOptions(nsteps = 20, mcmc_algorithm = AM(proposal_width = 0.05),
                                initial_params = [0.5, 1.5], loss_function = LogLikelihood(),
                                likelihood_noise_scale = 0.0)

        results, state = mcmcrun(target, model, mcmc_opts)

        # mcmcrun catches every exception and returns a truncated chain, so without this the
        # whole matrix would pass green on a crash.
        @test results.current_iter[] == 20
        @test size(results.chain) == (2, 20)
        @test all(isfinite, results.chain)
        @test all(isfinite, results.sschain)
        @test length(state.current_params) == 2
    end

    @testset "DRAM completes across both inference methods" begin
        for method in (GSL(), BSL(5))
            Random.seed!(888)
            opts = MethodsOptions(N_obs = N, verbose = false, training_resamplings = 40,
                                  resampling_type = NoResampling(), inference_method = method)
            target, _ = TargetData(truth, JointSummaryStatistics(StandardECDF(5)), opts;
                                   loss = LogLikelihood())
            mcmc_opts = MCMCOptions(nsteps = 20,
                                    mcmc_algorithm = DRAM(proposal_width = 0.05, n_stages = 2,
                                                          proposal_scale = [1.0, 0.5]),
                                    initial_params = [0.5, 1.5], likelihood_noise_scale = 0.0)
            results, _ = mcmcrun(target, model, mcmc_opts)
            @test results.current_iter[] == 20
            @test all(isfinite, results.chain)
        end
    end

    @testset "derivative summaries survive the full pipeline" begin
        Random.seed!(77)
        opts = MethodsOptions(N_obs = N, verbose = false, training_resamplings = 40,
                              bins_resamplings = 10, resampling_type = RademacherSplit())
        target, _ = TargetData(truth,
                               JointSummaryStatistics(StandardECDF(5), StandardECDFDiff(5, 1, 1.0)),
                               opts; loss = LogLikelihood())

        @test target.summary_length == 10
        @test sort(target.buffers.index_cache) == collect(2:N-1)     # trimmed by the diff order

        mcmc_opts = MCMCOptions(nsteps = 20, mcmc_algorithm = AM(proposal_width = 0.05),
                                initial_params = [0.5, 1.5], likelihood_noise_scale = 0.0)
        results, _ = mcmcrun(target, model, mcmc_opts)
        @test results.current_iter[] == 20
        @test all(isfinite, results.chain)
    end

    @testset "n_summaries > 1 averages several resampled summaries" begin
        Random.seed!(555)
        opts = MethodsOptions(N_obs = N, verbose = false, training_resamplings = 40,
                              resampling_type = RademacherSplit(), n_summaries = 4)
        target, _ = TargetData(truth, JointSummaryStatistics(StandardECDF(5)), opts;
                               loss = LogLikelihood())

        @test size(target.buffers.mcmc_buffer, 2) == 4

        mcmc_opts = MCMCOptions(nsteps = 20, mcmc_algorithm = AM(proposal_width = 0.05),
                                initial_params = [0.5, 1.5], likelihood_noise_scale = 0.0)
        results, _ = mcmcrun(target, model, mcmc_opts)
        @test results.current_iter[] == 20
    end

    @testset "n_loss_evals > 1 averages several simulations" begin
        Random.seed!(556)
        opts = MethodsOptions(N_obs = N, verbose = false, training_resamplings = 40,
                              resampling_type = NoResampling(), n_loss_evals = 3)
        target, _ = TargetData(truth, JointSummaryStatistics(StandardECDF(5)), opts;
                               loss = LogLikelihood())
        mcmc_opts = MCMCOptions(nsteps = 20, mcmc_algorithm = AM(proposal_width = 0.05),
                                initial_params = [0.5, 1.5], likelihood_noise_scale = 0.0)
        results, _ = mcmcrun(target, model, mcmc_opts)
        @test results.current_iter[] == 20
    end

    @testset "the chain moves toward the truth on a well-specified model" begin
        # A long enough run on data simulated from the same model should put the posterior mean
        # nearer the true parameters than a deliberately displaced starting point.
        Random.seed!(20259)
        opts = MethodsOptions(N_obs = 100, verbose = false, training_resamplings = 200,
                              resampling_type = RademacherSplit())
        data = solve_model(NormalModel(mu = 2.0, sigma = 1.0), 100.0; rng = StableRNG(1))
        target, _ = TargetData(data, JointSummaryStatistics(StandardECDF(8)), opts;
                               priors = (mu = Uniform(-5.0, 10.0), sigma = Uniform(0.05, 5.0)),
                               loss = LogLikelihood())

        start = [5.0, 3.0]
        mcmc_opts = MCMCOptions(nsteps = 3_000, mcmc_algorithm = AM(proposal_width = 0.1,
                                                                    adaptation_interval = 50),
                                initial_params = start, likelihood_noise_scale = 0.0,
                                update_interval = 10_000_000)
        results, _ = mcmcrun(target, model, mcmc_opts)

        @test results.current_iter[] == 3_000
        posterior_mean = vec(mean(results.chain[:, 1_500:end], dims = 2))
        truth_params = [2.0, 1.0]
        @test norm(posterior_mean - truth_params) < norm(start - truth_params)
    end
end
