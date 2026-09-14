# src/mcmc/loss_functions.jl and the dispatch in src/mcmc/loss.jl

@testset "loss functions" begin

    data = wiggly(40)

    @testset "LogLikelihood" begin
        @testset "scaling" begin
            @test LogLikelihood().scaling_parameter == 1.0
            @test LogLikelihood().inverse_scaling == 1.0
            @test LogLikelihood(2.0).inverse_scaling == 0.5
            @test LogLikelihood(4.0).inverse_scaling == 0.25
        end

        @testset "three-argument form is minus half the Mahalanobis distance" begin
            C = regularized_cholesky(cov(randn(StableRNG(51), 4, 30)'))
            x = [1.0, 2.0, 3.0, 4.0]
            xs = [0.5, 2.5, 2.0, 5.0]
            d = x - xs

            @test LogLikelihood()(x, xs, C) ≈ -0.5 * dot(d, C \ d)
            @test LogLikelihood()(x, x, C) == 0.0                 # zero at the target
            @test LogLikelihood()(x, xs, C) ≈ LogLikelihood()(xs, x, C)   # symmetric
            @test LogLikelihood()(x, xs, C) < 0.0
        end

        @testset "two-argument form reads the covariance off the target" begin
            target, _ = make_target(data, StandardECDF(5); training_resamplings = 100)
            sim = target.obs_mean .+ 0.01

            expected = -0.5 * dot(sim - target.obs_mean, target.cov_factorization \ (sim - target.obs_mean))
            @test LogLikelihood()(target, sim) ≈ expected
            @test LogLikelihood(2.0)(target, sim) ≈ expected / 2      # inverse_scaling applied
            @test LogLikelihood()(target, copy(target.obs_mean)) == 0.0
        end
    end

    @testset "RobustChamfer" begin
        @test RobustChamfer(2.0).inverse_scaling == 0.5

        target, _ = make_target(data, ChamferDistance(1); training_resamplings = 50)
        sim = [0.5]
        @test RobustChamfer()(target, sim) ≈ -0.5
        @test RobustChamfer(2.0)(target, sim) ≈ -0.25

        @testset "three-argument form weights by the inverse standard deviations" begin
            C = regularized_cholesky(cov(randn(StableRNG(52), 3, 20)'))
            x = [1.0, 2.0, 3.0]
            @test RobustChamfer()(x, zeros(3), C) ≈ -sum(x .* sqrt.(diag(inv(C))))
        end
    end

    @testset "calculate_loss" begin
        model = NormalModel(mu = 0.0, sigma = 1.0)
        # likelihood_noise_scale defaults to NaN as a sentinel that mcmcrun rewrites to 0.0. Calling
        # calculate_loss directly skips that normalization, so it must be set here or every loss
        # comes back NaN (see test_known_issues.jl).
        opts = MCMCOptions(mcmc_algorithm = AM(), nsteps = 10, loss_function = LogLikelihood(),
                           likelihood_noise_scale = 0.0)

        @testset "returns a finite log-density for a sane proposal" begin
            target, _ = make_target(data, StandardECDF(5); training_resamplings = 100)
            loss = calculate_loss([0.0, 1.0], target, model, opts; rng_seed = UInt64(7))
            @test isfinite(loss)
        end

        @testset "an out-of-support proposal short-circuits to -Inf without simulating" begin
            # The prior is evaluated before the model is ever touched, so passing a model that
            # cannot possibly simulate still returns cleanly. That is the proof no simulation ran.
            target, _ = make_target(data, StandardECDF(5);
                                    priors = (mu = Uniform(0.0, 1.0), sigma = Uniform(0.0, 2.0)),
                                    training_resamplings = 50)
            @test calculate_loss([5.0, 1.0], target, nothing, opts) == -Inf
        end

        @testset "the log-prior is added exactly once" begin
            priors = (mu = Normal(0.0, 1.0), sigma = Normal(1.0, 1.0))
            params = [0.2, 1.1]

            # NoResampling makes the likelihood fully determined by rng_seed: with a
            # length-changing resampler the per-step resampling draws from the global RNG and the
            # two evaluations would not share a likelihood term.
            flat, _ = make_target(data, StandardECDF(5);
                                  resampling_type = NoResampling(), training_resamplings = 20)
            informative = with_priors(flat, priors)
            # Same target, same seed, so the likelihood term is bit-identical and the difference
            # must be exactly the log-prior -- not twice it, which was the review's critical
            # "initial state counts the prior twice" bug.
            seed = UInt64(123)
            diff = calculate_loss(params, informative, model, opts; rng_seed = seed) -
                   calculate_loss(params, flat, model, opts; rng_seed = seed)
            @test diff ≈ evaluate_log_prior(params, priors)
        end

        @testset "GSL with NoResampling is deterministic given a seed" begin
            # n_summaries == 1 with a LengthPreservingSampler takes the fast path, which skips
            # resampling entirely -- so the only randomness left is the seeded simulation.
            target, _ = make_target(data, StandardECDF(5);
                                    resampling_type = NoResampling(), training_resamplings = 20)
            seed = UInt64(99)
            a = calculate_loss([0.0, 1.0], target, model, opts; rng_seed = seed)
            b = calculate_loss([0.0, 1.0], target, model, opts; rng_seed = seed)
            @test a == b
            @test a != calculate_loss([0.0, 1.0], target, model, opts; rng_seed = UInt64(100))
        end

        @testset "a length-changing resampler adds unseeded randomness" begin
            # RademacherSplit resamples through the global RNG, which rng_seed does not control.
            target, _ = make_target(data, StandardECDF(5);
                                    resampling_type = RademacherSplit(), training_resamplings = 50)
            seed = UInt64(99)
            vals = [calculate_loss([0.0, 1.0], target, model, opts; rng_seed = seed) for _ in 1:5]
            @test length(unique(vals)) > 1
        end

        @testset "BSL estimates its covariance from the simulation spread" begin
            target, _ = make_target(data, StandardECDF(5);
                                    resampling_type = NoResampling(),
                                    inference_method = BSL(20), training_resamplings = 20)
            loss = calculate_loss([0.0, 1.0], target, model, opts; rng_seed = UInt64(5))
            @test isfinite(loss)
            # The per-step covariance is written into the buffer, leaving the target's own
            # placeholder factorization untouched.
            @test Matrix(target.cov_factorization) ≈ Matrix(I, target.summary_length, target.summary_length)
            @test size(target.buffers.bsl_buffer) == (target.summary_length, 20)
        end

        @testset "likelihood noise widens the loss" begin
            target, _ = make_target(data, StandardECDF(5);
                                    resampling_type = NoResampling(), training_resamplings = 20)
            noisy = MCMCOptions(mcmc_algorithm = AM(), nsteps = 10,
                                loss_function = LogLikelihood(), likelihood_noise_scale = 1.0)
            seed = UInt64(11)
            vals = [calculate_loss([0.0, 1.0], target, model, noisy; rng_seed = seed) for _ in 1:5]
            @test length(unique(vals)) > 1        # noise uses the global RNG, not rng_seed
        end
    end
end
