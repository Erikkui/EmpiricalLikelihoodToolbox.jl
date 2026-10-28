# src/inference/ABC.jl -- kernels, the kernel-average loss, distance and ε resolution in TargetData,
# the prior-predictive pilot, option checks, show strings and a short ABC-MCMC run.

using StatsBase: mad

const ELT = EmpiricalLikelihoodToolbox

# Plain Euclidean distance as a user-supplied callable (Distances.jl is not a test dependency).
euclidean_distance(sim, obs) = sqrt(sum(abs2, sim .- obs))

# A target over NormalModel data, built through TargetData with an explicit rng for the pilot.
function abc_target(method; data = solve_model(NormalModel(mu = 1.0, sigma = 2.0), 100.0;
                                               rng = StableRNG(3)),
                    priors = (mu = Uniform(-5.0, 5.0), sigma = Uniform(0.1, 5.0)),
                    model = NormalModel(mu = 1.0, sigma = 2.0), rng = StableRNG(11),
                    stat = StandardECDF(8), kwargs...)
    opts = make_options(; N_obs = size(data, 2), inference_method = method, kwargs...)
    return TargetData(data, JointSummaryStatistics(stat), opts, model; priors = priors, rng = rng)
end

@testset "ABC" begin

    @testset "log_kernel" begin
        @test ELT.log_kernel(UniformKernel(), 0.5, 1.0) == 0.0
        @test ELT.log_kernel(UniformKernel(), 1.0, 1.0) == 0.0
        @test ELT.log_kernel(UniformKernel(), 1.5, 1.0) == -Inf
        @test ELT.log_kernel(GaussianKernel(), 2.0, 4.0) ≈ -0.5 * 0.25
        @test ELT.log_kernel(GaussianKernel(), 0.0, 4.0) == 0.0
        @test ELT.log_kernel(EpanechnikovKernel(), 0.5, 1.0) ≈ log(0.75)
        @test ELT.log_kernel(EpanechnikovKernel(), 1.0, 1.0) == -Inf
        @test ELT.log_kernel(EpanechnikovKernel(), 2.0, 1.0) == -Inf
        # A failed simulation has distance Inf: zero kernel value for every kernel.
        for kernel in (UniformKernel(), GaussianKernel(), EpanechnikovKernel())
            @test ELT.log_kernel(kernel, Inf, 1.0) == -Inf
        end
    end

    @testset "constructors and show strings" begin
        @test isnan(ABC().epsilon)
        @test ABC(0.5).epsilon == 0.5
        @test ABC(1; n_sim = 3).n_sim == 3
        @test ABC().distance === :scaled
        @test ABC().kernel isa UniformKernel
        @test isnothing(ABC().adapt_epsilon)

        @test string(ABC()) == "ABC(ε=pilot q=0.05)"
        @test string(ABC(epsilon_quantile = 0.01)) == "ABC(ε=pilot q=0.01)"
        @test string(ABC(0.12)) == "ABC(ε=0.12)"
        @test string(ABC(0.123456)) == "ABC(ε=0.1235)"
        @test string(ABC(0.12; kernel = GaussianKernel())) == "ABC(ε=0.12, GaussianKernel)"
        @test string(ABC(0.12; n_sim = 5)) == "ABC(ε=0.12, n_sim=5)"
    end

    @testset "allocate_method_buffer holds n_sim distances" begin
        @test size(ELT.allocate_method_buffer(ABC(n_sim = 4), 6)) == (4,)
    end

    @testset "check_method_options" begin
        data = wiggly(40)
        @test_throws ArgumentError make_target(data, StandardECDF(5); standardize = true,
                                               inference_method = ABC(0.1))
        @test_throws ArgumentError make_target(data, StandardECDF(5); resampling_type = NoResampling(),
                                               inference_method = ABC(0.1; distance = :mahalanobis))
        @test_throws ArgumentError make_target(data, StandardECDF(5); inference_method = ABC(0.1; n_sim = 0))
        @test_throws ArgumentError make_target(data, StandardECDF(5); inference_method = ABC(-0.1))
        @test_throws ArgumentError make_target(data, StandardECDF(5); inference_method = ABC(0.0))
        @test_throws ArgumentError make_target(data, StandardECDF(5);
                                               inference_method = ABC(epsilon_quantile = 1.0))
        @test_throws ArgumentError make_target(data, StandardECDF(5);
                                               inference_method = ABC(0.1; distance = :bogus))
    end

    @testset "calculate_loss with the uniform kernel" begin
        model = NormalModel(mu = 1.0, sigma = 2.0)
        opts = MCMCOptions(mcmc_algorithm = AM(), nsteps = 5)
        for (epsilon, expected) in ((1e10, 0.0), (1e-12, -Inf))
            target, _ = abc_target(ABC(epsilon; distance = euclidean_distance);
                                   resampling_type = NoResampling())
            method = target.options.inference_method
            @test ELT.calculate_loss(method, target, model, opts, StableRNG(1)) == expected
        end
    end

    @testset "n_sim > 1 averages kernels of the stored distances" begin
        model = NormalModel(mu = 0.5, sigma = 1.5)
        opts = MCMCOptions(mcmc_algorithm = AM(), nsteps = 5)
        epsilon = 0.3
        target, _ = abc_target(ABC(epsilon; distance = euclidean_distance, kernel = GaussianKernel(),
                                   n_sim = 6); resampling_type = NoResampling())
        method = target.options.inference_method
        loss = ELT.calculate_loss(method, target, model, opts, StableRNG(2))

        distances = copy(target.buffers.method_buffer)
        @test length(distances) == 6
        @test all(>(0), distances)
        @test length(unique(distances)) == 6             # six independent simulations
        @test loss ≈ log(mean(exp.(-0.5 .* (distances ./ epsilon) .^ 2)))

        # Each stored distance is what simulate_distance gives for the same random stream.
        rng = StableRNG(2)
        replayed = [simulate_distance(target, model, rng) for _ in 1:6]
        @test replayed ≈ distances
    end

    @testset "a failed simulation gives distance Inf" begin
        target, _ = abc_target(ABC(1e10; distance = euclidean_distance); resampling_type = NoResampling())
        broken_model = update_model_parameters(target.model, [0.0, NaN])
        @test simulate_distance(target, broken_model, StableRNG(1)) == Inf
        opts = MCMCOptions(mcmc_algorithm = AM(), nsteps = 5)
        method = target.options.inference_method
        @test ELT.calculate_loss(method, target, broken_model, opts, StableRNG(1)) == -Inf
    end

    @testset "distance resolution" begin
        @testset ":scaled with a resampler uses the training sd" begin
            target, tsum = abc_target(ABC(0.5); training_resamplings = 100)
            distance = target.options.inference_method.distance
            @test distance isa ELT.WeightedEuclidean
            # target.data.options is kept in sync
            @test target.data.options.inference_method === target.options.inference_method

            variances = vec(var(tsum, dims = 2))
            positive = variances .> 0
            @test any(positive)
            @test distance.weights[positive] ≈ 1 ./ variances[positive]
            # Constant components fall back to the smallest positive scale.
            @test all(distance.weights[.!positive] .≈ 1 / minimum(variances[positive]))
            @test target.options.inference_method.epsilon == 0.5
        end

        @testset ":scaled with NoResampling uses pilot MADs; ε from the pilot quantile" begin
            method = ABC(n_pilot = 200, epsilon_quantile = 0.1)
            target, _ = abc_target(method; resampling_type = NoResampling(), rng = StableRNG(21))
            resolved = target.options.inference_method

            # Replaying the pilot with the same rng reproduces TargetData's pilot.
            pilot = pilot_simulations(target, 200; rng = StableRNG(21))
            @test size(pilot.params) == (2, 200)
            @test size(pilot.summaries) == (target.summary_length, 200)
            @test length(pilot.distances) == 200
            @test all(-5.0 .<= pilot.params[1, :] .<= 5.0)
            @test all(0.1 .<= pilot.params[2, :] .<= 5.0)

            scales = [mad(row; normalize = true) for row in eachrow(pilot.summaries)]
            positive = scales .> 0
            @test resolved.distance.weights[positive] ≈ 1 ./ scales[positive] .^ 2
            @test all(resolved.distance.weights[.!positive] .≈ 1 / minimum(scales[positive])^2)

            finite_distances = filter(isfinite, pilot.distances)
            @test resolved.epsilon ≈ quantile(finite_distances, 0.1)
            @test resolved.epsilon > 0
            @test string(resolved) == "ABC(ε=$(round(resolved.epsilon, sigdigits = 4)))"
        end

        @testset ":mahalanobis uses the inverse regularized training covariance" begin
            reg(C) = C + 1e-2 * Diagonal(diag(C)) + 1e-6 * I
            target, tsum = abc_target(ABC(0.5; distance = :mahalanobis); training_resamplings = 100)
            distance = target.options.inference_method.distance
            @test distance isa ELT.Mahalanobis
            @test distance.qmat ≈ inv(reg(cov(tsum')))
        end

        @testset "a callable distance passes through" begin
            target, _ = abc_target(ABC(0.5; distance = euclidean_distance))
            @test target.options.inference_method.distance === euclidean_distance
        end
    end

    @testset "pilot needs proper priors" begin
        # ε = NaN needs the pilot, which cannot sample flat priors.
        @test_throws ArgumentError abc_target(ABC(); priors = nothing)
        @test_throws ArgumentError abc_target(ABC(); priors = (mu = Normal(), sigma = nothing))
        @test_throws ArgumentError abc_target(ABC(); priors = (mu = Normal(),))
        # :scaled under NoResampling needs the pilot too, even with ε given.
        @test_throws ArgumentError abc_target(ABC(0.5); priors = nothing, resampling_type = NoResampling())

        # With a fixed ε and a resampler, no pilot runs and no priors are needed.
        target, _ = abc_target(ABC(0.5); priors = nothing)
        @test_throws ArgumentError pilot_simulations(target, 10)
        @test_throws ArgumentError pilot_simulations(with_priors(target, (mu = Normal(), sigma = nothing)), 10)
        pilot = pilot_simulations(with_priors(target, (mu = Normal(), sigma = Uniform(0.5, 1.0))), 10)
        @test all(isfinite, pilot.distances)
    end

    @testset "pilot_simulations under GSL has no distances" begin
        target, _ = make_target(wiggly(40), StandardECDF(5);
                                priors = (mu = Normal(), sigma = Uniform(0.5, 1.0)))
        pilot = pilot_simulations(target, 5; rng = StableRNG(4))
        @test all(isnan, pilot.distances)
        @test all(isfinite, pilot.summaries)
    end

    @testset "loss functions are rejected under ABC" begin
        data = wiggly(40)
        opts = make_options(; N_obs = 40, inference_method = ABC(0.5))
        @test_throws ArgumentError TargetData(data, JointSummaryStatistics(StandardECDF(5)), opts,
                                              LogLikelihood(), NormalModel())

        target, _ = make_target(data, StandardECDF(5); inference_method = ABC(0.5; distance = euclidean_distance))
        with_loss = MCMCOptions(nsteps = 5, mcmc_algorithm = AM(), loss_function = LogLikelihood(),
                                initial_params = [0.0, 1.0])
        @test_throws ArgumentError mcmcrun(target, with_loss)
    end

    @testset "ABC-MCMC on NormalModel stays near the truth" begin
        truth = [1.0, 2.0]
        method = ABC(n_pilot = 300, epsilon_quantile = 0.05)
        target, _ = abc_target(method; resampling_type = NoResampling(),
                               data = solve_model(NormalModel(mu = 1.0, sigma = 2.0), 300.0;
                                                  rng = StableRNG(5)),
                               stat = StandardECDF(10), rng = StableRNG(6))
        opts = MCMCOptions(nsteps = 3_000, mcmc_algorithm = AM(proposal_width = 0.05),
                           initial_params = truth, update_interval = 0)

        Random.seed!(42)
        results, _ = mcmcrun(target, opts)
        @test results.current_iter[] == opts.nsteps
        @test results.stuck_kicks[] == 0
        @test 0.05 < results.acceptance[1] < 0.95
        @test all(isfinite, results.sschain)

        chain = results.chain[:, 1_000:end]
        @test vec(mean(chain, dims = 2)) ≈ truth atol = 0.6
    end
end
