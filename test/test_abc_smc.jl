# src/abc/ -- ABC rejection sampling and replenishment ABC-SMC on NormalModel (known truth).

# An ABC target over NormalModel data with a fixed ε (which skips the ε pilot; the :scaled MAD pilot
# still runs under NoResampling).
function smc_target(method; stat = StandardECDF(10))
    data = solve_model(NormalModel(mu = 1.0, sigma = 2.0), 100.0; rng = StableRNG(3))
    priors = (mu = Uniform(-5.0, 5.0), sigma = Uniform(0.1, 5.0))
    opts = make_options(; N_obs = size(data, 2), inference_method = method,
                        resampling_type = NoResampling())
    target, _ = TargetData(data, JointSummaryStatistics(stat), opts,
                           NormalModel(mu = 1.0, sigma = 2.0); priors = priors, rng = StableRNG(11))
    return target
end

@testset "ABC rejection and ABC-SMC" begin

    target = smc_target(ABC(1.0; n_pilot = 300))

    @testset "abc_rejection" begin
        result = abc_rejection(target, 400; n_keep = 40, rng = StableRNG(1))
        @test size(result.params) == (2, 40)
        @test length(result.distances) == 40
        @test all(result.distances .<= result.epsilon)
        @test issorted(result.distances)
        @test result.acceptance_rate == 40 / 400

        by_epsilon = abc_rejection(target, 400; epsilon = result.epsilon, rng = StableRNG(1))
        @test size(by_epsilon.params, 2) >= 40      # ties at ε can only add draws
        @test all(by_epsilon.distances .<= result.epsilon)

        default_epsilon = abc_rejection(target, 100; rng = StableRNG(2))
        @test default_epsilon.epsilon == target.options.inference_method.epsilon

        @test_throws ArgumentError abc_rejection(target, 100; epsilon = 1.0, n_keep = 10)
        @test_throws ArgumentError abc_rejection(target, 100; n_keep = 101)
        gsl_target, _ = make_target(wiggly(40), StandardECDF(5);
                                    priors = (mu = Uniform(-5.0, 5.0), sigma = Uniform(0.1, 5.0)))
        @test_throws ArgumentError abc_rejection(gsl_target, 100; n_keep = 10)
    end

    @testset "abc_smc" begin
        smc_options = ABCSMCOptions(n_particles = 200, max_generations = 5)
        result = abc_smc(target, smc_options; rng = StableRNG(5))

        n_generations = length(result.epsilons)
        @test 1 <= n_generations <= 5
        @test size(result.particles) == (2, 200)
        @test length(result.distances) == 200
        @test length(result.acceptance_rates) == n_generations
        @test length(result.mcmc_repeats) == n_generations
        @test all(result.mcmc_repeats .>= 1)
        @test result.n_simulations > 200
        @test issorted(result.epsilons; rev = true)
        @test all(result.distances .<= result.epsilons[end])

        posterior_mean = vec(mean(result.particles, dims = 2))
        @test abs(posterior_mean[1] - 1.0) < 0.75
        @test abs(posterior_mean[2] - 2.0) < 0.75

        # Same seeded rng, same result (NoResampling draws nothing from the global RNG)
        repeated = abc_smc(target, smc_options; rng = StableRNG(5))
        @test repeated.particles == result.particles
        @test repeated.epsilons == result.epsilons

        # epsilon_target: the last generation runs at exactly the target tolerance
        epsilon_target = result.epsilons[2]
        to_target = abc_smc(target, ABCSMCOptions(n_particles = 200, epsilon_target = epsilon_target);
                            rng = StableRNG(5))
        @test to_target.epsilons[end] == epsilon_target
        @test all(to_target.distances .<= epsilon_target)
    end

    @testset "abc_smc input checks" begin
        gaussian_target = smc_target(ABC(1.0; n_pilot = 300, kernel = GaussianKernel()))
        @test_throws ArgumentError abc_smc(gaussian_target, ABCSMCOptions(n_particles = 50))
        multi_sim_target = smc_target(ABC(1.0; n_pilot = 300, n_sim = 2))
        @test_throws ArgumentError abc_smc(multi_sim_target, ABCSMCOptions(n_particles = 50))
        @test_throws ArgumentError abc_smc(target, ABCSMCOptions(n_particles = 50, drop_fraction = 1.0))
        @test_throws ArgumentError abc_smc(target, ABCSMCOptions(n_particles = 3, drop_fraction = 0.5))
    end

    @testset "R_t = ceil(log c / log(1 - p_acc)), clamped" begin
        @test EmpiricalLikelihoodToolbox._mcmc_repeats(0.5, 0.01, 500) == ceil(Int, log(0.01) / log(0.5))
        @test EmpiricalLikelihoodToolbox._mcmc_repeats(0.0, 0.01, 500) == 500
        @test EmpiricalLikelihoodToolbox._mcmc_repeats(1.0, 0.01, 500) == 1
        @test EmpiricalLikelihoodToolbox._mcmc_repeats(1e-6, 0.01, 50) == 50
    end
end
