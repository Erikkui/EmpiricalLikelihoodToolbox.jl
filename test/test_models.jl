# src/models/*.jl -- review section 22.7: test models analytically.

@testset "models" begin

    @testset "NegExpModel is exactly its closed form" begin
        # With noise_scale = 0 the model is deterministic: x(t) = theta1 * (1 - exp(-theta2*t)),
        # evaluated at t = (ii-1)*dt_sol. The sharpest model assertion available.
        theta1, theta2, dt = 2.0, 0.5, 0.1
        m = NegExpModel(theta1 = theta1, theta2 = theta2,
                        dt_obs = dt, dt_sol = dt, noise_scale = 0.0)
        traj = solve_model(m, 5.0)

        steps = round(Int, 5.0 / dt)
        t = (0:steps-1) .* dt
        @test size(traj) == (1, steps)
        @test traj[1, :] ≈ theta1 .* (1 .- exp.(-theta2 .* t)) atol = 1e-12

        @testset "deterministic regardless of rng" begin
            @test solve_model(m, 5.0; rng = StableRNG(1)) == solve_model(m, 5.0; rng = StableRNG(2))
        end

        @testset "noise_scale > 0 makes it stochastic" begin
            mn = NegExpModel(theta1 = theta1, theta2 = theta2,
                             dt_obs = dt, dt_sol = dt, noise_scale = 0.5)
            @test solve_model(mn, 5.0; rng = StableRNG(1)) != solve_model(mn, 5.0; rng = StableRNG(2))
        end
    end

    @testset "NormalModel draws iid samples" begin
        mu, sigma = 3.0, 2.0
        m = NormalModel(mu = mu, sigma = sigma)
        traj = solve_model(m, 20000.0; rng = StableRNG(42))

        @test size(traj) == (1, 20000)
        # Column 1 is the initial state, not a draw (see test_known_issues.jl), so it is excluded.
        samples = traj[1, 2:end]
        @test mean(samples) ≈ mu atol = 0.05
        @test std(samples) ≈ sigma atol = 0.05
    end

    @testset "OUModel relaxes to its stationary distribution" begin
        theta, mu, sigma, dt = 2.0, 1.0, 0.5, 0.01
        m = OUModel(theta = theta, mu = mu, sigma = sigma, dt_obs = dt, dt_sol = dt)
        traj = solve_model(m, 2000.0; rng = StableRNG(7))

        # Discard a burn-in of several relaxation times (1/theta) before measuring.
        samples = traj[1, 10_000:end]
        @test mean(samples) ≈ mu atol = 0.05
        @test var(samples) ≈ sigma^2 / (2 * theta) rtol = 0.15
    end

    @testset "Lorenz63Model stays on the attractor" begin
        m = Lorenz63Model()
        traj = solve_model(m, 100.0; rng = StableRNG(3))

        @test size(traj, 1) == 3
        @test all(isfinite, traj)
        @test maximum(abs, traj) < 200.0        # the attractor is bounded
    end

    @testset "RickerModel" begin
        m = RickerModel()

        @testset "reproducible under a fixed rng" begin
            @test solve_model(m, 50.0; rng = StableRNG(5)) == solve_model(m, 50.0; rng = StableRNG(5))
            @test solve_model(m, 50.0; rng = StableRNG(5)) != solve_model(m, 50.0; rng = StableRNG(6))
        end

        @testset "shape and support" begin
            traj = solve_model(m, 50.0; rng = StableRNG(5))
            @test size(traj) == (1, 50)
            @test all(≥(0), traj)               # Poisson counts, log1p-transformed
            @test all(isfinite, traj)
        end

        @testset "hidden states can be returned alongside the trajectory" begin
            out = solve_model(m, 50.0; rng = StableRNG(5), return_hidden_states = true)
            @test out isa Tuple && length(out) == 2
            @test size(out[1]) == size(out[2])
        end
    end

    @testset "BlowflyModel" begin
        m = BlowflyModel()

        @testset "reproducible under a fixed rng" begin
            @test solve_model(m, 100.0; rng = StableRNG(11)) == solve_model(m, 100.0; rng = StableRNG(11))
            @test solve_model(m, 100.0; rng = StableRNG(11)) != solve_model(m, 100.0; rng = StableRNG(12))
        end

        @testset "shape and support" begin
            traj = solve_model(m, 100.0; rng = StableRNG(11))
            @test size(traj, 1) == 1
            @test all(≥(0), traj)               # a population count
            @test all(isfinite, traj)
        end
    end

    @testset "solve_model returns Matrix{Float64} for every model" begin
        # loss.jl asserts ::Matrix{Float64} on the simulation result, so a type change there
        # breaks the whole MCMC path.
        for m in all_models()
            traj = solve_model(m, 50.0; rng = StableRNG(2))
            @test traj isa Matrix{Float64}
            @test size(traj, 1) == (m isa Lorenz63Model ? 3 : 1)
        end
    end

    @testset "changing active parameters changes the simulation" begin
        for m in updatable_models()
            _, values = get_active_model_params(m)
            m2 = update_model_parameters(m, collect(values) .* 1.5 .+ 0.1)
            @test solve_model(m, 30.0; rng = StableRNG(4)) != solve_model(m2, 30.0; rng = StableRNG(4))
        end
    end
end
