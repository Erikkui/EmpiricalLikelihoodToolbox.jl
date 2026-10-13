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
        # x0's default (mu + sigma*randn()) makes column 1 a real draw too, not the initial state.
        samples = traj[1, :]
        @test mean(samples) ≈ mu atol = 0.05
        @test std(samples) ≈ sigma atol = 0.05

        @testset "each fresh construction draws a fresh x0" begin
            firsts = [solve_model(NormalModel(mu = 5.0, sigma = 2.0), 10.0; rng = StableRNG(s))[1, 1]
                      for s in 1:5]
            @test length(unique(firsts)) > 1
        end
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

    @testset "Lorenz63Model state-space version" begin
        ELT = EmpiricalLikelihoodToolbox
        p = ELT.SVector(10.0, 28.0, 8 / 3)
        u0 = ELT.SVector(12.0, 19.0, 23.0)

        @testset "RK4 flow agrees with Tsit5" begin
            prob = ELT.ODEProblem(ELT.lorenz_static, u0, (0.0, 0.5), p)
            ref = ELT.solve(prob, ELT.Vern9(); abstol = 1e-12, reltol = 1e-12).u[end]
            err(h) = maximum(abs, lorenz63_flow(u0, p, 0.5, h) - ref)
            @test err(0.01) < 1e-4                  # truncation error, far below the noise scales used (q ~ 0.3)
            @test err(0.01) / err(0.005) > 12       # fourth order: halving h divides the error by about 16
            @test err(0.005) / err(0.0025) > 12
        end

        @testset "noise levels" begin
            q, σ_o, Δ = 0.3, 1.0, 0.1
            m = Lorenz63Model(dt_obs = Δ, process_noise = q, obs_noise = σ_o)
            y, x = solve_model(m, 5000.0; rng = StableRNG(11), return_hidden_states = true)
            @test size(y) == (3, 50_000) && size(x) == size(y)
            @test all(isfinite, y)
            @test std(y .- x) ≈ σ_o rtol = 0.02
            # one-step residuals of the hidden state are the process noise
            resid = [lorenz63_flow(ELT.SVector{3}(x[:, t]), p, Δ, m.dt_sol) for t in 1:size(x, 2) - 1]
            resid = x[:, 2:end] .- stack(resid)
            @test std(resid) ≈ q rtol = 0.02
        end

        @testset "reproducible, and default model keeps the deterministic path" begin
            m = Lorenz63Model(dt_obs = 0.1, process_noise = 0.3, obs_noise = 1.0)
            @test solve_model(m, 20.0; rng = StableRNG(5)) == solve_model(m, 20.0; rng = StableRNG(5))
            m0 = Lorenz63Model()
            @test solve_model(m0, 20.0; rng = StableRNG(5)) == solve_model(m0, 20.0; rng = StableRNG(5))
            @test size(solve_model(m0, 20.0; rng = StableRNG(5))) == (3, 20)
        end
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

        @testset "has no embedding_dim field (that belongs on MethodsOptions)" begin
            # examples/main.jl used to pass embedding_dim to RickerModel(...) directly, which has
            # no such field; it now goes to MethodsOptions instead.
            @test !hasfield(RickerModel, :embedding_dim)
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

        @testset "can be constructed and updated" begin
            # x0::E ties BlowflyModel{E} to a field, so E is inferred from the default and
            # Accessors' positional rebuild works -- the model is usable from MCMC.
            @test (try; BlowflyModel(); true; catch; false; end)
            @test (try
                update_model_parameters(BlowflyModel{Int}(), [0.2, 6.0, 400.0, 0.1, 14.0, 0.1])
                true
            catch; false; end)
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
