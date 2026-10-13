"""
    Lorenz63Model{T}

A struct for the chaotic Lorenz 1963 dynamical system.

# Fields
- `sigma::Float64`: The Prandtl number (default: 10.0).
- `rho::Float64`: The Rayleigh number (default: 28.0).
- `beta::Float64`: The geometric factor (default: 8/3).
- `x0::T`: The initial state vector of the system. Default: [12.0, 19.0, 23.0].
- `dt_obs::Float64`: The observation time step (default: 1.0).
- `dt_sol::Float64`: The RK4 step of the stochastic version (default: 0.01). Not used by the deterministic version.
- `process_noise::Float64`: q, the sd of the Gaussian noise added to the state after each `dt_obs` (default: 0.0).
- `obs_noise::Float64`: σ_o, the sd of the Gaussian noise added to every observed component (default: 0.0).
- `burn_in::Float64`: Model time simulated before the first observation (default: 10 * dt_obs).

# Two versions
With `process_noise == 0` and `obs_noise == 0` the model is the deterministic ODE (Tsit5), with a 1% random
perturbation of `x0` as the only noise. Otherwise it is the state-space model
    x_{t+1} = Φ_θ^{dt_obs}(x_t) + q ξ_t,   y_t = x_t + σ_o ε_t,   ξ_t, ε_t ~ N(0, I),
where Φ is the RK4 flow of the ODE with step `dt_sol` (`lorenz63_flow`). The first state is the perturbed `x0`
after `burn_in` time units of the same noisy dynamics, and y_1 observes it.

# Examples
model = Lorenz63Model(dt_obs = 1.0, x0 = [1.0, 0.0, 0.0])
model = Lorenz63Model(dt_obs = 0.1, process_noise = 0.3, obs_noise = 1.0)
"""
Base.@kwdef struct Lorenz63Model{T <: AbstractVector{Float64}} <: AbstractSimulationModel
    sigma::Float64 = 10.0
    rho::Float64    = 28.0
    beta::Float64   = 8/3
    x0::T = [12.0, 19.0, 23.0]
    dt_obs::Float64 = 1.0
    dt_sol::Float64 = 0.01
    process_noise::Float64 = 0.0
    obs_noise::Float64 = 0.0
    burn_in::Float64 = 10.0 * dt_obs
    dim::Int = length(x0)
    all_parameters::Tuple{ Vararg{Symbol} } = (:sigma, :rho, :beta)
    active_parameters::Tuple{ Vararg{Symbol} } = (:sigma, :rho, :beta)
end


function lorenz_static(u, p, t)
    σ, ρ, β = p
    x, y, z = u

    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z

    return SVector(dx, dy, dz)
end


# One classical RK4 step of size h
@inline function _lorenz63_rk4_step(u::SVector{3, Float64}, p::SVector{3, Float64}, h::Float64)
    k1 = lorenz_static(u, p, 0.0)
    k2 = lorenz_static(u + (h / 2) * k1, p, 0.0)
    k3 = lorenz_static(u + (h / 2) * k2, p, 0.0)
    k4 = lorenz_static(u + h * k3, p, 0.0)
    return u + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end

"""
    lorenz63_flow(u, p, Δ, h)

The state after time Δ from state `u` for parameters `p = (σ, ρ, β)`: RK4 with `round(Int, Δ/h)` equal sub-steps
(at least one) of size close to `h`. Allocation-free for `SVector` inputs.
"""
@inline function lorenz63_flow(u::SVector{3, Float64}, p::SVector{3, Float64}, Δ::Float64, h::Float64)
    n_sub = max(1, round(Int, Δ / h))
    h_sub = Δ / n_sub
    for _ in 1:n_sub
        u = _lorenz63_rk4_step(u, p, h_sub)
    end
    return u
end

# Lorenz 63 utilizes solver from DifferentialEquations.jl, or the RK4 state-space version when noise is switched on
function solve_model(model::Lorenz63Model, t_end::Float64; rng=Random.default_rng(), return_hidden_states=false )
    if model.process_noise > 0.0 || model.obs_noise > 0.0
        return solve_noisy_lorenz63(model, t_end, rng, return_hidden_states)
    end
    data = solve_deterministic_lorenz63(model, t_end, rng)
    return return_hidden_states ? (data, copy(data)) : data
end

# State-space version: x_{t+1} = flow(x_t) + q ξ_t, y_t = x_t + σ_o ε_t. Number of observations: round(t_end / dt_obs).
function solve_noisy_lorenz63(model::Lorenz63Model, t_end::Float64, rng::AbstractRNG, return_hidden_states::Bool)
    Δ = model.dt_obs
    n_obs = round(Int, t_end / Δ)
    q = model.process_noise
    σ_o = model.obs_noise

    _, params = get_all_model_params(model)
    p = SVector{3}( params )

    x = SVector{3}( model.x0 ) .* (1.0 .+ 0.01 .* SVector( randn(rng), randn(rng), randn(rng) ))
    for _ in 1:round(Int, model.burn_in / Δ)
        x = lorenz63_flow(x, p, Δ, model.dt_sol) + q * SVector( randn(rng), randn(rng), randn(rng) )
    end

    states = Matrix{Float64}(undef, 3, n_obs)
    for t in 1:n_obs
        t > 1 && ( x = lorenz63_flow(x, p, Δ, model.dt_sol) + q * SVector( randn(rng), randn(rng), randn(rng) ) )
        states[:, t] .= x
    end
    if !all(isfinite, states)   # the solver left the attractor; same convention as the deterministic version
        return return_hidden_states ? (fill(NaN, 3, n_obs), states) : fill(NaN, 3, n_obs)
    end
    data = states .+ σ_o .* randn(rng, 3, n_obs)
    return return_hidden_states ? (data, states) : data
end

function solve_deterministic_lorenz63(model::Lorenz63Model, t_end::Float64, rng::AbstractRNG)
    dt_obs = model.dt_obs

    noise = SVector( randn(rng), randn(rng), randn(rng) )
    u0 = SVector{3}( model.x0 ) .* (1.0 .+ 0.01 .* noise)

    _, params = get_all_model_params(model)
    params = SVector{3}( params )

    t_start = model.burn_in
    t_end   = t_start + t_end

    save_steps = t_start:dt_obs:t_end-dt_obs

    problem = ODEProblem(lorenz_static, u0, (0.0, t_end), params)

    # save_start=false ensures we don't accidentally save t=0
    sol = solve( problem, Tsit5(); saveat=save_steps )

    raw_sol = sol.u

    # Check if the solver aborted or failed to reach the end
    if isempty(raw_sol) || length(raw_sol) < length(save_steps)
        # Return a matrix of NaNs. Assuming Lorenz 63 has 3 dimensions.
        return fill(NaN, 3, length(save_steps))
    end

    return stack(raw_sol)
end
