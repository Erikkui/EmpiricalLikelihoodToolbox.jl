"""
    Lorenz63Model{T}

A struct for the chaotic Lorenz 1963 dynamical system.

# Fields
- `sigma::Float64`: The Prandtl number (default: 10.0).
- `rho::Float64`: The Rayleigh number (default: 28.0).
- `beta::Float64`: The geometric factor (default: 8/3).
- `x0::T`: The initial state vector of the system. Default: [12.0, 19.0, 23.0].
- `dt_obs::Float64`: The observation time step (default: 1.0).

# Examples
model = Lorenz63Model(dt_obs = 1.0, x0 = [1.0, 0.0, 0.0])
"""
Base.@kwdef struct Lorenz63Model{T <: AbstractVector{Float64}} <: AbstractSimulationModel
    sigma::Float64 = 10.0
    rho::Float64    = 28.0
    beta::Float64   = 8/3
    x0::T = [12.0, 19.0, 23.0]
    dt_obs::Float64 = 1.0
    dt_sol::Float64 = 1.0
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


# Lorenz 63 utilizes solver from DifferentialEquations.jl
function solve_model(model::Lorenz63Model, t_end::Float64; rng=Random.default_rng() )
    dt_obs = model.dt_obs

    noise = SVector( randn(rng), randn(rng), randn(rng) )
    u0 = SVector{3}( model.x0 ) .* (1.0 .+ 0.01 .* noise)

    _, params = get_all_model_params(model)
    params = SVector{3}( params )

    t_start = 10.0 * dt_obs
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
