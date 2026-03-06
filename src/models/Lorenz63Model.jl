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
end

function lorenz_static(u, p, t)
    σ, ρ, β = p
    x, y, z = u

    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z

    return SVector(dx, dy, dz)
end

function get_params(m::Lorenz63Model)
    return [m.sigma, m.rho, m.beta]
end

function reconstruct(m::Lorenz63Model, new_params)
    return Lorenz63Model(new_params[1], new_params[2], new_params[3], m.x0, m.dt_obs, m.dt_sol, m.dim)
end
