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
Base.@kwdef struct Lorenz63Model{T <: AbstractVector{Float64}, P} <: AbstractSimulationModel
    sigma::Float64 = 10.0
    rho::Float64    = 28.0
    beta::Float64   = 8/3
    x0::T = [12.0, 19.0, 23.0]
    dt_obs::Float64 = 1.0
    dt_sol::Float64 = 1.0
    dim::Int = length(x0)
    all_parameters::Tuple{Symbol} = [:sigma, :rho, :beta]
    active_parameters::Tuple{Symbol} = [:sigma, :rho, :beta]
end


function lorenz_static(u, p, t)
    σ, ρ, β = p
    x, y, z = u

    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z

    return SVector(dx, dy, dz)
end
