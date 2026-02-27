Base.@kwdef struct Lorenz63Model{T <: AbstractVector{Float64}} <: AbstractSimulationModel
    sigma::Float64 = 10.0
    rho::Float64    = 28.0
    beta::Float64   = 8/3
    x0::T = [12.0, 19.0, 23.0]
    dt::Float64 = 0.01
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
    # Use the old 'dt' but new parameters
    return Lorenz63Model(new_params[1], new_params[2], new_params[3], m.x0, m.dt, m.dim)
end
