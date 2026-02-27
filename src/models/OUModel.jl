Base.@kwdef struct OUModel{T} <: AbstractSimulationModel
    theta::Float64 = 1.0
    mu::Float64    = 0.0
    sigma::Float64 = 0.5
    x0::T    = 0.0 # Initial condition
    dt::Float64 = 0.01
    dim::Int = length(x0)
end

initial_state(m::OUModel) = m.x0

function step!(rng::AbstractRNG, m::OUModel, x, dt)
    # Euler-Maruyama discretization
    dx = m.theta .* (m.mu .- x) .* dt .+ m.sigma .* sqrt(dt) .* randn(rng)
    return x .+ dx
end

function get_params(m::OUModel)
    return [m.theta, m.mu, m.sigma]
end

function reconstruct(m::OUModel, new_params)
    # Use the old 'dt' but new parameters
    return OUModel(new_params[1], new_params[2], new_params[3], m.x0, m.dt, m.dim)
end
