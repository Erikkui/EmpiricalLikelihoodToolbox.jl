Base.@kwdef struct OUModel{T} <: AbstractSimulationModel
    theta::Float64 = 3.0
    mu::Float64    = 0.0
    sigma::Float64 = 1.0
    x0::T    = 0.0 # Initial condition
    dt_obs::Float64 = 0.01
    dt_sol::Float64 = dt_obs
    dim::Int = length(x0)
    all_parameters::Tuple{Symbol} = [:theta, :mu, :sigma]
    active_parameters::Tuple{Symbol} = [:theta, :mu, :sigma]
end

function step!(rng::AbstractRNG, m::OUModel, x, dt_sol, cumulative_t)
    # Euler-Maruyama discretization
    dx = m.theta .* (m.mu .- x) .* dt_sol .+ m.sigma .* sqrt(dt_sol) .* randn(rng)
    return x .+ dx
end
