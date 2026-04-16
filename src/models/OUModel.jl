Base.@kwdef struct OUModel{T} <: AbstractSimulationModel
    theta::Float64 = 3.0
    mu::Float64    = 0.0
    sigma::Float64 = 1.0
    x0::T    = 0.0 # Initial condition
    dt_obs::Float64 = 0.01
    dt_sol::Float64 = dt_obs
    dim::Int = length(x0)
end

# initial_state(m::OUModel) = m.x0

function step!(rng::AbstractRNG, m::OUModel, x, dt_sol)
    # Euler-Maruyama discretization
    dx = m.theta .* (m.mu .- x) .* dt_sol .+ m.sigma .* sqrt(dt_sol) .* randn(rng)
    return x .+ dx
end

function get_params(m::OUModel)
    param_tuple = (
        theta = m.theta,
        mu = m.mu,
        sigma = m.sigma
    )
    return param_tuple
end

# function reconstruct(m::OUModel, new_params)
#     return OUModel(new_params[1], new_params[2], new_params[3], m.x0, m.dt_obs, m.dt_sol, m.dim)
# end
