Base.@kwdef struct NormalModel <: AbstractSimulationModel
    mu::Float64 = 0.0
    sigma::Float64 = 1.0
    dim::Int = 1
    dt_obs::Float64 = 1.0
    dt_sol::Float64 = dt_obs
end

# initial_state(m::NormalModel) = zeros(m.dim)

function step!(rng::AbstractRNG, m::NormalModel, state, dt_obs)
    return m.sigma .* randn(rng, m.dim) .+ m.mu
end

function get_params(m::NormalModel)
    param_tuple = (
        mu = m.mu,
        sigma = m.sigma
    )
    return param_tuple
end

# function reconstruct(m::NormalModel, new_params)
#     return NormalModel(new_params[1], new_params[2], m.dim, m.dt_obs, m.dt_sol)
# end
