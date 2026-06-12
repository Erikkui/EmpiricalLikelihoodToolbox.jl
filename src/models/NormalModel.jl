Base.@kwdef struct NormalModel <: AbstractSimulationModel
    mu::Float64 = 0.0
    sigma::Float64 = 1.0
    dim::Int = 1
    dt_obs::Float64 = 1.0
    dt_sol::Float64 = dt_obs
    all_parameters::Tuple{Symbol} = [:mu, :sigma]
    active_parameters::Tuple{Symbol} = [:mu, :sigma]
end

function step!(rng::AbstractRNG, m::NormalModel, state, dt_obs)
    return m.sigma .* randn(rng, m.dim) .+ m.mu
end
