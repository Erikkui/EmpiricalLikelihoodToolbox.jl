Base.@kwdef struct NormalModel{T} <: AbstractSimulationModel
    mu::Float64 = 0.0
    sigma::Float64 = 1.0
    x0::T = 0.0 # Initial condition (unused: each observation is an iid draw)
    dim::Int = 1
    dt_obs::Float64 = 1.0
    dt_sol::Float64 = dt_obs
    all_parameters::Tuple{ Vararg{Symbol} } = (:mu, :sigma)
    active_parameters::Tuple{ Vararg{Symbol} } = (:mu, :sigma)
end

function step!(rng::AbstractRNG, m::NormalModel, state, dt_sol, cumulative_t)
    return m.sigma .* randn(rng, m.dim) .+ m.mu
end
