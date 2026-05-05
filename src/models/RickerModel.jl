Base.@kwdef struct RickerModel{T} <: AbstractSimulationModel
    r::Float64 = 3.4
    K:: Float64 = 1000.0
    x0::T    = 10.0 # Initial condition
    dt_obs::Float64 = 0.01
    dt_sol::Float64 = dt_obs
    dim::Int = length(x0)
end

function step!(rng::AbstractRNG, m::RickerModel, x, dt_sol, cumulative_t)
    x = x*exp( m.r * ( 1 - x/500 ) )
    return x
end

function get_params(m::RickerModel)
    param_tuple = (
        r = m.r,
    )
    return param_tuple
end
