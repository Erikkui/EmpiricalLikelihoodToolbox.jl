Base.@kwdef struct NegExpModel <: AbstractSimulationModel
    theta1::Float64 = 1.0
    theta2::Float64 = 1.5
    x0::Float64 = 0.0 # Initial condition
    dt_obs::Float64 = 0.01
    dt_sol::Float64 = dt_obs
    dim::Int = length(x0)
    noise_scale::Float64 = 1.0
end

function step!( rng::AbstractRNG, m::NegExpModel, x, dt_sol, cumulative_t )
    # Euler-Maruyama discretization
    dx = m.theta1*m.theta2*exp( -m.theta2*cumulative_t )*dt_sol
    process_noise = m.noise_scale*sqrt(dt_sol)*randn(rng)
    return x + dx + process_noise
end

function get_params(m::NegExpModel)
    param_tuple = (
        theta1 = m.theta1,
        theta2 = m.theta2,
    )
    return param_tuple
end
