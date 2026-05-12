Base.@kwdef struct PredatorModel{T} <: AbstractSimulationModel
    a::Float64 = 2.0
    b::Float64 = 2.0
    c::Float64 = 2.0
    d::Float64 = 1.85
    alpha::Float64 = 0.1
    beta::Float64 = 0.1
    tau::Float64 = 1.27
    x0::T    = [1.5, 0.5] # Initial condition
    dt_obs::Float64 = 1.0
    dt_sol::Float64 = dt_obs
    dim::Int = length(x0)
end

function step!(rng::AbstractRNG, m::PredatorModel, s, dt_sol, cumulative_t)
    # Discrete prey-predator map
    x, y = s
    a, b, c, d, alpha, beta, tau = m.a, m.b, m.c, m.d, m.alpha, m.beta, m.tau

    dx = tau*x*( a - x - b*y / ( (1 + alpha*x ) * (1 + beta*y ) ) )
    dy = tau*y*( -c + d*x / ( (1 + alpha*x ) * (1 + beta*y ) ) )

    s[1] += dx
    s[2] += dy
    return s
end

function get_params(m::PredatorModel)
    param_tuple = (
        a = m.a,
        b = m.b,
        c = m.c,
        d = m.d,
        alpha = m.alpha,
        beta = m.beta,
        tau = m.tau
    )
    return param_tuple
end
