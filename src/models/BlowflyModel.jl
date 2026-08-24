"""
    BlowflyModel{T}

A struct for the Nicholson blowfly model, which is a delay differential equation model of blowfly population dynamics.

# Fields
- `delta::Float64`: (default: 0.16).
- `P::Float64`: (default: 6.5).
- `N0::Float64`: (default: 400).
- `sigma2_p::Float64`: (default: 0.1).
- `tau::Float64`: (default: 14).
- `sigma2_d::Float64`: (default: 0.1).
- `x0::T`: The initial population. Default: 180.
- `dt_obs::Float64`: The observation time step (default: 1.0).

# Examples
model = BlowflyModel(dt_obs = 1.0, x0 = [1.0, 0.0, 0.0])
"""
Base.@kwdef struct BlowflyModel <: AbstractSimulationModel
    delta::Float64 = 0.16
    P::Float64    = 6.5
    N0::Float64   = 400.0
    sigma2_p::Float64 = 0.1
    tau::Float64 = 14.0
    sigma2_d::Float64 = 0.1
    x0::Int = 180
    dt_obs::Float64 = 1.0
    dt_sol::Float64 = 1.0
    dim::Int = length(x0)
    burn_in::Int = 20
    mu::Float64 = 1.0
    all_parameters::Tuple{ Vararg{Symbol} } = (:delta, :P, :N0, :sigma2_p, :tau, :sigma2_d)
    active_parameters::Tuple{ Vararg{Symbol} } = (:delta, :P, :N0, :sigma2_p, :tau, :sigma2_d)
end

function solve_model( model::BlowflyModel, t_end::Float64; rng=Random.default_rng() )

    # Unpack parameters
    _, parameters = get_all_model_params(model)
    delta, P, N_0, sigma2_p, tau, sigma2_d = parameters

    burn_in = model.burn_in
    mu = model.mu
    N_init = model.x0

    # Initializing needed variables
    lag = Int( round(tau) ) > 0 ? Int( round(tau) ) : 1     # Lag time in days
    total_time = round( Int, t_end ) + lag + burn_in
    N = Matrix{Float64}(undef, 1, total_time )  # Population size vector
    N[ 1, 1:lag ] .= N_init  # Initializing first lag days with N_init

    gamma_p = Gamma( mu^2/sigma2_p, sigma2_p/mu )   # Gamma distribution for reproduction rate
    gamma_d = Gamma( mu^2/sigma2_d, sigma2_d/mu )   # Gamma distribution for death rate
    ee = rand( rng, gamma_p, total_time )                # Reproduction rate noise
    epsilon = rand( rng, gamma_d, total_time )           # Death rate noise

    # Iterate over time steps; first burn in period, then actual simulation
    # Note: We start from lag+1 because we need to access N[ii - lag]
    for ii in lag+1:total_time
        Nlag = @view N[ 1, ii - lag ]       # Population size at lag time
        Nprev = @view N[ 1, ii - 1 ]        # Population size at previous time step
        ee_t = @view ee[ ii - 1 ]           # Reproduction rate noise at lag time
        epsilon_t = @view epsilon[ ii - 1 ] # Death rate noise at lag time

        # Expected values: R ~ Poisson, S ~ binom
        R_t = P*Nlag*exp.( -Nlag/N_0 ).*ee_t      # E(X) = lambda
        S_t = Nprev*exp.( -delta*epsilon_t )      # E(X) = n*p
        N[ii] = R_t + S_t[]
    end

    # Remove initial lag and burn-in period from N
    N = N[:, lag+1+burn_in:end]

    return N
end
