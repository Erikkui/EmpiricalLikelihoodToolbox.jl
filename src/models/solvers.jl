# Solver for models that can be solved using Euler-Maryama
function solve_model( model::AbstractSimulationModel, t_end::Float64; rng=Random.default_rng() )
    dt_obs = model.dt_obs
    dt_sol = model.dt_sol

    steps = round(Int, t_end/dt_sol)
    obs_inds = 1:round(Int, dt_obs/dt_sol):steps
    trajectory = Matrix{Float64}(undef, model.dim, steps)

    current_state = initial_state( model )
    trajectory[:, 1] .= current_state

    cumulative_t = 0.0
    for ii in 2:steps
        cumulative_t += dt_sol
        current_state = step!(rng, model, current_state, dt_sol, cumulative_t)
        trajectory[:, ii] .= current_state
    end

    trajectory = trajectory[:, obs_inds]

    return trajectory
end

# Lorenz 63 utilizes solver from DifferentialEquations.jl
function solve_model(model::Lorenz63Model, t_end::Float64; rng=Random.default_rng() )
    dt_obs = model.dt_obs

    noise = SVector( randn(rng), randn(rng), randn(rng) )
    u0 = SVector{3}( model.x0 ) .* (1.0 .+ 0.01 .* noise)

    _, params = get_all_model_params(model)
    params = SVector{3}( params )

    t_start = 10.0 * dt_obs
    t_end   = t_start + t_end

    save_steps = t_start:dt_obs:t_end-dt_obs

    problem = ODEProblem(lorenz_static, u0, (0.0, t_end), params)

    # save_start=false ensures we don't accidentally save t=0
    sol = solve( problem, Tsit5(); saveat=save_steps )

    raw_sol = sol.u

    # Check if the solver aborted or failed to reach the end
    if isempty(raw_sol) || length(raw_sol) < length(save_steps)
        # Return a matrix of NaNs. Assuming Lorenz 63 has 3 dimensions.
        return fill(NaN, 3, length(save_steps))
    end

    return stack(raw_sol)
end

# Blowfly model solver requires a custom implementation due to its custom discretization
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
