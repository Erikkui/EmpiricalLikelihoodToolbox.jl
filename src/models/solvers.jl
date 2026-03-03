# General solver for models that do not rely on DifferentialEquations.jl
function solve_model( model::AbstractSimulationModel, t_end::Float64; rng=Random.default_rng() )
    dt_obs = model.dt_obs
    dt_sol = model.dt_sol

    steps = round(Int, t_end/dt_sol)
    obs_inds = 1:round(Int, dt_obs/dt_sol):steps
    trajectory = Matrix{Float64}(undef, model.dim, steps)

    current_state = initial_state(model)

    for ii in 1:steps
        current_state = step!(rng, model, current_state, dt_sol)
        trajectory[:, ii] .= current_state
    end

    # println( "traj: ", size(trajectory) )
    # sleep(2)
    return trajectory
end

# Lorenz 63 utilizes solver from DifferentialEquations.jl
function solve_model(model::Lorenz63Model, t_end::Float64; rng=Random.default_rng() )
    dt_obs = model.dt_obs

    noise = SVector( randn(rng), randn(rng), randn(rng) )
    u0 = SVector{3}( model.x0 ) .* (1.0 .+ 0.01 .* noise)

    params = SVector{3}( get_params(model) )

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
