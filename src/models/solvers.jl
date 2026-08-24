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
