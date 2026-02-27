# General solver for models that do not rely on DifferentialEquations.jl
function solve_model( model::AbstractSimulationModel, t_end::Float64; rng=Random.default_rng() )
    dt = model.dt

    steps = round(Int, t_end/dt)
    trajectory = Matrix{Float64}(undef, model.dim, steps)

    current_state = initial_state(model)

    for ii in 1:steps
        current_state = step!(rng, model, current_state, dt)
        trajectory[:, ii] .= current_state
    end

    # println( "traj: ", size(trajectory) )
    # sleep(2)
    return trajectory
end

# Lorenz 63 utilizes solver from DifferentialEquations.jl
function solve_model(model::Lorenz63Model, t_end::Float64; rng=Random.default_rng() )
    dt = model.dt

    noise = SVector( randn(rng), randn(rng), randn(rng) )
    u0 = SVector{3}( model.x0 ) .* (1.0 .+ 0.01 .* noise)

    params = SVector{3}( get_params(model) )

    t_start = 10.0 * dt
    t_end   = t_start + t_end

    problem = ODEProblem(lorenz_static, u0, (0.0, t_end), params)

    # save_start=false ensures we don't accidentally save t=0
    sol = solve( problem, Tsit5(); saveat=t_start:dt:t_end-dt )

    return stack(sol.u)
end
