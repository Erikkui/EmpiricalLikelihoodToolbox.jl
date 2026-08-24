Base.@kwdef struct RickerModel{T, E} <: AbstractSimulationModel
    r::Float64 = 44.7
    sigma::Float64 = 0.3
    phi::Float64 = 10.0
    x0::T    = 1.0 # Initial condition
    dt_obs::Float64 = 1.0
    dt_sol::Float64 = dt_obs
    embedding_dim::E = 0
    dim::Int = 1+length( embedding_dim )
    active_parameters::Tuple{Vararg{Symbol}} = (:r, :sigma, :phi)
    all_parameters::Tuple{Vararg{Symbol}} = (:r, :sigma, :phi)
end

function step!(rng::AbstractRNG, m::RickerModel, n )
    # Ricker map

    (; r, sigma, phi) = m

    z_t = sigma*randn( rng )
    n = r * n * exp( -n + z_t )

    lambda = phi*n
    y_t = rand( rng, Poisson( lambda ) )

    return y_t, n
end


function solve_model( model::RickerModel, t_end::Float64; rng=Random.default_rng(), transient_time = 500.0, transform_log1p=true, return_hidden_states=false )
    (; r, sigma, phi) = model
    dt_obs = model.dt_obs
    dt_sol = model.dt_sol
    embedding_dims = model.embedding_dim

    end_time_total = t_end + transient_time + maximum( embedding_dims )*dt_obs
    steps = round(Int, end_time_total/dt_sol )

    obs_start_ind = round( Int, (transient_time+dt_obs)/dt_sol )
    obs_end_ind = steps
    obs_ind_step = round( Int, dt_obs/dt_sol )
    obs_inds = obs_start_ind:obs_ind_step:obs_end_ind

    current_state = initial_state( model )
    trajectory = zeros( 1, steps )
    trajectory[ :, 1 ] .= rand( rng, Poisson( current_state*model.phi ) )

    if return_hidden_states
        hidden_states = zeros( 1, steps )
        hidden_states[ :, 1 ] .= current_state
    end

    for ii in 2:steps
        y_t, current_state = step!( rng, model, current_state )
        trajectory[ :, ii ] .= y_t
        if return_hidden_states
            hidden_states[ :, ii ] .= current_state
        end
    end

    trajectory = trajectory[ :, obs_inds ]

    if transform_log1p
        trajectory = log1p.( trajectory )
    end

    # If embedding dimension is specified, compute the lag embedding
    if embedding_dims != 0
        y_out = embedding( trajectory, embedding_dims )
        if return_hidden_states
            hidden_states = hidden_states[ :, obs_inds ]
            n_out = embedding( hidden_states, embedding_dims )
            return y_out, n_out
        end
    else
        return trajectory # Return as a 2D array with one row
    end
end
