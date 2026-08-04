function create_simulated_data( R0_all, model, target, buffers, options )
    diff_orders = target.data.difference_orders
    ndata = size( R0_all.observations, 2 )
    dt_obs = model.dt_obs

    Rsim = solve_model( model, ndata*dt_obs )::Matrix{Float64}

    if any(isnan, Rsim)
        # println("Simulation returned NaN values. Returning -Inf for likelihood.")
        Rsim .= -Inf
        Rsim_container = DataContainer(
            observations=Rsim,
            differences=buffers.simulation_diffs,
            difference_orders=diff_orders,
            options=options
            )
        return Rsim_container
    end

    copyto!( buffers.simulation_obs, Rsim )
    if maximum(diff_orders) > 0
        calculate_diffs!( buffers.simulation_diffs, Rsim, diff_orders, dt_obs )
    end

    Rsim_container = DataContainer(
        observations=Rsim,
        differences=buffers.simulation_diffs,
        difference_orders=diff_orders,
        options=options
        )
    return Rsim_container
end

function calculate_simulated_statistics( R0_all, Rsim_container, summaries, buffers, options )
    resampler = options.resampling_type
    n_summaries = options.n_summaries

    resample_buffer = buffers.mcmc_buffer
    index_cache = buffers.index_cache
    sim_statistic = buffers.simulation_statistic
    for ii in 1:n_summaries
        view_in = @view resample_buffer[ :, ii ]
        x_inds, y_inds = resampler( R0_all, options, index_cache )
        summaries( view_in, x_inds, y_inds, R0_all, Rsim_container, buffers )
    end

    mean!( sim_statistic, resample_buffer )

    return sim_statistic
end

function calculate_loss( params, target, model, mcmc_options )
    loss_function = mcmc_options.loss_function
    noise_scale = mcmc_options.likelihood_noise_scale
    noise_scale = ifelse( isnan(noise_scale), 0.0, noise_scale )

    logprior = evaluate_log_prior( params, target.priors )

    # Parameters with zero prior density should have zero likelihood so we can return to avoid unnecessary simulations
    if isinf( logprior )
        # println( "Parameters with zero prior density encountered. Returning -Inf for likelihood." )
        return -Inf
    end

    options = target.options
    R0_all = target.data
    summaries = target.summary_statistics
    buffers = target.buffers

    model = update_model_parameters( model, params )    # Update model with new parameters for simulation

    Rsim_container = create_simulated_data( R0_all, model, target, buffers, options )

    # If the simulation failed (e.g. due to numerical instability) and returned NaNs, we can return -Inf for the likelihood to reject this parameter proposal
    if isinf( Rsim_container.observations[1] )
        # println("Simulation failed for parameters: ", params, " with log prior: ", logprior, ". Returning -Inf for likelihood.")
        return -Inf
    end

    # Resample data and calculate summary statistics for current parameters
    loss = 0.0
    for _ in 1:options.n_loss_evals
        sim_statistic = calculate_simulated_statistics( R0_all, Rsim_container, summaries, buffers, options )
        loss += loss_function( target, sim_statistic )
        loss += logprior

        loss += noise_scale*randn() # Add noise to likelihood to simulate noisy likelihood
    end
    loss /= options.n_loss_evals

    return loss
end
