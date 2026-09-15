function create_simulated_data( model, target, buffers, options, rng )
    diff_orders = target.data.difference_orders
    ndata = options.N_obs
    embedding_dim = options.embedding_dim
    embedding_type = options.embedding_type
    dt_obs = model.dt_obs

    # TODO impement solve_model!()
    Rsim = solve_model( model, ndata*dt_obs; rng = rng )::Matrix{Float64}
    Rsim = embedding( Rsim, embedding_dim, embedding_type )

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

function calculate_simulated_statistics( target, Rsim_container, summaries, buffers, options, lossfun )
    resampler = options.resampling_type
    n_summaries = options.n_summaries

    resample_buffer = buffers.mcmc_buffer
    index_cache = buffers.index_cache
    sim_statistic = buffers.simulation_statistic

    if n_summaries <= 1 && isa(resampler, LengthPreservingSampler )
        view_in = @view resample_buffer[:, 1]
        summaries( view_in, index_cache, index_cache, target, Rsim_container, buffers
        )
    else
        for ii in 1:n_summaries
            view_in = @view resample_buffer[ :, ii ]
            x_inds, y_inds = resampler( target.data, options, index_cache )
            summaries( view_in, x_inds, y_inds, target, Rsim_container, buffers )
        end
    end

    # Average the resampled summaries to get the final simulated statistic which is then
    # compared to the target statistic to calculate the loss.
    mean!( sim_statistic, resample_buffer )

    return sim_statistic
end

function calculate_loss( params, target, model, mcmc_options; rng_seed::UInt64 = rand(UInt64) )
    logprior = evaluate_log_prior( params, target.priors )

    # Parameters with zero prior density should have zero likelihood so we can return -Inf
    # to avoid unnecessary simulations
    if isinf( logprior )
        # println( "Parameters with zero prior density encountered. Returning -Inf for likelihood." )
        return -Inf
    end

    model = update_model_parameters( model, params )    # Update model with new parameters for simulation
    rng = Xoshiro( rng_seed )                           # Set the same random seed for each data generation

    loss = calculate_loss( target.options.inference_method, target, model, mcmc_options, rng )
    isinf( loss ) && return loss

    noise_scale = mcmc_options.likelihood_noise_scale
    loss += logprior
    loss += noise_scale*randn() # Add noise to likelihood to simulate noisy likelihood

    return loss
end

# GSL: simulate once (or n_loss_evals times, averaging), and compare against the fixed target
# mean/covariance estimated once from the observed data (see train_target/TargetData).
function calculate_loss( ::GSL, target, model, mcmc_options, rng )
    loss_function = mcmc_options.loss_function
    options = target.options
    summaries = target.summary_statistics
    buffers = target.buffers

    loss = 0.0

    # n_loss_evals is the number of times to evaluate the loss function on new simulations and average the result to reduce the effect of noise.
    for _ in 1:options.n_loss_evals
        Rsim_container = create_simulated_data( model, target, buffers, options, rng )

        # If the simulation failed (e.g. due to numerical instability) and returned NaNs, we can
        # return -Inf for the likelihood to reject this parameter proposal
        if isinf( Rsim_container.observations[1] )
            return -Inf
        end

        sim_statistic = calculate_simulated_statistics( target, Rsim_container, summaries, buffers, options, loss_function )
        loss += loss_function( target, sim_statistic )
    end

    return loss / options.n_loss_evals
end

# BSL: simulate n_sim fresh datasets at the current parameters, resample/summarize each one exactly
# as GSL does at MCMC time (calculate_simulated_statistics, so any resampler works), and estimate
# the mean and covariance of the likelihood from the spread across those n_sim simulations. The
# fixed observed summary (target.obs_mean) was computed once in TargetData.
function calculate_loss( bsl::BSL, target, model, mcmc_options, rng )
    loss_function = mcmc_options.loss_function
    options = target.options
    summaries = target.summary_statistics
    buffers = target.buffers
    sim_summaries = buffers.bsl_buffer

    for jj in 1:bsl.n_sim
        Rsim_container = create_simulated_data( model, target, buffers, options, rng )

        if isinf( Rsim_container.observations[1] )
            return -Inf
        end

        sim_statistic = calculate_simulated_statistics( target, Rsim_container, summaries, buffers, options, loss_function )
        sim_summaries[:, jj] .= sim_statistic
    end

    sim_mean = vec( mean( sim_summaries, dims=2 ) )
    sim_cov_factorization = regularized_cholesky( cov( sim_summaries' ) )

    target = @set target.cov_factorization = sim_cov_factorization

    return loss_function( target, sim_mean )
end
