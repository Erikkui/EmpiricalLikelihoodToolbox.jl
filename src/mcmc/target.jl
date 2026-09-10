function initialize_datacontainer( data, statistics::Tuple, options, diff_orders )
    if maximum(diff_orders) > 0
        diff_inds = diff_orders .> 0
        ind = findfirst( diff_inds )
        dt_obs = statistics[ ind ].dt_obs
        difference_data = calculate_diffs( data, diff_orders, dt_obs )
    else
        difference_data = Vector{Matrix{Float64}}(undef, 0)
    end

    data_container = DataContainer(
        observations = data,
        differences = difference_data,
        difference_orders = diff_orders,
        options = options
        )
    return data_container
end

function allocate_buffers( statistics::Tuple, data_container, options, diff_orders )
    training_resamplings = options.training_resamplings
    n_summaries = options.n_summaries
    resampling_type = options.resampling_type

    observations = data_container.observations

    ind_size = get_index_size( resampling_type, observations, options )
    max_diff_order = maximum(diff_orders)
    if maximum(diff_orders) > 0
        index_cache = collect( 1:ind_size )
        buffer_differences = Vector{Matrix{Float64}}(undef, maximum(diff_orders)+2 )
        for ii in 1:max_diff_order
            if ii in diff_orders
                buffer_differences[ii] = zeros( size(observations) )
            else
                buffer_differences[ii] = Matrix{Float64}(undef, 0, 0)
            end
        end
        buffer_differences[ end-1 ] = zeros( size(observations) )
        buffer_differences[ end ] = zeros( size(observations) )
    else
        buffer_differences = Vector{Matrix{Float64}}(undef, 0)

        # For each diff order, observation is lost at the beginning and end of the data,
        # so we need to adjust the index cache accordingly.
        min_ind = max_diff_order + 1
        max_ind = ind_size-max_diff_order
        index_cache = collect( min_ind:max_ind )
    end

    effective_nobs = length( index_cache )
    data = @set data_container.options.effective_N_obs = effective_nobs

    stat_buffers_vals  = map( stat -> allocate_buffer( stat, data_container ), statistics )
    stat_buffers_names = map( stat -> Symbol( generate_stat_name( stat ) ), statistics )
    stat_buffers = NamedTuple{stat_buffers_names}(stat_buffers_vals)

    training_summary_length = map( stat -> get_summary_length( stat, data_container ), statistics ) |> sum

    training_buffer = zeros( training_summary_length, training_resamplings )
    mcmc_buffer = zeros( training_summary_length, n_summaries )
    simulation_statistic_buffer = zeros( training_summary_length )

    buffer_observations = zeros( size(observations) )

    inference_method = options.inference_method
    bsl_buffer = isa( inference_method, BSL ) ?
        zeros( training_summary_length, inference_method.n_sim ) :
        Matrix{Float64}( undef, 0, 0 )

    buffers = BufferContainer(
        stat_buffers,
        training_buffer,
        mcmc_buffer,
        buffer_observations,
        buffer_differences,
        simulation_statistic_buffer,
        index_cache,
        bsl_buffer,
        )
    return buffers, data, training_summary_length
end


function train_target( ::GSL, statistics, data_container, buffer_container, options )
    training_summaries = buffer_container.training_buffer
    index_cache = buffer_container.index_cache

    resampling_type = options.resampling_type
    training_resamplings = options.training_resamplings
    cov_type = options.covariance_type

    if options.verbose
        println( "Resampling data for target mean and covariance, ndata = $(size(data_container.observations, 2))" )
        iter = ProgressBar( 1:training_resamplings, printing_delay=0.1 )
    else
        iter = 1:training_resamplings
    end

    for ii in iter
        x_inds, y_inds = resampling_type( data_container, options, index_cache )

        view_in = @view training_summaries[:, ii]
        statistics( view_in, x_inds, y_inds, data_container, buffer_container )
    end

    mean_summary = mean( training_summaries, dims=2 ) |> vec

    C = nothing
    if cov_type == :cov
        C = cov( training_summaries' )
    elseif cov_type == :donsker
        ndata = size( data_container.observations, 2 )
        C = donsker_covariance( mean_summary, ndata )
    end

    return mean_summary, C, training_summaries
end

# Under BSL the observed summary is a fixed reference value; its covariance is irrelevant, since
# the likelihood's covariance is re-estimated from simulations at every MCMC step (see calculate_loss).
# With a deterministic, length-preserving resampler (eg. NoResampling), a single pass over all the
# available data already *is* the reference statistic, so the training_resamplings loop is skipped
# entirely. For any other resampler (eg. length-changing splitters like RademacherSplit, which have
# no well-defined "use all the data" shortcut), we still need to average over the resampler's own
# variability to get a stable reference value, computed consistently with how simulated summaries
# will be resampled during MCMC - so we simply reuse GSL's training loop and discard its covariance.
function train_target( ::BSL, statistics, data_container, buffer_container, options )
    if options.resampling_type isa NoResampling
        index_cache = buffer_container.index_cache
        view_in = @view buffer_container.training_buffer[:, 1]
        statistics( view_in, index_cache, index_cache, data_container, buffer_container )

        mean_summary = collect( view_in )
        training_summaries = reshape( mean_summary, :, 1 )
    else
        mean_summary, _, training_summaries = train_target( GSL(), statistics, data_container, buffer_container, options )
    end

    return mean_summary, nothing, training_summaries
end

# GSL's target covariance is fixed for the whole run, so it is regularized and inverted once here.
finalize_target_covariance( ::GSL, cov_mat, summary_length ) = regularized_inverse( cov_mat )

# BSL re-estimates and inverts its covariance from simulations at every MCMC step (see
# calculate_loss); the placeholder here only needs to give `target.inverse_cov` a concrete,
# correctly-sized type up front so later `@set`s in the MCMC loop don't change its element type.
finalize_target_covariance( ::BSL, cov_mat, summary_length ) = zeros( summary_length, summary_length )



function TargetData(
    data::AbstractMatrix{Float64},
    summary_stats::JointSummaryStatistics,
    options::MethodsOptions;
    priors = nothing,
    loss = nothing
    )

    if isnothing( priors )
       priors = ( nothing, )
    end

    statistics = summary_stats.statistics
    training_resamplings = options.training_resamplings

    diff_orders = map( required_diff_order, statistics )

    # Create DataContainer
    data_container = initialize_datacontainer( data, statistics, options, diff_orders )

    # Create buffers for use in resampling
    buffer_container, data_container, total_summary_length = allocate_buffers( statistics, data_container, options, diff_orders )

    # Initialize bins for all summary statistics, overwriting original summary statistics
    statistics = map(
        stat -> initialize_bins( data_container, stat, options, buffer_container.index_cache ),
        statistics
        )

    # Finalize summaries if there are some params yet needed to be set
    statistics = map(
        stat -> finalize_summary( stat, data_container, buffer_container ),
        statistics
        )

    # Create final JointSummaryStatistics object with updated summary statistics
    statistics = JointSummaryStatistics( statistics )

    # Resample observations and calculate summary statistics mean and cov for MCMC target
    inference_method = options.inference_method
    mean_summary, cov_mat, training_summaries = train_target( inference_method, statistics, data_container, buffer_container, options )
    inv_cov_mat = finalize_target_covariance( inference_method, cov_mat, total_summary_length )

    # Standardization is calibrated from the spread of losses across training_summaries. NoResampling
    # is deterministic, so every training draw is identical regardless of inference_method (GSL loops
    # training_resamplings times over the same fixed indices; BSL skips the loop and uses a single
    # deterministic pass, see train_target above) - the resulting spread is degenerate (zero variance),
    # so standardization is meaningless and disallowed under NoResampling for both GSL and BSL.
    if options.resampling_type isa NoResampling && options.standardize
        throw( ArgumentError( "standardize=true is not supported with NoResampling (no variability to standardize against)." ) )
    end

    # Calculate standardization factors for loss function if requested
    mean_standardization = nothing
    sd_standardization = nothing
    if options.standardize
        losses = zeros( training_resamplings )
        for (ii, col) in enumerate( eachcol( training_summaries ) )
            temp = loss( col, mean_summary, inv_cov_mat )
            losses[ii] = temp
        end
        mean_standardization = mean(losses)
        sd_standardization = std(losses; corrected=false)
    end

    target = TargetData(
        data_container,
        statistics,
        priors,
        options,
        buffer_container,
        mean_summary,
        inv_cov_mat,
        total_summary_length,
        mean_standardization,
        sd_standardization )

    return target, training_summaries
end
#-------------------------------


# # Target generation for cases where we want simple mcmc (data against observations), no
# # hassling with summary statistics

# function initialize_datacontainer( data, options )
#     data_container = DataContainer(
#         observations = data,
#         differences = nothing,
#         difference_orders = nothing,
#         options = options
#         )
#     return data_container
# end


# function allocate_bufers( data_container )
#     observations = data_container.observations
#     buffer_simulations = zeros( size(observations) )

#     buffers = BufferContainer(
#         nothing,
#         nothing,
#         nothing,
#         buffer_simulations,
#         nothing,
#         nothing,
#         nothing,
#         )
#     return buffers
# end

# function TargetData(
#     data::AbstractMatrix{Float64},
#     options::MethodsOptions;
#     priors = nothing,
#     loss = nothing
#     )

#     if isnothing( priors )
#         priors = ( nothing, )
#     end

#     # Create DataContainer
#     data_container = initialize_datacontainer( data, options )

#     # Create buffers for use in resampling
#     buffer_container, total_summary_length = allocate_buffers( data_container )

#     inv_cov =

#     target = TargetData(
#         data_container,
#         nothing,
#         priors,
#         options,
#         buffer_container,
#         nothing,
#         nothing,
#         nothing,
#         nothing,
#         nothing
#         )

#     return target
# end
