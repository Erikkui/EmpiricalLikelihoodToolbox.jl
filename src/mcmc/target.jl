function initialize_datacontainer( data, statistics, options, diff_orders )
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

function allocate_buffers( statistics, data_container, options, diff_orders )
    training_resamplings = options.training_resamplings
    n_summaries = options.n_summaries
    resampling_type = options.resampling_type

    observations = data_container.observations

    training_summary_length = map( stat -> stat.summary_length, statistics ) |> sum

    stat_buffers_vals  = map( stat -> allocate_buffer( stat, data_container ), statistics )
    stat_buffers_names = map( stat -> nameof( typeof(stat) ), statistics )
    stat_buffers = NamedTuple{stat_buffers_names}(stat_buffers_vals)

    training_buffer = zeros( training_summary_length, training_resamplings )
    mcmc_buffer = zeros( training_summary_length, n_summaries )
    simulation_statistic_buffer = zeros( training_summary_length )

    ind_size = get_index_size( resampling_type, observations, options )
    index_cache = collect( 1:ind_size )

    buffer_observations = zeros( size(observations) )

    if maximum(diff_orders) > 0
        buffer_differences = Vector{Matrix{Float64}}(undef, maximum(diff_orders)+2 )
        for ii in 1:maximum(diff_orders)
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
    end

    buffers = BufferContainer(
        stat_buffers,
        training_buffer,
        mcmc_buffer,
        buffer_observations,
        buffer_differences,
        simulation_statistic_buffer,
        index_cache,
        )
    return buffers, training_summary_length
end


function train_target( statistics, data_container, buffer_container, options )
    training_summaries = buffer_container.training_buffer
    index_cache = buffer_container.index_cache

    resampling_type = options.resampling_type
    training_resamplings = options.training_resamplings

    # Show progress bar only on first thread to avoid cluttering output
    if Threads.nthreads() == 1 || Threads.threadid() == 1
        println( "Resampling data for target mean and covariance, ndata = $(size(data_container.observations, 2))" )
        iter = ProgressBar( 1:training_resamplings, printing_delay=0.1 )
    else
        iter = 1:training_resamplings
    end

    for ii in iter
        view_in = @view training_summaries[:, ii]
        x_inds, y_inds = resampling_type( data_container, options, index_cache )
        statistics( view_in, x_inds, y_inds, data_container, buffer_container )
    end

    mean_summary = mean( training_summaries, dims=2 ) |> vec
    C = cov( training_summaries' )
    inverse_cov = pinv( C )
    return mean_summary, inverse_cov, training_summaries
end



function TargetData(
    data::AbstractMatrix{Float64},
    summary_stats::JointSummaryStatistics,
    options::MethodsOptions;
    )
    statistics = summary_stats.statistics
    training_resamplings = options.training_resamplings

    diff_orders = map( required_diff_order, statistics )

    # Create DataContainer
    data_container = initialize_datacontainer( data, statistics, options, diff_orders )

    # Create buffers for use in resampling
    buffer_container, total_summary_length = allocate_buffers( statistics, data_container, options, diff_orders )

    # Initialize bins for all summary statistics, overwriting original summary statistics
    statistics = map( stat -> initialize_bins( data_container, stat, options ), statistics )
    statistics = JointSummaryStatistics( statistics )

    # Resample observations and calculate summary statistics mean and cov for MCMC target
    mean_summary, inverse_cov, training_summaries = train_target( statistics, data_container, buffer_container, options )

    target = TargetData(
        data_container,
        statistics,
        options,
        buffer_container,
        mean_summary,
        inverse_cov,
        total_summary_length )

    return target, training_summaries
end
#-------------------------------
