function TargetData(
    data::AbstractMatrix{Float64},
    summaries::JointSummaryStatistics,
    options::MethodsOptions
    )
    println( "Started creating training target with ", size(data, 2), " data points" )

    statistics = summaries.statistics
    training_resamplings = options.training_resamplings

    training_summary_length = map( summary -> summary.summary_length, statistics ) |> sum

    # Create DataContainer
    diff_orders = map( required_diff_order, summaries.statistics )
    if maximum(diff_orders) > 0
        diff_inds = diff_orders .> 0
        ind = findfirst( diff_inds )
        dt_obs = summaries.statistics[ ind ].dt_obs
        difference_data = calculate_diffs( data, diff_orders, dt_obs )
        buffer_differences = Vector{Matrix{Float64}}(undef, maximum(diff_orders) )
        for ii in 1:maximum(diff_orders)
            if ii in diff_orders
                buffer_differences[ii] = zeros( size(data) )
            else
                buffer_differences[ii] = Matrix{Float64}(undef, 0, 0)
            end
        end
    else
        difference_data = Vector{Matrix{Float64}}(undef, 0)
        buffer_differences = Vector{Matrix{Float64}}(undef, 0)
    end
    data_container = DataContainer( observations=data, differences=difference_data, difference_orders=diff_orders, options=options )

    # Create buffers for use in resampling
    stat_buffers_vals  = map( summary -> allocate_buffer( summary, data_container ), statistics )
    stat_buffers_names = map( summary -> nameof(typeof(summary)), statistics )
    stat_buffers = NamedTuple{stat_buffers_names}(stat_buffers_vals)

    training_buffer = zeros( training_summary_length, training_resamplings )
    mcmc_buffer = zeros( training_summary_length, options.n_summaries )
    mean_buffer = zeros( training_summary_length )

    ind_size = get_index_size( options.resampling_type, data, options )
    index_cache = collect( 1:ind_size )

    buffer_observations = zeros( size(data) )

    buffers = BufferContainer( stat_buffers, training_buffer, mcmc_buffer, buffer_observations, buffer_differences, mean_buffer, index_cache )

    # Initialize bins for all summary statistics
    println( "Initializing bins for ecdf summary statistics..." )
    statistics = map( summary -> initialize_bins( data_container, summary, options ), statistics )
    summaries = JointSummaryStatistics( statistics )

    # Resample observations and calculate summary statistics mean and cov for MCMC target
    println( "Resampling data for target mean and covariance..." )
    iter = ProgressBar( 1:training_resamplings, printing_delay=0.01 )
    for ii in iter
        view_in = @view buffers.training_buffer[:, ii]
        x_inds, y_inds = options.resampling_type( data_container, options, index_cache )
        summaries( view_in, x_inds, y_inds, data_container, buffers )
        # println( round.(view_in, digits=3) )
        # sleep(0.5)
    end

    training_summaries = buffers.training_buffer
    mean_summary = mean( training_summaries, dims=2 ) |> vec
    C = cov( training_summaries' )
    inverse_cov = pinv( C )

    target = TargetData( data_container, summaries, options, buffers, mean_summary, inverse_cov, training_summary_length )
    return target, training_summaries
end
#-------------------------------
