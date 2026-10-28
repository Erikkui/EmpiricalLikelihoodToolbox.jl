function initialize_datacontainer( data, options, diff_orders, dt_obs::Float64 )
    if maximum(diff_orders) > 0
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
    if max_diff_order > 0
        buffer_differences = Vector{Matrix{Float64}}(undef, max_diff_order+2 )
        for ii in 1:max_diff_order
            if ii in diff_orders
                buffer_differences[ii] = zeros( size(observations) )
            else
                buffer_differences[ii] = Matrix{Float64}(undef, 0, 0)
            end
        end
        buffer_differences[ end-1 ] = zeros( size(observations) )
        buffer_differences[ end ] = zeros( size(observations) )

        # calculate_diffs pads the boundaries by nearest-neighbour copy to keep the column count
        # equal to the observations', so index ii means the same time point in both - but the
        # padded columns are fabricated, and each successive order propagates that fabrication one
        # column further inward. Dropping max_diff_order columns from each end excludes every
        # fabricated point while preserving that one-to-one indexing.
        min_ind = max_diff_order + 1
        max_ind = ind_size - max_diff_order
        index_cache = collect( min_ind:max_ind )
    else
        buffer_differences = Vector{Matrix{Float64}}(undef, 0)
        index_cache = collect( 1:ind_size )
    end

    # Buffers sized off a resampler's output lengths must be built from the effective (post-trim)
    # count, since that is what the resampler will actually be handed at run time.
    effective_nobs = length( index_cache )
    data_container = @set data_container.options.effective_N_obs = effective_nobs

    stat_buffers_vals  = map( stat -> allocate_buffer( stat, data_container ), statistics )
    stat_buffers_names = map( stat -> Symbol( stat.name ), statistics )
    stat_buffers = NamedTuple{stat_buffers_names}(stat_buffers_vals)

    training_summary_length = map( stat -> get_summary_length( stat, data_container ), statistics ) |> sum

    training_buffer = zeros( training_summary_length, training_resamplings )
    mcmc_buffer = zeros( training_summary_length, n_summaries )
    simulation_statistic_buffer = zeros( training_summary_length )

    method_buffer = allocate_method_buffer( options.inference_method, training_summary_length )

    buffers = BufferContainer(
        stat_buffers,
        training_buffer,
        mcmc_buffer,
        buffer_differences,
        simulation_statistic_buffer,
        index_cache,
        method_buffer,
        )
    return buffers, data_container, training_summary_length
end


# A parameter with no prior falls back to a flat (uninformative) prior in evaluate_log_prior;
# just warn so the user notices a partially-specified priors NamedTuple.
function _check_priors( priors, model::AbstractSimulationModel )
    active_param_names, _ = get_active_model_params( model )
    if !( isnothing( priors ) || isempty( priors ) )
        missing_names = Tuple( setdiff( active_param_names, keys( priors ) ) )
        if !isempty( missing_names )
            @warn "No prior specified for parameter(s) $missing_names; using a flat (uninformative) prior for them."
        end
    end
end

# The observed series fixes the length of every simulation. Any N_obs passed in the options must agree.
function _set_n_obs( options::MethodsOptions, data )
    N_obs = size( data, 2 )
    if options.N_obs != 0 && options.N_obs != N_obs
        throw( ArgumentError( "MethodsOptions.N_obs = $(options.N_obs) does not match the $(N_obs) observations in the data. N_obs is taken from the data; leave it unset." ) )
    end
    return @set options.N_obs = N_obs
end


# `data` is the raw observed series. It is embedded here with options.embedding_dim/embedding_type,
# exactly as create_simulated_data embeds each simulation, and its length sets options.N_obs, the
# length of every simulation. `model` is stored in the target and is the one mcmcrun runs. The loss
# lives in the inference method (`options.inference_method.loss`). `rng` drives the method's
# finalization (eg. ABC's pilot simulations); training resampling does not use it.
function TargetData(
    data::AbstractMatrix{Float64},
    summary_stats::JointSummaryStatistics,
    options::MethodsOptions,
    model::AbstractSimulationModel;
    priors = nothing,
    rng = Random.default_rng(),
    )

    statistics = summary_stats.statistics
    training_resamplings = options.training_resamplings

    diff_orders = map( required_diff_order, statistics )

    check_method_options( options.inference_method, options )
    _check_priors( priors, model )  # Only warning if some priors are missing
    options = _set_n_obs( options, data )
    observations = embedding( data, options.embedding_dim, options.embedding_type )

    # Create DataContainer. The observed data's differences use the model's dt_obs, the same step
    # create_simulated_data uses for the simulated data's differences.
    data_container = initialize_datacontainer( observations, options, diff_orders, model.dt_obs )

    # Create buffers for use in resampling
    buffer_container, data_container, total_summary_length = allocate_buffers( statistics, data_container, options, diff_orders )

    # allocate_buffers writes the post-trim effective_N_obs into data_container.options; keep this
    # binding in sync so target.options and target.data.options don't diverge.
    options = data_container.options

    # Initialize bins for all summary statistics, overwriting original summary statistics
    statistics = map(
        stat -> create_bins( data_container, stat, options, buffer_container.index_cache ),
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
    cov_factorization = finalize_target_covariance( inference_method, cov_mat, total_summary_length )

    # Calculate standardization factors for loss function if requested
    mean_standardization = nothing
    sd_standardization = nothing
    if options.standardize
        losses = zeros( training_resamplings )
        loss = inference_method.loss
        for (ii, col) in enumerate( eachcol( training_summaries ) )
            temp = loss( col, mean_summary, cov_factorization )
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
        cov_factorization,
        total_summary_length,
        mean_standardization,
        sd_standardization,
        model )

    # Resolve method settings that need the finished target (eg. ABC's distance and ε, which may
    # need pilot simulations). Returns the target unchanged for GSL and BSL.
    target = finalize_inference_method( inference_method, target, training_summaries, rng )

    return target, training_summaries
end
#-------------------------------

# Deprecated: the loss now belongs to the inference method. Kept so existing scripts keep working;
# it stores `loss` in `options.inference_method` and calls the new method. ABC has no loss, so
# passing one with ABC throws rather than silently ignoring it.
function TargetData(
    data::AbstractMatrix{Float64},
    summary_stats::JointSummaryStatistics,
    options::MethodsOptions,
    loss,
    model::AbstractSimulationModel;
    priors = nothing,
    )
    Base.depwarn(
        "TargetData(data, stats, options, loss, model) is deprecated: set the loss in the inference method (eg. `GSL(loss = $(_loss_string( loss )))` or `BSL(n_sim; loss = ...)`) and call TargetData(data, stats, options, model).",
        :TargetData,
        )
    if options.inference_method isa ABC
        throw( ArgumentError( "ABC has no loss function. Call TargetData(data, stats, options, model) without the loss." ) )
    end
    options = @set options.inference_method.loss = loss
    return TargetData( data, summary_stats, options, model; priors = priors )
end
