# Inference method interface (implemented once per method, see also BSL.jl):
#   train_target( method, statistics, data_container, buffer_container, options ) -> (mean, cov or nothing, training_summaries)
#   finalize_target_covariance( method, cov_mat, summary_length ) -> factorization stored in the target
#   allocate_method_buffer( method, summary_length ) -> method-specific scratch buffer
#   check_method_options( method, options ) -> throws ArgumentError on an invalid combination
#   calculate_loss( method, target, model, mcmc_options, rng ) -> log-likelihood (no prior)
# Optional (default in ABC.jl returns the target unchanged):
#   finalize_inference_method( method, target, training_summaries, rng ) -> target, run at the end of TargetData

#------------Gaussian Subset Likelihood (Haario et al., 2015): the target mean/covariance are
# estimated once from the observed data (via resampling), and compared against a single simulated
# summary at each MCMC step. The `loss` is applied to the (simulated, target) summary pair.
struct GSL{L} <: AbstractInferenceMethod
    loss::L
end
GSL(; loss = LogLikelihood()) = GSL( loss )

# `GSL()` for the default loss, `GSL(RobustChamfer(1.0))` otherwise. Stable: used in file names.
function Base.show( io::IO, method::GSL )
    if method.loss == LogLikelihood()
        print( io, "GSL()" )
    else
        print( io, "GSL(", _loss_string( method.loss ), ")" )
    end
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
    else
        throw( ArgumentError( "Unknown covariance_type $(cov_type). Must be :cov or :donsker." ) )
    end

    return mean_summary, C, training_summaries
end

# GSL's target covariance is fixed for the whole run, so it is factorized once here.
finalize_target_covariance( ::GSL, cov_mat, summary_length ) = regularized_cholesky( cov_mat )

# GSL needs no extra buffer.
allocate_method_buffer( ::GSL, summary_length ) = Matrix{Float64}( undef, 0, 0 )

# Standardization is calibrated from the spread of losses across training_summaries. NoResampling
# is deterministic, so every training draw is identical (the loop runs training_resamplings times
# over the same fixed indices) - the resulting spread is degenerate (zero variance), so
# standardization is meaningless and disallowed under NoResampling.
function check_method_options( ::GSL, options )
    if options.resampling_type isa NoResampling && options.standardize
        throw( ArgumentError( "standardize=true is not supported with NoResampling (no variability to standardize against)." ) )
    end
    return nothing
end

# Simulate once (or n_loss_evals times, averaging), and compare against the fixed target
# mean/covariance estimated once from the observed data (see train_target/TargetData).
function calculate_loss( gsl::GSL, target, model, mcmc_options, rng )
    loss_function = gsl.loss
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

        sim_statistic = calculate_simulated_statistics( target, Rsim_container, summaries, buffers, options )
        raw_loss = loss_function( sim_statistic, target.obs_mean, target.cov_factorization )
        loss += scale_loss( loss_function, raw_loss, target )
    end

    return loss / options.n_loss_evals
end
