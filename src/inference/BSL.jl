#------------Bayesian Synthetic Likelihood (Wood, 2010): the observed summary is a fixed quantity,
# and the mean/covariance used in the likelihood are re-estimated at every MCMC step from `n_sim`
# fresh simulations at the current parameter proposal. The `loss` must use a covariance, so
# RobustChamfer is rejected.
struct BSL{L} <: AbstractInferenceMethod
    n_sim::Int
    loss::L

    function BSL( n_sim::Integer, loss::L ) where {L}
        if loss isa RobustChamfer
            throw( ArgumentError( "BSL does not support RobustChamfer: it ignores the covariance that BSL estimates from simulations." ) )
        end
        return new{L}( n_sim, loss )
    end
end
BSL( n_sim::Integer; loss = LogLikelihood() ) = BSL( n_sim, loss )
BSL(; n_sim::Integer = 10, loss = LogLikelihood() ) = BSL( n_sim, loss )

# `BSL(500)` for the default loss, `BSL(500, LogLikelihood(2.0))` otherwise. Stable: used in file names.
function Base.show( io::IO, method::BSL )
    if method.loss == LogLikelihood()
        print( io, "BSL(", method.n_sim, ")" )
    else
        print( io, "BSL(", method.n_sim, ", ", _loss_string( method.loss ), ")" )
    end
end

# Under BSL the observed summary is a fixed reference value; its covariance is irrelevant, since
# the likelihood's covariance is re-estimated from simulations at every MCMC step (see log_likelihood!).
# The covariance from _train_reference_summary is therefore discarded.
function train_target( ::BSL, statistics, data_container, buffer_container, options )
    mean_summary, _, training_summaries = _train_reference_summary( statistics, data_container, buffer_container, options )
    return mean_summary, nothing, training_summaries
end

# Fixed reference summary of the observed data, shared by BSL and ABC.
# With a deterministic, length-preserving resampler (eg. NoResampling), a single pass over all the
# available data already *is* the reference statistic, so the training_resamplings loop is skipped
# entirely and there is no covariance (`nothing`). For any other resampler (eg. length-changing
# splitters like RademacherSplit, which have no well-defined "use all the data" shortcut), we still
# need to average over the resampler's own variability to get a stable reference value, computed
# consistently with how simulated summaries will be resampled during MCMC - so we simply reuse
# GSL's training loop, which also returns the training covariance.
function _train_reference_summary( statistics, data_container, buffer_container, options )
    if options.resampling_type isa NoResampling
        index_cache = buffer_container.index_cache
        view_in = @view buffer_container.training_buffer[:, 1]
        statistics( view_in, index_cache, index_cache, data_container, buffer_container )

        mean_summary = collect( view_in )
        training_summaries = reshape( mean_summary, :, 1 )
        return mean_summary, nothing, training_summaries
    end

    return train_target( GSL(), statistics, data_container, buffer_container, options )
end

# BSL refactorizes from simulations at every MCMC step and never reads the target's factorization;
# this identity placeholder only fixes the type and size of the TargetData field.
finalize_target_covariance( ::BSL, cov_mat, summary_length ) = cholesky( Matrix( 1.0I, summary_length, summary_length ) )

# Holds the n_sim simulated summaries of one MCMC step, one per column.
allocate_method_buffer( bsl::BSL, summary_length ) = zeros( summary_length, bsl.n_sim )

# Standardization is calibrated against the identity placeholder covariance, which is inconsistent
# with the covariance BSL estimates from simulations, so it is not supported.
function check_method_options( ::BSL, options )
    if options.standardize
        throw( ArgumentError( "standardize=true is not supported with BSL (standardization is calibrated with a placeholder covariance, not BSL's simulated one)." ) )
    end
    return nothing
end

# Simulate n_sim fresh datasets at the current parameters, resample/summarize each one exactly
# as GSL does at MCMC time (calculate_simulated_statistics, so any resampler works), and estimate
# the mean and covariance of the likelihood from the spread across those n_sim simulations. The
# fixed observed summary (target.obs_mean) was computed once in TargetData.
function log_likelihood!( method_data, bsl::BSL, inference_state, target, model, mcmc_options, rng )
    loss_function = bsl.loss
    options = target.options
    summaries = target.summary_statistics
    buffers = target.buffers
    sim_summaries = buffers.method_buffer

    for jj in 1:bsl.n_sim
        Rsim_container = create_simulated_data( model, target, buffers, options, rng )

        if isinf( Rsim_container.observations[1] )
            return -Inf
        end

        sim_statistic = calculate_simulated_statistics( target, Rsim_container, summaries, buffers, options )
        sim_summaries[:, jj] .= sim_statistic
    end

    sim_mean = vec( mean( sim_summaries, dims=2 ) )
    sim_cov_factorization = regularized_cholesky( cov( sim_summaries' ) )

    # The loss is symmetric in which side is "observed", so the simulated mean is compared against
    # the fixed observed summary using the simulated covariance. The whole value is scaled.
    raw_loss = loss_function( sim_mean, target.obs_mean, sim_cov_factorization )
    raw_loss -= 0.5 * logdet( sim_cov_factorization )

    return scale_loss( loss_function, raw_loss, target )
end
