#------------EvaluatedState: one chain state (parameter vector) and its evaluated log-target
# The samplers own preallocated evaluated states (the current state and one per proposal stage) and swap
# them on acceptance, so the per-state data of the method (`method_data`, eg. the ABC distances)
# always belongs to the state it was simulated for.
# - `log_target`: log_prior + log_likelihood (+ likelihood_noise_scale noise); -Inf for a zero
#   prior density or a failed simulation.
# - `method_data`: see allocate_method_data (ABC: the n_sim distances; GSL/BSL: nothing).
# - `simulated`: false when the log-likelihood was not evaluated (zero prior density, or early
#   rejection); the method data is then invalid (NaN).
mutable struct EvaluatedState{A}
    params::Vector{Float64}
    log_prior::Float64
    log_likelihood::Float64
    log_target::Float64
    method_data::A
    simulated::Bool
end

function EvaluatedState( method::AbstractInferenceMethod, npar::Integer )
    params = zeros( npar )
    method_data = allocate_method_data( method )
    return EvaluatedState( params, -Inf, -Inf, -Inf, method_data, false )
end

# Evaluate the log-target at `params` (which may be `evaluated_state.params` itself) into `evaluated_state`.
# The simulations use Xoshiro( rng_seed ), so two evaluations with the same seed share their random
# numbers (common random numbers, see reevaluate_current_loss). A zero prior density returns before
# any simulation. Early rejection: when even the largest possible log-likelihood of the method
# (max_log_likelihood) gives a log-target below `log_threshold`, the simulation is skipped and the
# log-target is -Inf. The sampler passes log_threshold = log_u + current log-target, so such a
# proposal could not be accepted and skipping it is exact. The default -Inf never skips.
function evaluate_log_target!( evaluated_state, target, model, params, mcmc_options, inference_state;
                               rng_seed::UInt64 = rand(UInt64), log_threshold::Float64 = -Inf )
    copyto!( evaluated_state.params, params )

    log_prior = evaluate_log_prior( evaluated_state.params, target.priors, model.active_parameters )
    evaluated_state.log_prior = log_prior

    # Parameters with zero prior density have zero posterior density: no simulation is needed
    if isinf( log_prior )
        evaluated_state.log_likelihood = -Inf
        evaluated_state.log_target = -Inf
        evaluated_state.simulated = false
        invalidate_method_data!( evaluated_state.method_data )
        return evaluated_state
    end

    method = target.options.inference_method

    # Early rejection: the proposal could not be accepted even with the largest log-likelihood
    log_target_bound = log_prior + max_log_likelihood( method, inference_state )
    if log_target_bound < log_threshold
        evaluated_state.log_likelihood = -Inf
        evaluated_state.log_target = -Inf
        evaluated_state.simulated = false
        invalidate_method_data!( evaluated_state.method_data )
        return evaluated_state
    end

    updated_model = update_model_parameters( model, evaluated_state.params )
    rng = Xoshiro( rng_seed )

    log_likelihood = log_likelihood!( evaluated_state.method_data, method, inference_state, target, updated_model, mcmc_options, rng )
    evaluated_state.log_likelihood = log_likelihood
    evaluated_state.simulated = true

    if !isfinite( log_likelihood )
        evaluated_state.log_target = -Inf
        return evaluated_state
    end

    # Noise added to the likelihood to simulate a noisy likelihood (testing aid). Drawn only when
    # used, so an evaluated_state skipped by early rejection leaves the global RNG where a full one would.
    noise_scale = mcmc_options.likelihood_noise_scale
    if noise_scale == 0
        noise = 0.0
    else
        noise = noise_scale * randn()
    end
    evaluated_state.log_target = log_likelihood + log_prior + noise

    return evaluated_state
end

# The log-target (log prior + log-likelihood) at `params`, evaluated once. A thin wrapper around
# evaluate_log_target! with a temporary evaluated_state and a fresh run state of the target's method.
function evaluate_log_target( params, target, model, mcmc_options; rng_seed::UInt64 = rand(UInt64) )
    method = target.options.inference_method
    inference_state = allocate_inference_state( method )
    evaluated_state = EvaluatedState( method, length( params ) )
    evaluate_log_target!( evaluated_state, target, model, params, mcmc_options, inference_state; rng_seed = rng_seed )
    return evaluated_state.log_target
end

# Metropolis acceptance probability min( 1, exp( log_ratio ) ); 0 for a NaN log ratio (eg. both
# log-targets -Inf).
function _acceptance_probability( log_ratio::Float64 )
    if isnan( log_ratio )
        return 0.0
    end
    return min( 1.0, exp( log_ratio ) )
end
