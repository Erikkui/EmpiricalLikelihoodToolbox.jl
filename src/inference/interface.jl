#------------Inference method interface
# An inference method (GSL.jl, BSL.jl, ABC/) sets up the target and evaluates the log-likelihood
# of one parameter vector. The samplers (src/mcmc/AM.jl, DRAM.jl) only see these functions, through
# `evaluate_log_target!` (src/mcmc/evaluated_state.jl) and the run-state functions below.
#
# Setup, implemented once per method:
#   train_target( method, statistics, data_container, buffer_container, options ) -> (mean, cov or nothing, training_summaries)
#   finalize_target_covariance( method, cov_mat, summary_length ) -> factorization stored in the target
#   allocate_method_buffer( method, summary_length ) -> method-specific scratch buffer, shared by all evaluated states
#   check_method_options( method, options ) -> throws ArgumentError on an invalid combination
#   finalize_inference_method( method, target, training_summaries, rng ) -> target, run at the end of TargetData
#
# Evaluation, implemented once per method:
#   log_likelihood!( method_data, method, inference_state, target, model, mcmc_options, rng ) -> Float64
#     The log-likelihood (no prior) at the parameters already set in `model`. Writes the per-state
#     data of the method (see allocate_method_data) into `method_data`.
#
# Optional, with the fallbacks below:
#   allocate_method_data( method ) -> per-state data an evaluated state must keep (ABC: its n_sim distances)
#   invalidate_method_data!( method_data ) -> marks the data of a state that was not simulated
#   max_log_likelihood( method, inference_state ) -> upper bound of log_likelihood! (Inf: none known)
#   early_rejection_allowed( method, ii ) -> false when AM step ii needs every proposal simulated
#   allocate_inference_state( method ) -> mutable run state of one chain (ABC: the current ε)
#   adapt_inference_state!( inference_state, method, current, acceptance_probability, ii ) -> Bool, after each AM step
#   allocate_results( method, model, chain_length, n_stages ) -> ResultsBuffer
#   record_chain_step!( results, method, inference_state, current, ii ) -> stores the current state at step ii
#   finalize_results( results, method, inference_state, last_step ) -> results of a run that stored last_step states

# GSL and BSL need nothing resolved.
finalize_inference_method( method::AbstractInferenceMethod, target, training_summaries, rng ) = target

# The loss function of a method (checked against MCMCOptions.loss_function). ABC has none.
_method_loss( method::AbstractInferenceMethod ) = method.loss


#------------Per-state data
allocate_method_data( ::AbstractInferenceMethod ) = nothing

invalidate_method_data!( ::Nothing ) = nothing

function invalidate_method_data!( method_data::AbstractArray{Float64} )
    fill!( method_data, NaN )
    return nothing
end

# No upper bound is known for GSL and BSL.
max_log_likelihood( ::AbstractInferenceMethod, inference_state ) = Inf

# Whether AM may skip the simulation of a proposal that could not be accepted at step ii (early
# rejection, see MCMCOptions.early_rejection).
early_rejection_allowed( ::AbstractInferenceMethod, ii ) = true


#------------Run state of one chain
allocate_inference_state( ::AbstractInferenceMethod ) = nothing

# Called by AM after the accept/reject step (and the noisy-likelihood policy) of iteration ii, with
# the step's acceptance probability. May change the run state and the current state's log-target;
# returns true when it did.
adapt_inference_state!( inference_state, ::AbstractInferenceMethod, current, acceptance_probability, ii ) = false


#------------Results
function allocate_results( ::AbstractInferenceMethod, model, chain_length, n_stages )
    distance_chain = zeros( 0, chain_length )
    return _allocate_results_buffer( model, chain_length, n_stages, distance_chain )
end

function record_chain_step!( results, ::AbstractInferenceMethod, inference_state, current, ii )
    _record_chain_state!( results, current, ii )
    return nothing
end

# The final ε stays NaN for GSL and BSL.
finalize_results( results, ::AbstractInferenceMethod, inference_state, last_step ) = _slice_results( results, last_step )

# The parameters and log-target of the current state at step ii. current_iter is set last, so a
# crash leaves it at the last fully stored step.
function _record_chain_state!( results, current, ii )
    results.chain[:, ii] .= current.params
    results.sschain[ii] = current.log_target
    results.current_iter[] = ii
    return nothing
end

# The first `last_step` steps of the chains (after a mid-chain failure). A complete run is returned
# unchanged.
function _slice_results( results, last_step )
    chain_length = size( results.chain, 2 )
    if last_step == chain_length
        return results
    end

    partial_chain = results.chain[:, 1:last_step]
    partial_sschain = results.sschain[1:last_step]
    partial_distance_chain = results.distance_chain[:, 1:last_step]
    partial_epsilon_chain = results.epsilon_chain[1:last_step]

    sliced_results = ResultsBuffer(
        partial_chain,
        partial_sschain,
        results.prop_cov,
        Ref( last_step ),
        results.acceptance,
        results.stuck_kicks,
        partial_distance_chain,
        partial_epsilon_chain,
        results.epsilon,
        )
    return sliced_results
end
