"""
    AM

A struct for the adaptive MCMC algorithm.

# Fields
- `proposal_width`: The variance of the initial proposal distribution (default: 0.01), the same for every parameter; or a vector of per-parameter variances (a diagonal initial proposal, for parameters on different scales).
- `adaptation_interval::Int`: The interval at which to adapt the proposal distribution (default: 50).

AM honours the noisy-likelihood options of `MCMCOptions` (`update_interval`,
`discard_noisy_updates`, `reevaluate_current_loss`). After every step it calls the inference
method's `adapt_inference_state!` hook: under ABC with `ABC( adapt_epsilon = EpsilonAdaptation( ... ) )`, this
adapts the tolerance ε during the first `n_adapt` iterations (see `EpsilonAdaptation`). With
`MCMCOptions.early_rejection`, AM skips the simulation of proposals that could not be accepted.

"""
Base.@kwdef struct AM{T}
    proposal_width::T = 0.01
    adaptation_interval::Int = 50
end

# One AM chain. run_context.chain_states holds the current state ([1]) and the proposal ([2]); they
# swap on acceptance. Returns the results; the final state is run_context.chain_states[1].
function (::AM)( target, model, run_context, mcmc_options, results )
    # Initialization
    chain_states = run_context.chain_states
    adaptive_proposal = run_context.proposal
    policy = run_context.policy
    inference_state = run_context.inference_state
    method = target.options.inference_method

    chain_length = mcmc_options.nsteps
    discard_noisy_updates = policy.discard_noisy_updates
    verbose = target.options.verbose

    # Early rejection is exact only for a noise-free log-target
    early_rejection = mcmc_options.early_rejection && mcmc_options.likelihood_noise_scale == 0

    if discard_noisy_updates || !verbose
        pbar = nothing
    else
        pbar = ProgressBar( 1:chain_length, printing_delay=0.25)
    end

    println( "Starting MCMC with AM algorithm for ", chain_length, " iterations..." )

    ii = 1
    while ii <= chain_length
        current = chain_states[1]
        proposal = chain_states[2]

        # 1. Propose and evaluate
        propose_parameters!( proposal.params, adaptive_proposal, current.params )

        step_seed = rand(UInt64)
        if policy.reevaluate_current_loss
            # Recalculate the current state's log-target with the proposal's random numbers
            evaluate_log_target!( current, target, model, current.params, mcmc_options, inference_state; rng_seed=step_seed )
        end

        # The uniform of the accept/reject step is drawn before the evaluation, so a proposal that
        # could not be accepted is not simulated (early rejection, see MCMCOptions.early_rejection)
        log_u = log( rand() )
        if early_rejection && early_rejection_allowed( method, ii )
            log_threshold = log_u + current.log_target
        else
            log_threshold = -Inf
        end

        evaluate_log_target!( proposal, target, model, proposal.params, mcmc_options, inference_state;
                              rng_seed=step_seed, log_threshold=log_threshold )

        # 2. Metropolis accept/reject
        log_ratio = proposal.log_target - current.log_target
        accepted = log_u < log_ratio

        if accepted
            # Save the last known "good" state for a rewind, then make the proposal the current state
            backup_state!( policy, current, adaptive_proposal )
            chain_states[1] = proposal
            chain_states[2] = current
            current = proposal

            results.acceptance[1] += 1
        end

        # 3. Noisy-likelihood policy: kick the stuck chain, or rewind and repeat the step
        action = check_stuck_chain!( policy, accepted )
        repeat_step = apply_stuck_chain_action!( action, policy, current, adaptive_proposal,
                                                 target, model, mcmc_options, inference_state, results )
        if repeat_step
            continue
        end

        # 4. Inference-method hook (eg. the ABC ε adaptation), then store
        step_acceptance_probability = _acceptance_probability( log_ratio )
        adapt_inference_state!( inference_state, method, current, step_acceptance_probability, ii )
        record_chain_step!( results, method, inference_state, current, ii )

        # 5. Adaptive update of the proposal covariance
        update_proposal_covariance!( adaptive_proposal, current.params, ii )

        # Progress display update
        if verbose
            if discard_noisy_updates && ii % 25 == 0
                progress_text = round(ii/chain_length * 100, digits=1)
                accepted_text = round.( results.acceptance ./ ii, digits=2 )
                ss_text = round(current.log_target, digits=2)
                print("\rProgress: $progress_text% | Acc: $accepted_text | SS: $ss_text")
            elseif !isnothing(pbar)
                accepted_text = round.( results.acceptance ./ ii, digits=2 )
                ss_text = round(current.log_target, digits=2)
                set_description(pbar, "Acc: $(accepted_text) SS: $(ss_text)")
                update(pbar)
            end
        end

        # Advance the loop
        ii += 1
    end

    println()

    return results
end
