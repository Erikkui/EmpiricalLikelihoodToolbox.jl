"""
    DRAM

A struct for the delayed rejection adaptive Metropolis MCMC algorithm.

# Fields
- `proposal_width::Float64`: The width of the proposal distribution (default: 0.01).
- `adaptation_interval::Int`: The interval at which to adapt the proposal distribution (default: 50).
- `n_stages::Int`: The number of proposal stages, INCLUDING the initial proposal) (default: 2).
- `proposal_scale`: The scaling factor for the proposal covariance at each stage. Can be a single number (applied as gamma^(2k+1), k = 0, ..., n_stages-2) or a vector of length n_stages specifying the scale for each stage (default: [1, 0.5]).

DRAM honours the noisy-likelihood options of `MCMCOptions` (`update_interval`,
`discard_noisy_updates`, `reevaluate_current_loss`; the re-evaluation shares its random numbers with
the first-stage proposal). It does not call the inference method's `adapt_inference_state!` hook, so ABC ε
adaptation (`EpsilonAdaptation`) is not supported: `mcmcrun` throws an `ArgumentError`.

"""
Base.@kwdef struct DRAM{P}
    proposal_width::Float64 = 0.01
    adaptation_interval::Int = 50
    n_stages::Int = 2
    proposal_scale::P = [1.0, 0.5]
end

function compute_log_path( states, log_pi, idx_range, zero_mean_dists )
    n = length( idx_range )
    idx_anchor = idx_range[1]

    # Base Case: Sequence of exactly 2 states
    if n == 2
        idx_dest = idx_range[2]
        diff = states[idx_dest] .- states[ idx_anchor ]
        log_q = logpdf( zero_mean_dists[1], diff )
        return log_pi[ idx_anchor ] + log_q
    end

    # Recursive Step
    # 1. Forward sub-path
    sub_fwd = idx_range[1:end-1]
    log_D_sub = compute_log_path( states, log_pi, sub_fwd, zero_mean_dists )

    # 2. Reverse sub-path
    sub_rev = reverse( sub_fwd )
    log_N_sub = compute_log_path( states, log_pi, sub_rev, zero_mean_dists )

    # 3. Calculate rejection probability: log(1 - alpha)
    log_alpha = min( 0.0, log_N_sub - log_D_sub )
    if log_alpha == 0.0
        return -Inf
    end
    log_rej = log1mexp( log_alpha )

    # 4. Calculate the final jump
    idx_final = idx_range[end]
    diff = states[ idx_final ] .- states[ idx_anchor ]

    log_q = logpdf( zero_mean_dists[n-1], diff )

    return log_D_sub + log_rej + log_q
end

# Propose and evaluate delayed-rejection stage `stage` (≥ 2) into chain_states[stage + 1], and return
# the log acceptance ratio of the path current → stage-1 proposal → ... → this proposal.
function delayed_rejection_stage!(
    stage,
    chain_states,
    adaptive_proposal,
    target,
    model,
    mcmc_options,
    inference_state,
    dram_buffers
    )

    # Unpack buffers
    path_params = dram_buffers.path_params
    path_log_targets = dram_buffers.path_log_targets
    mv_normals = dram_buffers.mv_normals
    gamma_values = dram_buffers.gamma_values

    # Generate and evaluate the next proposal, with the proposal covariance scaled by gamma
    current = chain_states[1]
    stage_proposal = chain_states[stage + 1]
    propose_parameters!( stage_proposal.params, adaptive_proposal, current.params; scale=gamma_values[ stage ] )
    evaluate_log_target!( stage_proposal, target, model, stage_proposal.params, mcmc_options, inference_state )

    # The states and log-targets of the path, in stage order (no copies)
    for jj in 1:(stage + 1)
        path_params[ jj ] = chain_states[ jj ].params
        path_log_targets[ jj ] = chain_states[ jj ].log_target
    end

    # Define exact index ranges
    forward_range = 1:(stage + 1)
    reverse_range = (stage + 1):-1:1

    # Calculate probabilities
    log_D_kk = compute_log_path( path_params, path_log_targets, forward_range, mv_normals )
    log_N_kk = compute_log_path( path_params, path_log_targets, reverse_range, mv_normals )

    # Final acceptance probability for this stage
    log_ratio = log_N_kk - log_D_kk
    return log_ratio
end

# path_params/path_log_targets: the states of the current delayed-rejection path, filled from the
# chain_states before each compute_log_path call.
struct DRAMBuffers{T}
    path_params::Vector{Vector{Float64}}
    path_log_targets::Vector{Float64}
    mv_normals::Vector{T}
    gamma_values::Vector{Float64}
end


# One DRAM chain. run_context.chain_states holds the current state ([1]) and one proposal per stage
# ([2:n_stages+1]); the accepted proposal swaps with the current state. Returns the results; the
# final state is run_context.chain_states[1].
function (dram::DRAM)( target, model, run_context, mcmc_options, results )
    # Initialization
    chain_states = run_context.chain_states
    adaptive_proposal = run_context.proposal
    policy = run_context.policy
    inference_state = run_context.inference_state
    method = target.options.inference_method

    n_stages = dram.n_stages
    gamma = dram.proposal_scale

    if isa( gamma, Number )
        gamma_values = vcat( 1.0, map( k -> Float64(gamma^(2*k+1)), 0:n_stages-2 ) )
    else
        gamma_values = Float64.(gamma)
    end

    chain_length = mcmc_options.nsteps
    verbose = target.options.verbose

    npar = length( chain_states[1].params )

    # Buffers for the delayed rejection
    path_params = [ zeros(npar) for _ in 0:n_stages ]
    path_log_targets = zeros( Float64, n_stages+1 )

    # Properly initialize the zero-mean distributions using Symmetric
    proposal_cov = adaptive_proposal.proposal_cov
    mv_normals = [ MvNormal(Symmetric(g * proposal_cov)) for g in gamma_values ]

    dram_buffers = DRAMBuffers(
        path_params,
        path_log_targets,
        mv_normals,
        gamma_values,
    )

    is_master_thread = Threads.nthreads() == 1 || Threads.threadid() == 1

    if verbose
        pbar = ProgressBar( 1:chain_length, printing_delay=0.25)
    else
        pbar = nothing
    end

    ii = 1
    while ii <= chain_length
        current = chain_states[1]
        first_proposal = chain_states[2]

        # 1. Propose and evaluate
        propose_parameters!( first_proposal.params, adaptive_proposal, current.params; scale=gamma_values[1] )

        step_seed = rand(UInt64)
        if policy.reevaluate_current_loss
            # Recalculate the current state's log-target with the proposal's random numbers
            evaluate_log_target!( current, target, model, current.params, mcmc_options, inference_state; rng_seed=step_seed )
        end

        evaluate_log_target!( first_proposal, target, model, first_proposal.params, mcmc_options, inference_state; rng_seed=step_seed )

        # 2. Metropolis accept/reject, then delayed rejection. accepted_stage = 0: all stages rejected
        log_ratio = first_proposal.log_target - current.log_target
        log_u = log( rand() )
        accepted_stage = log_u < log_ratio ? 1 : 0

        if accepted_stage == 0
            for kk in 2:n_stages
                log_ratio = delayed_rejection_stage!(
                    kk,
                    chain_states,
                    adaptive_proposal,
                    target,
                    model,
                    mcmc_options,
                    inference_state,
                    dram_buffers
                )

                log_u = log( rand() )
                if log_u < log_ratio
                    accepted_stage = kk
                    break
                end
            end
        end

        accepted = accepted_stage > 0
        if accepted
            # Save the last known "good" state for a rewind, then make the accepted proposal the current state
            backup_state!( policy, current, adaptive_proposal )
            accepted_index = accepted_stage + 1
            chain_states[1] = chain_states[accepted_index]
            chain_states[accepted_index] = current
            current = chain_states[1]

            results.acceptance[accepted_stage] += 1
        end

        # 3. Noisy-likelihood policy: kick the stuck chain, or rewind and repeat the step
        action = check_stuck_chain!( policy, accepted )
        repeat_step = apply_stuck_chain_action!( action, policy, current, adaptive_proposal,
                                                 target, model, mcmc_options, inference_state, results )
        if repeat_step
            continue
        end

        # 4. Store
        record_chain_step!( results, method, inference_state, current, ii )

        # 5. Adaptive update of the proposal covariance
        covariance_changed = update_proposal_covariance!( adaptive_proposal, current.params, ii )
        if covariance_changed
            # Rebuild scaled distributions
            for (idx, g) in enumerate(gamma_values)
                mv_normals[idx] = MvNormal( Symmetric(g * proposal_cov) )
            end
        end

        # Progress display update
        if !isnothing(pbar)
            accepted_text = round.( results.acceptance ./ ii, digits=2 )
            set_description(pbar, "Acc: $(accepted_text) SS: $(round(current.log_target, digits=2))")
            update(pbar)
        end

        # Advance the loop
        ii += 1
    end

    if is_master_thread
        println()
    end

    return results
end
