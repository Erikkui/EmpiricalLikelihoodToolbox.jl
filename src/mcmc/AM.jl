"""
    AM

A struct for the adaptive MCMC algorithm.

# Fields
- `proposal_width::Float64`: The width of the proposal distribution (default: 0.01).
- `adaptation_interval::Int`: The interval at which to adapt the proposal distribution (default: 50).

"""
Base.@kwdef struct AM
    proposal_width::Float64 = 0.01
    adaptation_interval::Int = 50
end


function (AM::AM)( target, model, state, mcmc_options, results )
    # Initialization
    adaptation_interval = AM.adaptation_interval

    chain_length = mcmc_options.nsteps
    update_interval = mcmc_options.update_interval
    discard_noisy_updates = mcmc_options.discard_noisy_updates
    verbose = target.options.verbose

    npar = length( state.current_params )
    chain = results.chain
    sschain = results.sschain
    current_iter = results.current_iter

    am_scaling_parameter = 2.38^2 / npar
    epsilon = 1e-6

    noise_buffer = zeros( npar )
    params_proposal = zeros( npar )

    # Buffers for the Welford covariance update
    running_mean = copy(state.current_params)
    running_cov = copy(state.proposal_cov)
    diff1 = zeros(npar)
    diff2 = zeros(npar)

    # Backup buffers for "unsticking" the chain if discard_noisy_updates is true
    backup_mean = similar( running_mean )
    backup_cov  = similar( running_cov )
    backup_params = similar( state.current_params )
    backup_ss = state.ss_current
    backup_ii = 1

    # Initial Cholesky
    proposal_cov = copy( state.proposal_cov )
    proposal_cov_L = cholesky( state.proposal_cov ).L

    # Bookkeeping
    n_stuck = 0

    if discard_noisy_updates || !verbose
        print_interval = min( 50, update_interval, adaptation_interval ) - 5
        print_interval = max( print_interval, 20 ) # Ensure a reasonable print interval
        pbar = nothing
    else
        pbar = ProgressBar( 1:chain_length, printing_delay=0.25)
    end

    println( "Starting MCMC with AM algorithm for ", chain_length, " iterations..." )

    ii = 1
    while ii <= chain_length

        # 1. Propose
        # params_proposal = state.current_params + proposal_cov_L * noise_buffer
        randn!( noise_buffer )
        mul!( params_proposal, proposal_cov_L, noise_buffer )
        params_proposal .+= state.current_params

        ss_proposal = calculate_loss( params_proposal, target, model, mcmc_options )

        # 2. Metropolis accept/reject
        log_ratio = ss_proposal - state.ss_current
        rd = log( rand() )
        accepted = rd < log_ratio

        if accepted
            # println("Accepted proposal at iteration $ii with log ratio: ", log_ratio)
            # Create a backup of the last known "good" state before accepting the new proposal
            copyto!( backup_params, state.current_params )
            copyto!( backup_mean, running_mean )
            copyto!( backup_cov, running_cov )
            backup_ss = state.ss_current
            backup_ii = ii

            # Then accept the new proposal
            copyto!( state.current_params, params_proposal )
            @reset state.ss_current = ss_proposal

            # Update acceptance count
            results.acceptance[1] += 1

            n_stuck = 0 # Reset stuck counter on move
        else
            n_stuck += 1
            if n_stuck > update_interval    # If mcmc gets stuck in "too good" proposal
                if discard_noisy_updates
                    # Restore the last known "good" state
                    copyto!( state.current_params, backup_params )
                    copyto!( running_mean, backup_mean )
                    copyto!( running_cov, backup_cov )

                    ss_recalc = calculate_loss( state.current_params, target, model, mcmc_options )

                    @reset state.ss_current = ss_recalc
                    results.stuck_kicks[] += 1

                    n_stuck = 0 # Reset counter

                    # Skip storing/updating for this step and jump to the beginning
                    continue
                else
                    # Standard behavior: Just kick it to recalculate, do not "rewind time".
                    ss_recalc = calculate_loss( state.current_params, target, model, mcmc_options )

                    @reset state.ss_current = ss_recalc
                    results.stuck_kicks[] += 1

                    n_stuck = 0 # Reset counter
                end
            end
        end

        # Store
        chain[:, ii] .= state.current_params
        sschain[ii] = state.ss_current
        current_iter[] = ii

        # 3. Adaptive Update
        # Update running mean and covariance using Welford's algorithm
        recursive_welford!( running_mean, running_cov, state.current_params, diff1, diff2, npar, ii )
        if ii > 100 && ii % adaptation_interval == 0
            for col in 1:npar
                for row in 1:npar
                    proposal_cov[row, col] = running_cov[row, col] * am_scaling_parameter
                end
                # Add epsilon to diagonal without allocating a new Matrix(I)
                proposal_cov[col, col] += epsilon
            end
            proposal_cov_L = cholesky(Symmetric(proposal_cov)).L
            state.proposal_cov .= proposal_cov
        end

        # Progress display update
        if verbose
            if discard_noisy_updates && ii % 25 == 0
                progress_text = round(ii/chain_length * 100, digits=1)
                accepted_text = round.( results.acceptance ./ ii, digits=2 )
                ss_text = round(state.ss_current, digits=2)
                print("\rProgress: $progress_text% | Acc: $accepted_text | SS: $ss_text")
            elseif !isnothing(pbar)
                accepted_text = round.( results.acceptance ./ ii, digits=2 )
                ss_text = round(state.ss_current, digits=2)
                set_description(pbar, "Acc: $(accepted_text) SS: $(ss_text)")
                update(pbar)
            end
        end

        # Advance the loop
        ii += 1
    end

    println()

    return results, state
end
